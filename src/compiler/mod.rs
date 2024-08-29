use std::{
    collections::{BTreeMap, LinkedList},
    ffi::c_void,
    mem::offset_of,
    rc::{Rc, Weak},
};

use crate::{
    dwarf::*,
    exception::*,
    perfstats::measure_jit_codegen_time,
    runnable::{Runnable, RunnableVTable},
    unwind::*,
    Function, Operation, Parameter, Program, Scope, SourceSpan, UnpackedValue, Value, ValueKind,
};
use iced_x86::{code_asm::*, BlockEncoderOptions};
use nix::libc::{MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

mod runtime;
use runtime::*;

#[derive(Debug)]
pub enum CompiledParameter {
    Ident(String),
    Pattern {
        entries: Vec<(String, Option<Rc<Runnable<Executable>>>)>,
        binding: Option<String>,
        ignore_unmatched: bool,
    },
}

#[derive(Debug)]
pub struct ExecutableClosure {
    _strings: LinkedList<String>,
    _runnables: Vec<Rc<Runnable<Executable>>>,
    _values: Vec<Value>,
    parameter: Option<Box<CompiledParameter>>,
}

impl ExecutableClosure {
    pub fn new(parameter: Option<Box<CompiledParameter>>) -> ExecutableClosure {
        Self {
            _strings: LinkedList::new(),
            _runnables: Vec::new(),
            _values: Vec::new(),
            parameter,
        }
    }
}

trait ExecutableInternable: Sized {
    type Output;
    fn intern(self, closure: &mut ExecutableClosure) -> Self::Output;
}

impl ExecutableInternable for String {
    type Output = *const String;
    fn intern(self, closure: &mut ExecutableClosure) -> Self::Output {
        closure._strings.push_back(self);
        unsafe { closure._strings.back().unwrap_unchecked() as *const _ }
    }
}

impl ExecutableInternable for Value {
    type Output = Value;
    fn intern(self, closure: &mut ExecutableClosure) -> Self::Output {
        closure
            ._values
            .push(unsafe { Value::from_raw(self.as_raw()) });
        self
    }
}

impl ExecutableInternable for Rc<Runnable<Executable>> {
    type Output = *const Runnable<Executable>;
    fn intern(self, closure: &mut ExecutableClosure) -> Self::Output {
        let ptr = Rc::as_ptr(&self);
        closure._runnables.push(self);
        ptr
    }
}

impl ExecutableClosure {
    fn intern<T: ExecutableInternable>(&mut self, internable: T) -> T::Output {
        internable.intern(self)
    }
}

#[derive(Debug)]
pub struct Executable {
    // FIXME: this doesn't seem to work
    //        Do we have reference cycles here?
    _closure: ExecutableClosure,
    eh_frame: Box<[u8]>,
    len: usize,
    code: extern "C-unwind" fn(*const u8, *mut Scope, Value) -> Value,
    // start address -> (end address, source span) map
    source_map: BTreeMap<u64, (u64, SourceSpan)>,
}

impl Executable {
    #[inline(always)]
    pub fn run(&self, scope: *mut Scope, arg: &Value) -> Value {
        unsafe { (self.code)(std::ptr::null(), scope, Value::from_raw(arg.as_raw())) }
    }

    #[inline(always)]
    pub fn code_address_range(&self) -> (u64, u64) {
        (self.code as u64, self.code as u64 + self.len as u64)
    }

    pub fn source_span_for(&self, address: u64) -> Option<&SourceSpan> {
        self.source_map
            .range(..=address)
            .rev()
            .find_map(
                |(_, (end, span))| {
                    if *end <= address {
                        None
                    } else {
                        Some(span)
                    }
                },
            )
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        // println!("executable at {:?} dropped", self.code as *mut ());
        unsafe {
            if nix::libc::munmap(self.code as *mut c_void, self.len) < 0 {
                panic!("munmap failed");
            }
            __deregister_frame(self.eh_frame.as_ptr());
        }
    }
}

// TODO: is this macro a good idea?
macro_rules! emit_asm {
    ($assembler: expr, { $($asm: tt)* }) => {{
        emit_asm!(@phase1 $assembler; $($asm)*);
        emit_asm!(@phase2 $assembler; $($asm)*);
    }};
    (@phase1 $asm: expr;) => {};
    (@phase1 $asm: expr; $opcode: ident $($operand: expr),+; $($rest: tt)*) => {
        emit_asm!(@phase1 $asm; $($rest)*);
    };
    (@phase1 $asm: expr; $expr: block $($rest: tt)*) => {
        emit_asm!(@phase1 $asm; $($rest)*);
    };
    (@phase1 $asm: expr; $label: ident: $($rest: tt)*) => {
        let mut $label = $asm.create_label();
        emit_asm!(@phase1 $asm; $($rest)*);
    };
    (@phase2 $asm: expr;) => {};
    (@phase2 $asm: expr; $opcode: ident $($operand: expr),+; $($rest: tt)*) => {
        $asm.$opcode($($operand),*)?;
        emit_asm!(@phase2 $asm; $($rest)*);
    };
    (@phase2 $asm: expr; { $($block: tt)* } $($rest: tt)*) => {
        $($block)*
        emit_asm!(@phase2 $asm; $($rest)*);
    };
    (@phase2 $asm: expr; $label: ident: $($rest: tt)*) => {
        $asm.set_label(&mut $label)?;
        emit_asm!(@phase2 $asm; $($rest)*);
    };
}

pub struct Compiler {
    source_maps: BTreeMap<u64, Weak<Runnable<Executable>>>,
}

impl Compiler {
    pub const fn new() -> Self {
        Self {
            source_maps: BTreeMap::new(),
        }
    }

    pub fn source_span_for(&self, address: u64) -> Option<SourceSpan> {
        self.source_maps
            .range(..=address)
            .rev()
            .find_map(|(_, executable)| executable.upgrade())
            .and_then(|executable| {
                unsafe { &*executable.inner.get() }
                    .source_span_for(address)
                    .cloned()
            })
    }

    pub fn compile(
        &mut self,
        program: &Program,
        param: Option<&Parameter>,
    ) -> Result<Rc<Runnable<Executable>>, IcedError> {
        let mut _tc = measure_jit_codegen_time();
        // let debug_header = format!("{program:?}");

        let mut closure = ExecutableClosure::new(param.map(|param| {
            Box::new(match param {
                Parameter::Ident(name) => CompiledParameter::Ident(name.to_string()),
                Parameter::Pattern {
                    entries,
                    binding,
                    ignore_unmatched,
                } => CompiledParameter::Pattern {
                    entries: entries
                        .into_iter()
                        .map(|(name, default)| {
                            Ok::<(String, _), IcedError>((
                                name.to_string(),
                                default
                                    .as_ref()
                                    .map(|program| self.compile(program, None))
                                    .transpose()?,
                            ))
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                    binding: binding.clone(),
                    ignore_unmatched: *ignore_unmatched,
                },
            })
        }));
        let mut source_span_stack = vec![];
        let mut source_spans = vec![];

        let function_enter_ins: usize;
        let function_leave_ins: usize;
        let result = {
            let mut asm = CodeAssembler::new(64)?;
            asm.push(rbp)?;
            asm.mov(rbp, rsp)?;

            // callee saved registers
            asm.push(rbx)?;
            asm.push(r12)?;
            asm.push(r13)?;
            asm.push(r14)?;
            asm.push(r15)?;

            function_enter_ins = asm.instructions().len();

            // keep track of whether we need to align the stack when calling
            // and also verify the operations are valid and won't consume
            // more values on the stack than they should
            let mut stack_values = 0;

            macro_rules! call {
                (reg $reg: ident) => {
                    if stack_values % 2 == 0 {
                        asm.sub(rsp, 8)?;
                    }

                    asm.call($reg)?;

                    if stack_values % 2 == 0 {
                        asm.add(rsp, 8)?;
                    }
                };
                (rust $function: expr) => {
                    asm.mov(rax, $function as u64)?;
                    call!(reg rax)
                };
            }

            macro_rules! unpack {
                (Attrset $reg: ident, tmp = $tmp: ident, else => $else: ident) => {
                    unpack!(@pointer Attrset; $reg; $tmp; $else);
                };
                (List $reg: ident, tmp = $tmp: ident, else => $else: ident) => {
                    unpack!(@pointer List; $reg; $tmp; $else);
                };
                (String $reg: ident, tmp = $tmp: ident, else => $else: ident) => {
                    unpack!(@pointer String; $reg; $tmp; $else);
                };
                (Function $reg: ident, tmp = $tmp: ident, else => $else: ident) => {
                    unpack!(@pointer Function; $reg; $tmp; $else);
                };
                (Lazy $reg: ident, tmp = $tmp: ident, else => $else: ident) => {
                    unpack!(@pointer Lazy; $reg; $tmp; $else);
                };
                (Integer $reg: ident, tmp = $tmp: ident, else => $else: ident) => {
                    unpack!(@check Integer; $reg; $tmp; $else);
                };
                (Bool $reg: ident, tmp = $tmp: ident, else => $else: ident) => {
                    unpack!(@check Bool; $reg; $tmp; $else);
                };
                (@pointer $kind: ident; $reg: ident; $tmp: ident; $else: ident) => {{
                    asm.mov($tmp, $reg)?;
                    asm.shr($tmp, 48)?;
                    asm.cmp($tmp, ValueKind::$kind.as_shifted() as i32)?;
                    asm.jne($else)?;
                    // FIXME: Does this actually sign extend?
                    asm.shl($reg, 16)?;
                    asm.shr($reg, 16)?;
                }};
                (@check $kind: ident; $reg: ident; $tmp: ident; $else: ident) => {{
                    asm.mov($tmp, $reg)?;
                    asm.shr($tmp, 48)?;
                    asm.cmp($tmp, ValueKind::$kind.as_shifted() as i32)?;
                    asm.jne($else)?;
                }};
            }


            if let Some(ref parameter) = closure.parameter {
                let param = &**parameter as *const CompiledParameter;
                // rsi = Parent scope
                // rdx = Argument Value (if a function)

                asm.mov(rdi, param as u64)?;
                call!(rust create_function_scope);
                asm.mov(r15, rax)?;
            } else {
                // rsi = Parent scope
                asm.mov(r15, rsi)?;
            }

            for operation in program.operations.iter() {
                macro_rules! unlazy {
                ($reg: ident, tmp = $tmp: ident) => {
                    let mut _unlazy = asm.create_label();
                    unpack!(Lazy $reg, tmp = $tmp, else => _unlazy);
                    unlazy!(@mov_rdi_if_required $reg);
                    call!(rust value_into_evaluated);
                    asm.mov($reg, rax)?;
                    asm.set_label(&mut _unlazy)?;
                };
                (@mov_rdi_if_required rdi) => {};
                (@mov_rdi_if_required $src: ident) => {
                    asm.mov(rdi, $src)?;
                };
            }

            macro_rules! impl_binary_operator {
                (int => $(retag($tag: ident))? $if_int: block, fallback = $other: expr) => {{
                    asm.pop(r12)?;
                    stack_values -= 1;
                    unlazy!(r12, tmp = rdi);
                    asm.pop(rdi)?;
                    stack_values -= 1;
                    unlazy!(rdi, tmp = rcx);
                    asm.mov(rsi, r12)?;

                    let mut end = asm.create_label();

                    asm.mov(rcx, rdi)?;
                    asm.shr(rcx, 48)?;
                    asm.mov(rdx, rsi)?;
                    asm.shr(rdx, 48)?;

                    let mut not_an_integer = asm.create_label();

                    asm.cmp(ecx, ValueKind::Integer.as_shifted() as i32)?;
                    asm.jnz(not_an_integer)?;

                    asm.cmp(edx, ValueKind::Integer.as_shifted() as i32)?;
                    asm.jnz(not_an_integer)?;

                    let register = $if_int;
                    $(
                        asm.mov(r10, ValueKind::$tag.as_nan_bits())?;
                        asm.or(register, r10)?;
                    )?

                    asm.push(register)?;

                    asm.jmp(end)?;

                    asm.set_label(&mut not_an_integer)?;

                    call!(rust $other);

                    asm.push(rax)?;

                    asm.set_label(&mut end)?;
                    stack_values += 1;
                }};
                (comparison $cmov: ident or $fallback: expr) => {{
                    impl_binary_operator!(
                        int => {
                            asm.mov(rax, Value::FALSE.as_raw())?;
                            asm.mov(rbx, Value::TRUE.as_raw())?;
                            asm.cmp(rdi, rsi)?;
                            asm.$cmov(rax, rbx)?;
                            rax
                        },
                        fallback = $fallback
                    )
                }};
                (boolean $second: expr, shortcircuit(rdi = $value: expr) => $sresult: expr, else(rax) => $nresult: expr) => {emit_asm!(asm, {
                    { let second = closure.intern(self.compile($second, None)?); }

                    pop rdi;
                    {
                        stack_values -= 1;
                        unlazy!(rdi, tmp = rcx);
                    }
                    mov r12, Value::from_bool(!$value).into_raw();
                    cmp rdi, r12;
                    jne shortcircuit_possible;

                    mov rsi, r15;
                    { call!(rust unsafe { (*second).vtable().run }); }
                    mov rcx, rax;
                    shr rcx, 48;
                    cmp rcx, ValueKind::Bool.as_shifted() as i32;
                    jne second_not_boolean;

                    push $nresult;
                    jmp end;

                shortcircuit_possible:
                    xor r12, 1;
                    cmp rdi, r12;
                    jne first_not_boolean;
                    { let register = $sresult; }
                    push register;
                    jmp end;

                first_not_boolean:
                    mov rdi, c"first operand of boolean operator is not a boolean".as_ptr() as u64;
                    { call!(rust asm_throw); }

                second_not_boolean:
                    mov rdi, c"second operand of boolean operator is not a boolean".as_ptr() as u64;
                    { call!(rust asm_throw); }

                end:
                    { stack_values += 1; }
                })};
            }

                match operation {
                    Operation::Push(value) => {
                        asm.mov(rdi, closure.intern(value.clone()).into_raw())?;
                        call!(rust value_ref);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::PushFunction(param, program) => {
                        let raw = closure.intern(self.compile(&*program, Some(param))?);
                        asm.mov(rdi, raw as u64)?;
                        asm.mov(rsi, r15)?;
                        call!(rust create_function_value);
                        asm.push(rax)?;
                        stack_values += 1;
                    }

                    Operation::MapBegin { parents, rec } => {
                        emit_asm!(asm, {
                            { call!(rust map_create); }
                            mov r12, rax;
                        });

                        if *rec { emit_asm!(asm, {
                            mov rdi, rax;
                            mov rsi, r15;
                            { call!(rust map_create_recursive_scope); }
                            mov r15, rax;
                        }); }

                        for program in parents.into_iter().rev() {
                            let runnable = closure.intern(self.compile(&*program, None)?);
                            emit_asm!(asm, {
                                mov rdi, r15;
                                mov rsi, runnable as u64;
                                { call!(rust create_lazy_value); }
                                push rax;
                                { stack_values += 1; }
                            });
                        }

                        emit_asm!(asm, {
                            push r12;
                            { stack_values += 1; }
                        })
                    },

                    Operation::ConstantAttrPop(name) => {
                        let name = closure.intern(name.to_string());
                        emit_asm!(asm, {
                            pop rdx;
                            { stack_values -= 1; }
                            mov rdi, qword_ptr(rsp);
                            mov rsi, name as *const _ as u64;
                            { call!(rust map_insert_constant); }
                        });
                    },
                    Operation::ConstantAttrLazy(name, value) => {
                        let name = closure.intern(name.to_string());
                        let runnable = closure.intern(self.compile(&*value, None)?);
                        emit_asm!(asm, {
                            mov rdi, r15;
                            mov rsi, runnable as u64;
                            { call!(rust create_lazy_value); }
                            mov rdi, qword_ptr(rsp);
                            mov rsi, name as *const _ as u64;
                            mov rdx, rax;
                            { call!(rust map_insert_constant); }
                        });
                    },
                    Operation::ConstantAttrLoad(name) => {
                        let name = closure.intern(name.to_string());
                        emit_asm!(asm, {
                            mov rdi, r15;
                            mov rsi, name as *const _ as u64;
                            { call!(rust scope_lookup); }
                            mov rdi, qword_ptr(rsp);
                            mov rsi, name as *const _ as u64;
                            mov rdx, rax;
                            { call!(rust map_insert_constant); }
                        });
                    },
                    Operation::ConstantAttrInherit(name, index) => {
                        let string_ptr = Rc::as_ptr(name);
                        let name = closure.intern(UnpackedValue::String(name.clone()).pack());
                        emit_asm!(asm, {
                            mov rdi, qword_ptr(rsp + 8 * (index + 1));
                            mov rsi, name.into_raw();
                            { call!(rust create_lazy_attrset_access); }
                            mov rdi, qword_ptr(rsp);
                            mov rsi, string_ptr as *const _ as u64;
                            mov rdx, rax;
                            { call!(rust map_insert_constant); }
                        });
                    },

                    Operation::DynamicAttrPop => emit_asm!(asm, {
                        pop rdx;
                        pop rsi;
                        { stack_values -= 2; }
                        mov rdi, qword_ptr(rsp);
                        { call!(rust map_insert_dynamic); }
                    }),
                    Operation::DynamicAttrLazy(value) => {
                        let runnable = closure.intern(self.compile(&*value, None)?);
                        emit_asm!(asm, {
                            mov rdi, r15;
                            mov rsi, runnable as u64;
                            { call!(rust create_lazy_value); }
                            pop rsi;
                            { stack_values -= 1; }
                            mov rdi, qword_ptr(rsp);
                            mov rdx, rax;
                            { call!(rust map_insert_dynamic); }
                        });
                    },

                    Operation::Add => impl_binary_operator!(
                        int => retag(Integer) { asm.add(edi, esi)?; rdi },
                        fallback = Value::c_owned_add
                    ),
                    Operation::Sub => impl_binary_operator!(
                        int => retag(Integer) { asm.sub(edi, esi)?; rdi },
                        fallback = Value::c_owned_sub
                    ),
                    Operation::Mul => impl_binary_operator!(
                        int => retag(Integer) { asm.imul_2(edi, esi)?; rdi },
                        fallback = Value::c_owned_mul
                    ),
                    Operation::Div => impl_binary_operator!(
                        int => retag(Integer) {
                            asm.xor(edx, edx)?;
                            asm.mov(eax, edi)?;
                            asm.idiv(esi)?;
                            rax
                        },
                        fallback = Value::c_owned_div
                    ),
                    Operation::And(second) => impl_binary_operator!(
                        boolean second,
                        shortcircuit(rdi = false) => rdi,
                        else(rax) => rax
                    ),
                    Operation::Or(second) => impl_binary_operator!(
                        boolean second,
                        shortcircuit(rdi = true) => rdi,
                        else(rax) => rax
                    ),
                    Operation::Implication(second) => impl_binary_operator!(
                        boolean second,
                        shortcircuit(rdi = false) => {
                            asm.mov(rdi, Value::TRUE.as_raw())?;
                            rdi
                        },
                        else(rax) => rax
                    ),
                    Operation::Less => impl_binary_operator!(
                        comparison cmovl or Value::c_owned_less
                    ),
                    Operation::LessOrEqual => impl_binary_operator!(
                        comparison cmovle or Value::c_owned_less_or_equal
                    ),
                    Operation::MoreOrEqual => impl_binary_operator!(
                        comparison cmovge or Value::c_owned_greater_or_equal
                    ),
                    Operation::More => impl_binary_operator!(
                        comparison cmovg or Value::c_owned_greater
                    ),
                    Operation::Equal => impl_binary_operator!(
                        comparison cmove or Value::c_owned_equal
                    ),
                    Operation::NotEqual => impl_binary_operator!(
                        comparison cmovne or Value::c_owned_not_equal
                    ),
                    Operation::PushAttrset { parents } => {
                        asm.pop(rdi)?;
                        asm.add(rsp, (parents * 8) as i32)?;
                        stack_values -= 1 + parents;
                        call!(rust map_into_attrset);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    &Operation::GetAttrConsume { components, ref default } => {
                        let default = if let Some(program) = default {
                            closure.intern(self.compile(program, None)?)
                        } else { std::ptr::null() };

                        assert!(stack_values > components);

                        for i in 0..=components {
                            asm.mov(rdi, qword_ptr(rsp + 8 * i))?;
                            unlazy!(rdi, tmp = rcx);
                            asm.mov(qword_ptr(rsp + 8 * i), rdi)?;
                        }

                        asm.mov(rdi, rsp)?;
                        asm.mov(rsi, components as u64)?;
                        asm.mov(rdx, r15)?;
                        asm.mov(rcx, default as u64)?;

                        call!(rust attrset_get);
                        asm.add(rsp, ((components + 1) * 8) as i32)?;
                        asm.push(rax)?;

                        stack_values -= components;
                    }
                    &Operation::HasAttrpath(len) => {
                        assert!(stack_values > len);
                        asm.mov(r12, qword_ptr(rsp + 8 * len))?;
                        unlazy!(r12, tmp = rdi);

                        let mut not_an_attrset = asm.create_label();
                        let mut not_a_string = asm.create_label();
                        unpack!(Attrset r12, tmp = rcx, else => not_an_attrset);

                        for i in 0..len {
                            asm.mov(rdi, qword_ptr(rsp + 8 * i))?;
                            unlazy!(rdi, tmp = rax);
                            unpack!(String rdi, tmp = rcx, else => not_a_string);
                            asm.mov(qword_ptr(rsp + 8 * i), rdi)?;
                        }

                        asm.mov(rdi, r12)?;
                        asm.mov(rsi, rsp)?;
                        asm.mov(rdx, len as u64)?;
                        call!(rust attrset_hasattrpath);
                        stack_values -= len + 1;
                        asm.add(rsp, (8 * (len + 1)) as i32)?;
                        asm.push(rax)?;

                        let mut end = asm.create_label();
                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_attrset)?;
                        asm.mov(rdi, c"hasattr called on non-attrset value".as_ptr() as u64)?;
                        call!(rust asm_panic);

                        asm.set_label(&mut not_a_string)?;
                        asm.mov(
                            rdi,
                            c"hasattr called with non-string path component".as_ptr() as u64,
                        )?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::Apply => {
                        assert!(stack_values >= 2);
                        asm.pop(rbx)?;
                        stack_values -= 1;
                        unlazy!(rbx, tmp = rdi);
                        asm.pop(rdx)?;
                        stack_values -= 1;

                        let mut end = asm.create_label();

                        let mut not_a_function = asm.create_label();

                        // rdi (1st param) = runnable ptr
                        // rsi (2nd param) = invocation target scope
                        // rdx (3rd param) = argument
                        unpack!(Function rbx, tmp = rcx, else => not_a_function);
                        asm.mov(
                            rdi,
                            qword_ptr(
                                rbx + offset_of!(Function, runnable) as i32
                            ),
                        )?;
                        asm.mov(
                            rsi,
                            qword_ptr(
                                rbx + offset_of!(Function, parent_scope) as i32
                            )
                        )?;
                        asm.mov(
                            rax,
                            qword_ptr(
                                rdi
                                + offset_of!(Runnable, vtable) as i32
                                + offset_of!(RunnableVTable, run) as i32
                            )
                        )?;
                        call!(reg rax);
                        asm.push(rax)?;
                        asm.jmp(end)?;

                        asm.set_label(&mut not_a_function)?;

                        asm.mov(rdi, c"apply called on non-function value".as_ptr() as u64)?;
                        call!(rust asm_throw);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::Load(name) => {
                        let name = closure.intern(name.to_string());
                        asm.mov(rdi, r15)?;
                        asm.mov(rsi, name as *const _ as u64)?;
                        call!(rust scope_lookup);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::PushList => {
                        call!(rust list_create);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::ListAppend(value_program) => {
                        let raw = closure.intern(self.compile(value_program, None)?);

                        assert!(stack_values >= 1);
                        asm.mov(rdi, qword_ptr(rsp))?;

                        let mut end = asm.create_label();
                        let mut not_a_list = asm.create_label();

                        unpack!(List rdi, tmp = rax, else => not_a_list);
                        asm.mov(rsi, r15)?;
                        asm.mov(rdx, raw as u64)?;
                        call!(rust list_append_value);
                        asm.jmp(end)?;

                        asm.set_label(&mut not_a_list)?;

                        asm.mov(rdi, c"append called on non-list value".as_ptr() as u64)?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                    }
                    Operation::ScopeEnter { parents } => {
                        asm.pop(rdi)?;
                        asm.add(rsp, (parents * 8) as i32)?;
                        stack_values -= 1 + parents;
                        asm.mov(rsi, r15)?;
                        call!(rust map_into_scope);
                        asm.mov(r15, rax)?;
                    }
                    Operation::ScopeWith(program) => {
                        let namespace = closure.intern(self.compile(program, None)?);
                        asm.mov(rdi, r15)?;
                        asm.mov(rsi, namespace as u64)?;
                        call!(rust scope_create_with);
                        asm.mov(r15, rax)?;
                    }
                    Operation::ScopeLeave => {
                        // FIXME: make scopes rc or smth
                        asm.mov(r15, qword_ptr(r15 + offset_of!(Scope, previous)))?;
                    }
                    Operation::IfElse(if_true, if_false) => {
                        // TODO: inline these into this executable instead
                        let if_true = closure.intern(self.compile(if_true, None)?);
                        let if_false = closure.intern(self.compile(if_false, None)?);

                        assert!(stack_values >= 1);
                        asm.pop(rsi)?;
                        stack_values -= 1;

                        unlazy!(rsi, tmp = rdi);

                        let mut end = asm.create_label();
                        let mut not_a_boolean = asm.create_label();
                        unpack!(Bool rsi, tmp = rdx, else => not_a_boolean);

                        unsafe {
                            asm.mov(rax, (*if_false).vtable().run as u64)?;
                            asm.mov(rbx, (*if_true).vtable().run as u64)?;
                        }
                        asm.cmp(esi, 0)?;
                        asm.cmovne(rax, rbx)?;

                        asm.mov(rsi, r15)?;
                        call!(reg rax);
                        asm.push(rax)?;
                        asm.jmp(end)?;

                        asm.set_label(&mut not_a_boolean)?;
                        asm.mov(
                            rdi,
                            c"if attempted to branch on non-boolean value".as_ptr() as u64,
                        )?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::Negate => {
                        assert!(stack_values >= 1);
                        asm.pop(rdi)?;
                        stack_values -= 1;
                        unlazy!(rdi, tmp = rcx);

                        let mut end = asm.create_label();
                        let mut not_an_integer = asm.create_label();

                        unpack!(Integer rdi, tmp = rax, else => not_an_integer);
                        asm.neg(edi)?;
                        asm.mov(r10, ValueKind::Integer.as_nan_bits())?;
                        asm.or(rdi, r10)?;

                        asm.push(rdi)?;
                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_integer)?;
                        asm.mov(
                            rdi,
                            c"negate attempted on non-integer value".as_ptr() as u64,
                        )?;
                        call!(rust asm_throw);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::Invert => {
                        assert!(stack_values >= 1);
                        asm.pop(rdi)?;
                        stack_values -= 1;

                        let mut end = asm.create_label();
                        let mut not_a_boolean = asm.create_label();

                        unpack!(Bool rdi, tmp = rax, else => not_a_boolean);
                        asm.xor(rdi, 0b1)?;
                        asm.push(rdi)?;
                        asm.jmp(end)?;

                        asm.set_label(&mut not_a_boolean)?;
                        asm.mov(
                            rdi,
                            c"invert attempted on non-boolean value".as_ptr() as u64,
                        )?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::Concat => {
                        assert!(stack_values >= 2);
                        asm.pop(r12)?;
                        stack_values -= 1;
                        unlazy!(r12, tmp = rdi);
                        asm.pop(rdi)?;
                        stack_values -= 1;
                        unlazy!(rdi, tmp = rcx);
                        asm.mov(rsi, r12)?;

                        call!(rust list_concat);

                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::Update => {
                        assert!(stack_values >= 2);
                        asm.pop(r12)?;
                        stack_values -= 1;
                        unlazy!(r12, tmp = rdi);
                        asm.pop(rdi)?;
                        stack_values -= 1;
                        unlazy!(rdi, tmp = rcx);
                        asm.mov(rsi, r12)?;

                        let mut end = asm.create_label();
                        let mut not_an_attrset = asm.create_label();
                        unpack!(Attrset rdi, tmp = rcx, else => not_an_attrset);
                        unpack!(Attrset rsi, tmp = rcx, else => not_an_attrset);
                        call!(rust attrset_update);
                        asm.push(rax)?;

                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_attrset)?;
                        asm.mov(
                            rdi,
                            c"update attempted on non-attrset values".as_ptr() as u64,
                        )?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::StringMutAppend => {
                        assert!(stack_values >= 2);
                        asm.pop(rdi)?;
                        stack_values -= 1;
                        unlazy!(rdi, tmp = rcx);

                        let mut end = asm.create_label();
                        let mut not_a_string = asm.create_label();

                        unpack!(String rdi, tmp = rcx, else => not_a_string);
                        asm.mov(rsi, qword_ptr(rsp))?;
                        asm.shl(rsi, 16)?;
                        asm.shr(rsi, 16)?;

                        call!(rust string_mut_append);
                        asm.jmp(end)?;

                        asm.set_label(&mut not_a_string)?;
                        asm.mov(
                            rdi,
                            c"string append attempted with non-string value".as_ptr() as u64,
                        )?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                    }
                    Operation::StringCloneToMut => {
                        assert!(stack_values >= 1);
                        asm.mov(rdi, qword_ptr(rsp))?;
                        call!(rust value_string_to_mut);
                        asm.mov(qword_ptr(rsp), rax)?;
                    }
                    Operation::StringToPath => {
                        assert!(stack_values >= 1);
                        asm.mov(rdi, qword_ptr(rsp))?;
                        call!(rust value_string_to_path);
                        asm.mov(qword_ptr(rsp), rax)?;
                    }
                    Operation::SourceSpanPush(span) => {
                        source_span_stack.push((
                            asm.instructions().len(),
                            span.clone()
                        ))
                    }
                    Operation::SourceSpanPop => {
                        let (start, span) = source_span_stack.pop().unwrap();
                        source_spans.push((
                            start,
                            asm.instructions().len(),
                            span
                        ));
                    }
                }
            }

            assert_eq!(stack_values, 1);
            asm.pop(rax)?;

            // callee saved registers
            asm.pop(r15)?;
            asm.pop(r14)?;
            asm.pop(r13)?;
            asm.pop(r12)?;
            asm.pop(rbx)?;

            asm.leave()?;
            function_leave_ins = asm.instructions().len();
            asm.ret()?;

            asm.assemble_options(
                0,
                BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS
            )?
        }.inner;

        // eprintln!("{}", debug_header);
        // let decoder = iced_x86::Decoder::new(64, &result.code_buffer, 0);
        // let mut formatter = iced_x86::IntelFormatter::new();
        // let mut output = String::new();
        // for instruction in decoder {
        //     output.clear();
        //     iced_x86::Formatter::format(&mut formatter, &instruction, &mut output);
        //     eprintln!("\t{output}")
        // }

        unsafe {
            let code = &result.code_buffer;
            let code_mem = nix::libc::mmap(
                std::ptr::null_mut(),
                code.len(),
                PROT_READ | PROT_WRITE,
                MAP_ANONYMOUS | MAP_PRIVATE,
                -1,
                0,
            );
            if code_mem == MAP_FAILED {
                panic!("mmap failed");
            }
            (code_mem as *mut u8).copy_from(code.as_ptr(), code.len());
            if nix::libc::mprotect(code_mem, code.len(), PROT_EXEC | PROT_READ) < 0 {
                panic!("mprotect failed");
            }

            // FIXME: still broken
            #[allow(unused_assignments)]
            let frame = EhFrameBuilder::new(
                1,
                -8,
                // TODO: The CIE could be shared
                jit_personality,
                |cie| {
                    DW_CFA_advance_loc(
                        cie,
                        result.new_instruction_offsets[function_enter_ins] as u8,
                    );
                    DW_CFA_def_cfa(cie, 7, 8);
                    DW_CFA_offset(cie, 16, 1);
                },
            )
            .add_fde(code_mem as u64, code.len() as u64, |out: &mut Vec<u8>| {
                let mut ip = 0u64;
                macro_rules! advance_to {
                    ($fun: ident[$width: ident], $x: expr) => {{
                        let _x: u64 = $x;
                        let advance = _x - ip;
                        $fun(out, advance as $width);
                        ip += advance;
                    }};
                }

                advance_to!(
                    DW_CFA_advance_loc[u8],
                    result.new_instruction_offsets[function_enter_ins].into()
                );
                DW_CFA_def_cfa_offset(out, 16);
                DW_CFA_offset(out, /* rbp */ 6, 2);

                DW_CFA_offset(out, 3, 3);
                DW_CFA_offset(out, 12, 4);
                DW_CFA_offset(out, 13, 5);
                DW_CFA_offset(out, 14, 6);

                DW_CFA_def_cfa_register(out, /* rbp */ 6);
                advance_to!(
                    DW_CFA_advance_loc4[u32],
                    result.new_instruction_offsets[function_leave_ins].into()
                );
                DW_CFA_def_cfa(out, /* rsp */ 7, 8);
                DW_CFA_same_value(out, 3);
                DW_CFA_same_value(out, 12);
                DW_CFA_same_value(out, 13);
                DW_CFA_same_value(out, 14);
            })
            .build()
            .into_boxed_slice();
            // println!("{:?}", frame.as_ptr());
            // walk_eh_frame_fdes(&frame[..], |offset| {
            //     println!("registering fde at offset {offset}");
            //     __register_frame(frame.as_ptr().add(offset));
            // });
            __register_frame(frame.as_ptr());

            let source_map = source_spans
                .into_iter()
                .map(|(start_ins, end_ins, span)| {
                    (
                        code_mem as u64 + result.new_instruction_offsets[start_ins] as u64,
                        (
                            code_mem as u64 + result.new_instruction_offsets[end_ins] as u64,
                            span,
                        ),
                    )
                })
                .collect();

            // eprintln!("new executable at 0x{:x}", code_mem as u64);
            // for (start, (end, span)) in (&source_map as &BTreeMap<u64, (u64, SourceSpan)>).iter() {
            //     eprintln!("0x{start:x}-0x{end:x} -> {span:?}")
            // }

            let rc = Rc::new(Runnable::new(
                RunnableVTable {
                    run: std::mem::transmute(code_mem),
                    drop_in_place: |this| {
                        std::ptr::drop_in_place((*Runnable::upcast::<Executable>(this)).inner.get())
                    },
                },
                Executable {
                    _closure: closure,
                    eh_frame: frame,
                    // TODO: Use RunnableVTable.run instead
                    code: std::mem::transmute(code_mem),
                    len: code.len(),
                    source_map,
                },
            ));
            self.source_maps.insert(code_mem as u64, Rc::downgrade(&rc));
            Ok(rc)
        }
    }
}

pub static mut COMPILER: Compiler = Compiler::new();

unsafe extern "C" fn jit_personality(
    version: std::ffi::c_int,
    actions: std::ffi::c_int,
    exception_class: u64,
    exception: *const _Unwind_Exception,
    context: *const _Unwind_Context,
) -> _Unwind_Reason_Code {
    assert_eq!(version, 1);
    if actions & _UA_SEARCH_PHASE > 0 {
        let ip = _Unwind_GetIP(context);
        let span = COMPILER.source_span_for(ip);

        if exception_class == NIX_EXCEPTION_CLASS {
            let exception = &mut *(exception as *mut NixException);
            if let Some(span) = span {
                exception.stacktrace.push(span);
            } else {
                eprintln!("WARNING: jit_personality: span not found for ip 0x{:x}", ip)
            }
        }
    }

    if actions & _UA_CLEANUP_PHASE > 0 {
        // TODO: Cleanup stack
        //       This requires saving more information in unwind data.
        //       (we need to know when the stack pointer was when the unwind happened)
    }

    _URC_CONTINUE_UNWIND
}
