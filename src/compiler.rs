use std::{
    arch::asm,
    collections::BTreeMap,
    ffi::c_void,
    mem::offset_of,
    rc::{Rc, Weak},
};

use crate::{
    dwarf::*, unwind::*, exception::*, Function, Operation, Parameter, Program, Scope, SourceSpan, Value,
    ValueKind, ATTRSET_TAG_LAZY, ATTRSET_TAG_MASK, VALUE_TAG_MASK, VALUE_TAG_WIDTH,
};
use iced_x86::{code_asm::*, BlockEncoderOptions};
use nix::libc::{MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

mod runtime;
use runtime::*;

#[derive(Debug)]
pub enum CompiledParameter {
    Ident(String),
    Pattern {
        entries: Vec<(String, Option<Rc<Executable>>)>,
        binding: Option<String>,
        ignore_unmatched: bool,
    },
}

#[derive(Debug)]
pub struct ExecutableClosure {
    _strings: Vec<String>,
    _executables: Vec<Rc<Executable>>,
    _values: Vec<Value>,
    parameter: Option<Box<CompiledParameter>>,
}

impl ExecutableClosure {
    pub fn new(parameter: Option<Box<CompiledParameter>>) -> ExecutableClosure {
        Self {
            _strings: Vec::new(),
            _executables: Vec::new(),
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
    // NOTE: This lifetime is not 'static, it has just been erased
    //       Doing this "properly" is not worth it at all.
    type Output = &'static str;
    fn intern(self, closure: &mut ExecutableClosure) -> Self::Output {
        let erased = unsafe { &*(self.as_str() as *const str) };
        closure._strings.push(self);
        erased
    }
}

impl ExecutableInternable for Value {
    type Output = Value;
    fn intern(self, closure: &mut ExecutableClosure) -> Self::Output {
        closure._values.push(unsafe { self.leaking_copy() });
        self
    }
}

impl ExecutableInternable for Rc<Executable> {
    type Output = *const Executable;
    fn intern(self, closure: &mut ExecutableClosure) -> Self::Output {
        let ptr = Rc::as_ptr(&self);
        closure._executables.push(self);
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
    code: extern "C-unwind" fn(Value, Value) -> Value,
    // start address -> (end address, source span) map
    source_map: BTreeMap<u64, (u64, SourceSpan)>,
}

impl Executable {
    #[inline(always)]
    pub fn run(&self, scope: *mut Scope, arg: &Value) -> Value {
        // println!("executable at {:?} running", self.code as *const ());
        let r = unsafe {
            asm!("mov r10, {}", in(reg) scope, out("r10") _);
            (self.code)(Value::NULL, arg.leaking_copy())
        };
        // println!("executable at {:?} ran", self.code as *const ());
        r
    }

    #[inline(always)]
    pub fn code(&self) -> unsafe extern "C-unwind" fn(Value, Value) -> Value {
        self.code
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

pub struct Compiler {
    source_maps: BTreeMap<u64, Weak<Executable>>,
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
            .and_then(|executable| executable.source_span_for(address).cloned())
    }

    pub fn compile(
        &mut self,
        program: Program,
        param: Option<Parameter>,
    ) -> Result<Rc<Executable>, IcedError> {
        let debug_header = format!("{program:?}");

        let mut closure = ExecutableClosure::new(param.map(|param| {
            Box::new(match param {
                Parameter::Ident(name) => CompiledParameter::Ident(name),
                Parameter::Pattern {
                    entries,
                    binding,
                    ignore_unmatched,
                } => CompiledParameter::Pattern {
                    entries: entries
                        .into_iter()
                        .map(|(name, default)| {
                            Ok::<(String, Option<Rc<Executable>>), IcedError>((
                                name,
                                default
                                    .map(|program| self.compile(program, None))
                                    .transpose()?,
                            ))
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                    binding,
                    ignore_unmatched,
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

            if let Some(ref parameter) = closure.parameter {
                let param = &**parameter as *const CompiledParameter;
                // rdi = Current function Value (if a function)
                // rsi = Argument Value (if a function)

                let mut not_a_function = asm.create_label();
                let mut end = asm.create_label();

                // SANITY CHECK: Make sure rdi is a function
                asm.mov(rax, rdi)?;
                asm.and(rax, VALUE_TAG_MASK as i32)?;
                asm.cmp(al, ValueKind::Function as i32)?;
                asm.jne(not_a_function)?;

                asm.mov(
                    rdi,
                    qword_ptr(
                        rdi + offset_of!(Function, parent_scope) - ValueKind::Function as i32,
                    ),
                )?;
                asm.mov(rdx, param as u64)?;

                call!(rust create_function_scope);
                asm.mov(r15, rax)?;
                asm.jmp(end)?;

                asm.set_label(&mut not_a_function)?;
                asm.mov(rsi, rdi)?;
                asm.mov(rdi, c"INTERNAL ERROR: Function receiver is not a Function".as_ptr() as u64)?;
                call!(rust asm_panic_with_value);

                asm.set_label(&mut end)?;
            } else {
                // r10 contains our scope (set by the caller)
                asm.mov(r15, r10)?;
            }

            // These values are always kept here
            asm.mov(r13, ATTRSET_TAG_LAZY)?;
            asm.mov(r14, ATTRSET_TAG_MASK)?;

            for operation in program.operations {
                macro_rules! unlazy {
                ($reg: ident, rdi) => {
                    let mut _unlazy = asm.create_label();
                    asm.mov(rdi, $reg)?;
                    asm.and(rdi, r14)?;
                    asm.cmp(rdi, r13)?;
                    asm.jne(_unlazy)?;
                    asm.mov(rdi, $reg)?;
                    call!(rust value_into_evaluated);
                    asm.mov($reg, rax)?;
                    asm.set_label(&mut _unlazy)?;
                };
                (rdi, $scratch: ident) => {
                    let mut _unlazy = asm.create_label();
                    asm.mov($scratch, rdi)?;
                    asm.and($scratch, r14)?;
                    asm.cmp($scratch, r13)?;
                    asm.jne(_unlazy)?;
                    call!(rust value_into_evaluated);
                    asm.mov(rdi, rax)?;
                    asm.set_label(&mut _unlazy)?;
                };
            }

                macro_rules! impl_binary_operator {
                ($(int => $(retag($tag: ident))? $if_int: block,)? $(bool => $if_bool: expr,)? fallback = $other: expr) => {{
                    asm.pop(r12)?;
                    stack_values -= 1;
                    unlazy!(r12, rdi);
                    asm.pop(rdi)?;
                    stack_values -= 1;
                    unlazy!(rdi, rcx);
                    asm.mov(rsi, r12)?;

                    let mut end = asm.create_label();

                    asm.mov(rcx, rdi)?;
                    asm.and(rcx, VALUE_TAG_MASK as i32)?;
                    asm.mov(rdx, rsi)?;
                    asm.and(rdx, VALUE_TAG_MASK as i32)?;

                    $(
                        let mut not_an_integer = asm.create_label();

                        asm.cmp(cl, ValueKind::Integer as i32)?;
                        asm.jnz(not_an_integer)?;

                        asm.cmp(dl, ValueKind::Integer as i32)?;
                        asm.jnz(not_an_integer)?;

                        asm.shr(rdi, VALUE_TAG_WIDTH as i32)?;
                        asm.shr(rsi, VALUE_TAG_WIDTH as i32)?;
                        let register = $if_int;
                        $(
                            asm.shl(register, VALUE_TAG_WIDTH as i32)?;
                            asm.or(register, ValueKind::$tag as i32)?;
                        )?

                        asm.push(register)?;

                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_integer)?;
                    )?

                    $(
                        let mut not_a_boolean = asm.create_label();

                        asm.cmp(cl, ValueKind::Bool as i32)?;
                        asm.jnz(not_a_boolean)?;

                        asm.cmp(dl, ValueKind::Bool as i32)?;
                        asm.jnz(not_a_boolean)?;

                        $if_bool;

                        asm.push(rdi)?;

                        asm.jmp(end)?;

                        asm.set_label(&mut not_a_boolean)?;
                    )?

                    call!(rust $other);

                    asm.push(rax)?;

                    asm.set_label(&mut end)?;
                    stack_values += 1;
                }};
                (comparison $cmov: ident or $fallback: expr) => {{
                    impl_binary_operator!(
                        int => {
                            asm.mov(rax, Value::FALSE.0)?;
                            asm.mov(rbx, Value::TRUE.0)?;
                            asm.cmp(rdi, rsi)?;
                            asm.$cmov(rax, rbx)?;
                            rax
                        },
                        fallback = $fallback
                    )
                }};
            }

                macro_rules! unpack {
                    (Attrset $reg: ident tmp = $tmp: ident else => $else: ident) => {{
                        asm.mov($tmp, $reg)?;
                        asm.and($tmp, r14)?;
                        asm.cmp($tmp, ValueKind::Attrset as i32)?;
                        asm.jne($else)?;
                        asm.sub($reg, ValueKind::Attrset as i32)?;
                    }};
                }

                match operation {
                    Operation::Push(value) => {
                        asm.mov(rdi, closure.intern(value).0)?;
                        call!(rust value_ref);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::PushFunction(param, program) => {
                        let raw = closure.intern(self.compile(program, Some(param))?);
                        asm.mov(rdi, raw as u64)?;
                        asm.mov(rsi, r15)?;
                        call!(rust create_function_value);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::Add => impl_binary_operator!(
                        int => retag(Integer) { asm.add(edi, esi)?; rdi },
                        fallback = Value::add
                    ),
                    Operation::Sub => impl_binary_operator!(
                        int => retag(Integer) { asm.sub(edi, esi)?; rdi },
                        fallback = Value::sub
                    ),
                    Operation::Mul => impl_binary_operator!(
                        int => retag(Integer) { asm.imul_2(edi, esi)?; rdi },
                        fallback = Value::mul
                    ),
                    Operation::Div => impl_binary_operator!(
                        int => retag(Integer) {
                            asm.xor(edx, edx)?;
                            asm.mov(eax, edi)?;
                            asm.idiv(esi)?;
                            rax
                        },
                        fallback = Value::div
                    ),
                    Operation::And => impl_binary_operator!(
                        bool => asm.and(rdi, rsi)?,
                        fallback = Value::and
                    ),
                    Operation::Or => impl_binary_operator!(
                        bool => asm.or(rdi, rsi)?,
                        fallback = Value::or
                    ),
                    Operation::Implication => impl_binary_operator!(
                        bool => {
                            asm.cmp(rdi, Value::TRUE.0 as i32)?;
                            asm.mov(rdi, Value::TRUE.0)?;
                            asm.cmove(rdi, rsi)?;
                        },
                        fallback = Value::implication
                    ),
                    Operation::Less => impl_binary_operator!(
                        comparison cmovl or Value::less
                    ),
                    Operation::LessOrEqual => impl_binary_operator!(
                        comparison cmovle or Value::less_or_equal
                    ),
                    Operation::MoreOrEqual => impl_binary_operator!(
                        comparison cmovge or Value::greater_or_equal
                    ),
                    Operation::More => impl_binary_operator!(
                        comparison cmovg or Value::greater
                    ),
                    Operation::Equal => impl_binary_operator!(
                        comparison cmove or Value::equal
                    ),
                    Operation::NotEqual => impl_binary_operator!(
                        comparison cmovne or Value::not_equal
                    ),
                    Operation::CreateAttrset { rec } => {
                        call!(rust attrset_create);
                        asm.push(rax)?;
                        stack_values += 1;

                        if rec {
                            asm.mov(rdi, rax)?;
                            asm.sub(rdi, ValueKind::Attrset as i32)?;
                            asm.mov(rsi, r15)?;
                            call!(rust scope_create_rec);
                            asm.mov(r15, rax)?;
                        }
                    }
                    Operation::SetAttrpath(components, value_program) => {
                        let raw = closure.intern(self.compile(value_program, None)?);

                        assert!(stack_values > components);
                        asm.mov(rdi, qword_ptr(rsp + (8 * components)))?;

                        let mut not_an_attrset = asm.create_label();
                        for _ in 0..(components - 1) {
                            asm.mov(rbx, rdi)?;
                            asm.and(rbx, r14)?;
                            asm.cmp(rbx, ValueKind::Attrset as i32)?;
                            asm.jne(not_an_attrset)?;

                            asm.sub(rdi, ValueKind::Attrset as i32)?;

                            asm.pop(rsi)?;
                            stack_values -= 1;
                            // FIXME: THIS IS VERY WRONG!!!
                            // nix-repl> x = { a = 1; }
                            //
                            // nix-repl> { a = x; a.l = 10; }
                            // error: attribute 'a.l' already defined at «string»:1:3
                            //
                            //        at «string»:1:10:
                            //
                            //             1| { a = x; a.l = 10;
                            call!(rust attrset_get_or_insert_attrset);
                            asm.mov(rdi, rax)?;
                        }

                        asm.mov(rbx, rdi)?;
                        asm.and(rbx, r14)?;
                        asm.cmp(rbx, ValueKind::Attrset as i32)?;
                        asm.jne(not_an_attrset)?;

                        asm.sub(rdi, ValueKind::Attrset as i32)?;
                        asm.mov(rsi, r15)?;
                        asm.mov(rdx, raw as u64)?;
                        asm.pop(rcx)?;
                        stack_values -= 1;

                        call!(rust attrset_set);

                        let mut end = asm.create_label();
                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_attrset)?;

                        asm.mov(rdi, c"setattr called on non-attrset value".as_ptr() as u64)?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                    }
                    Operation::InheritAttrs(source, names) => {
                        let names = Vec::leak(names);
                        let mut not_an_attrset = asm.create_label();

                        assert!(stack_values >= 1);
                        asm.mov(rdi, qword_ptr(rsp))?;

                        asm.mov(rbx, rdi)?;
                        asm.and(rbx, r14)?;
                        asm.cmp(rbx, ValueKind::Attrset as i32)?;
                        asm.jne(not_an_attrset)?;
                        asm.sub(rdi, ValueKind::Attrset as i32)?;

                        let mut end = asm.create_label();
                        if let Some(program) = source {
                            asm.mov(rsi, rdi)?;
                            asm.mov(rdi, r15)?;
                            asm.mov(rdx, closure.intern(self.compile(program, None)?) as u64)?;
                            asm.mov(rcx, names.as_ptr() as u64)?;
                            asm.mov(r8, names.len() as u64)?;
                            call!(rust map_inherit_from);
                        } else {
                            asm.mov(rsi, r15)?;
                            asm.mov(rdx, names.as_ptr() as u64)?;
                            asm.mov(rcx, names.len() as u64)?;
                            call!(rust attrset_inherit_parent);
                        };

                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_attrset)?;

                        asm.mov(
                            rdi,
                            c"inheritattr called on non-attrset value or non-attrset from".as_ptr()
                                as u64,
                        )?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                    }
                    Operation::GetAttrConsume { default } => {
                        let default = if let Some(program) = default {
                            closure.intern(self.compile(program, None)?)
                        } else { std::ptr::null() };

                        assert!(stack_values >= 2);
                        asm.pop(r12)?;
                        stack_values -= 1;
                        unlazy!(r12, rdi);
                        asm.pop(rdi)?;
                        stack_values -= 1;
                        unlazy!(rdi, rcx);
                        asm.mov(rsi, r12)?;

                        let mut not_an_attrset = asm.create_label();
                        asm.mov(rcx, rdi)?;
                        asm.and(rcx, r14)?;
                        asm.cmp(rcx, ValueKind::Attrset as i32)?;
                        asm.jne(not_an_attrset)?;

                        asm.sub(rdi, ValueKind::Attrset as i32)?;

                        asm.mov(rdx, r15)?;
                        asm.mov(rcx, default as u64)?;

                        call!(rust attrset_get);
                        asm.push(rax)?;

                        let mut end = asm.create_label();
                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_attrset)?;

                        asm.mov(rsi, rdi)?;
                        asm.mov(rdi, c"getattr called on non-attrset value".as_ptr() as u64)?;
                        call!(rust asm_panic_with_value);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::HasAttrpath(len) => {
                        assert!(stack_values > len);
                        asm.mov(r12, qword_ptr(rsp + 8 * len))?;
                        unlazy!(r12, rdi);

                        let mut not_an_attrset = asm.create_label();
                        let mut not_a_string = asm.create_label();
                        asm.mov(rcx, r12)?;
                        asm.and(rcx, r14)?;
                        asm.cmp(rcx, ValueKind::Attrset as i32)?;
                        asm.jne(not_an_attrset)?;
                        asm.sub(r12, ValueKind::Attrset as i32)?;

                        for i in 0..len {
                            asm.mov(rdi, qword_ptr(rsp + 8 * i))?;
                            unlazy!(rdi, rax);
                            asm.mov(rcx, rdi)?;
                            asm.and(rcx, VALUE_TAG_MASK as i32)?;
                            asm.cmp(rcx, ValueKind::String as i32)?;
                            asm.jne(not_a_string)?;
                            asm.sub(rdi, ValueKind::String as i32)?;
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
                        unlazy!(rbx, rdi);
                        asm.pop(rsi)?;
                        stack_values -= 1;

                        let mut end = asm.create_label();

                        let mut not_a_function = asm.create_label();

                        asm.mov(rcx, rbx)?;
                        asm.and(rcx, VALUE_TAG_MASK as i32)?;
                        asm.cmp(cl, ValueKind::Function as u32)?;
                        asm.jne(not_a_function)?;

                        asm.mov(rdi, rbx)?;
                        asm.mov(
                            rax,
                            qword_ptr(
                                rbx + offset_of!(Function, call) as i32
                                    - ValueKind::Function as i32,
                            ),
                        )?;
                        call!(reg rax);
                        asm.push(rax)?;
                        asm.jmp(end)?;

                        asm.set_label(&mut not_a_function)?;

                        asm.mov(rdi, c"apply called on non-function value".as_ptr() as u64)?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::Load(name) => {
                        let name = closure.intern(name);
                        asm.mov(rdi, r15)?;
                        asm.mov(rsi, name.as_ptr() as u64)?;
                        asm.mov(rdx, name.len() as u64)?;
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

                        asm.mov(rax, rdi)?;
                        asm.and(rax, VALUE_TAG_MASK as i32)?;
                        asm.cmp(al, ValueKind::List as u32)?;
                        asm.jne(not_a_list)?;

                        asm.sub(rdi, ValueKind::List as i32)?;
                        asm.mov(rsi, r15)?;
                        asm.mov(rdx, raw as u64)?;
                        call!(rust list_append_value);
                        asm.jmp(end)?;

                        asm.set_label(&mut not_a_list)?;

                        asm.mov(rdi, c"append called on non-list value".as_ptr() as u64)?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                    }
                    Operation::ScopePush => {
                        asm.mov(rdi, r15)?;
                        call!(rust scope_create);
                        asm.mov(r15, rax)?;
                    }
                    Operation::ScopeSet(name, value_program) => {
                        let name = closure.intern(name);
                        let raw = closure.intern(self.compile(value_program, None)?);
                        asm.mov(rdi, r15)?;
                        asm.mov(rsi, name.as_ptr() as u64)?;
                        asm.mov(rdx, name.len() as u64)?;
                        asm.mov(rcx, raw as u64)?;
                        call!(rust scope_set);
                    }
                    Operation::ScopeInherit(from, names) => {
                        let names = Vec::leak(names);
                        if let Some(program) = from {
                            asm.mov(rdi, r15)?;
                            asm.mov(rsi, qword_ptr(r15 + offset_of!(Scope, values)))?;
                            asm.mov(rdx, closure.intern(self.compile(program, None)?) as u64)?;
                            asm.mov(rcx, names.as_ptr() as u64)?;
                            asm.mov(r8, names.len() as u64)?;
                            call!(rust map_inherit_from);
                        } else {
                            asm.mov(rsi, qword_ptr(r15 + offset_of!(Scope, previous)))?;
                            asm.mov(rdi, r15)?;
                            asm.mov(rdx, names.as_ptr() as u64)?;
                            asm.mov(rcx, names.len() as u64)?;
                            call!(rust scope_inherit_parent);
                        };
                    }
                    Operation::ScopePop => {
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

                        unlazy!(rsi, rdi);

                        let mut end = asm.create_label();
                        let mut not_a_boolean = asm.create_label();
                        asm.mov(rdx, rsi)?;
                        asm.and(rdx, VALUE_TAG_MASK as i32)?;
                        asm.cmp(dl, ValueKind::Bool as i32)?;
                        asm.jne(not_a_boolean)?;

                        unsafe {
                            asm.mov(rax, (*if_false).code() as u64)?;
                            asm.mov(rbx, (*if_true).code() as u64)?;
                        }
                        asm.cmp(rsi, ValueKind::Bool as i32)?;
                        asm.cmovne(rax, rbx)?;

                        asm.mov(r10, r15)?;
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
                        unlazy!(rdi, rcx);

                        let mut end = asm.create_label();
                        let mut not_an_integer = asm.create_label();

                        asm.mov(rax, rdi)?;
                        asm.and(rax, VALUE_TAG_MASK as i32)?;
                        asm.cmp(rax, ValueKind::Integer as i32)?;
                        asm.jne(not_an_integer)?;

                        asm.shr(rdi, 3)?;
                        asm.neg(edi)?;
                        asm.shl(rdi, 3)?;
                        asm.or(rdi, ValueKind::Integer as i32)?;

                        asm.push(rdi)?;
                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_integer)?;
                        asm.mov(
                            rdi,
                            c"negate attempted on non-integer value".as_ptr() as u64,
                        )?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::Invert => {
                        assert!(stack_values >= 1);
                        asm.pop(rdi)?;
                        stack_values -= 1;

                        let mut end = asm.create_label();
                        let mut not_an_integer = asm.create_label();
                        let mut not_a_boolean = asm.create_label();

                        asm.mov(rax, rdi)?;
                        asm.and(rax, VALUE_TAG_MASK as i32)?;
                        asm.cmp(rax, ValueKind::Integer as i32)?;
                        asm.jne(not_an_integer)?;

                        asm.shr(rax, 3)?;
                        asm.not(rax)?;
                        asm.shl(rax, 3)?;
                        asm.or(rax, ValueKind::Integer as i32)?;

                        asm.push(rax)?;
                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_integer)?;

                        asm.cmp(rax, ValueKind::Bool as i32)?;
                        asm.jne(not_a_boolean)?;

                        asm.xor(rdi, 0b1000)?;
                        asm.push(rdi)?;
                        asm.jmp(end)?;

                        asm.set_label(&mut not_a_boolean)?;
                        asm.mov(
                            rdi,
                            c"invert attempted on non-integer value".as_ptr() as u64,
                        )?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::Concat => {
                        assert!(stack_values >= 2);
                        asm.pop(rdi)?;
                        asm.pop(rsi)?;
                        stack_values -= 2;

                        call!(rust list_concat);

                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::Update => {
                        assert!(stack_values >= 2);
                        asm.pop(r12)?;
                        stack_values -= 1;
                        unlazy!(r12, rdi);
                        asm.pop(rdi)?;
                        stack_values -= 1;
                        unlazy!(rdi, rcx);
                        asm.mov(rsi, r12)?;

                        let mut end = asm.create_label();
                        let mut not_an_attrset = asm.create_label();
                        unpack!(Attrset rdi tmp = rcx else => not_an_attrset);
                        unpack!(Attrset rsi tmp = rcx else => not_an_attrset);
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
                        unlazy!(rdi, rcx);

                        let mut end = asm.create_label();
                        let mut not_a_string = asm.create_label();

                        asm.mov(rcx, rdi)?;
                        asm.and(rcx, VALUE_TAG_MASK as i32)?;
                        asm.cmp(rcx, ValueKind::String as i32)?;
                        asm.jne(not_a_string)?;

                        asm.sub(rdi, ValueKind::String as i32)?;
                        asm.mov(rsi, qword_ptr(rsp))?;
                        asm.sub(rsi, ValueKind::String as i32)?;

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
                            span
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

            let rc = Rc::new(Executable {
                _closure: closure,
                eh_frame: frame,
                code: std::mem::transmute(code_mem),
                len: code.len(),
                source_map,
            });
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
        let cfa = _Unwind_GetCFA(context);
        let region_start = _Unwind_GetRegionStart(context);
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
