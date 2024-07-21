#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::type_complexity)]

use core::arch::asm;
use std::{
    collections::HashMap,
    ffi::CStr,
    fmt::{Debug, Display, Write},
    mem::offset_of,
    ops::{Deref, DerefMut},
    os::raw::c_void,
    rc::Rc,
};

use iced_x86::{code_asm::*, Formatter};
use nix::libc::{MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};
use rnix::ast::{Expr, HasEntry, InterpolPart};

#[derive(Debug)]
enum Operation {
    Push(Value),
    // This is different from Push because this also sets the parent scope of the function
    PushFunction(Executable),
    // identifier lookup
    Load(String),
    CreateAttrset,
    SetAttr,
    GetAttrConsume,
    CreateList,
    ListAppend,
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Less,
    LessOrEqual,
    MoreOrEqual,
    More,
    Equal,
    NotEqual,
    Implication,
    Apply,
}

#[derive(Debug)]
struct Program {
    operations: Vec<Operation>,
}

fn build_program(expr: Expr, executable: &mut Program) {
    match expr {
        Expr::Apply(x) => {
            build_program(x.argument().unwrap(), executable);
            build_program(x.lambda().unwrap(), executable);
            executable.operations.push(Operation::Apply);
        }
        Expr::Assert(_) => todo!(),
        Expr::Error(_) => todo!(),
        Expr::IfElse(_) => todo!(),
        Expr::Select(x) => {
            build_program(x.expr().unwrap(), executable);
            for attr in x.attrpath().unwrap().attrs() {
                executable.operations.push(Operation::Push(
                    UnpackedValue::new_string(attr.to_string()).pack(),
                ));
                executable.operations.push(Operation::GetAttrConsume)
            }
        }
        Expr::Str(x) => {
            let mut parts = x.normalized_parts();
            match &mut parts[..] {
                [InterpolPart::Literal(value)] => executable.operations.push(Operation::Push(
                    UnpackedValue::new_string(std::mem::take(value)).pack(),
                )),
                [] => executable.operations.push(Operation::Push(
                    UnpackedValue::new_string(String::new()).pack(),
                )),
                other => todo!(),
            }
        }
        Expr::Path(_) => todo!(),
        Expr::Literal(x) => match x.kind() {
            rnix::ast::LiteralKind::Float(_) => todo!(),
            rnix::ast::LiteralKind::Integer(x) => {
                executable.operations.push(Operation::Push(
                    UnpackedValue::Integer(x.value().unwrap()).pack(),
                ));
            }
            rnix::ast::LiteralKind::Uri(_) => todo!(),
        },
        Expr::Lambda(x) => {
            let param = x.param().unwrap();
            let param_name = String::leak(param.to_string());
            let body = x.body().unwrap();
            let mut function_executable = Program::new();
            build_program(body, &mut function_executable);
            let compiled = function_executable
                .compile(Some(param.to_string()))
                .unwrap();
            executable
                .operations
                .push(Operation::PushFunction(compiled))
        }
        Expr::LegacyLet(_) => todo!(),
        Expr::LetIn(_) => todo!(),
        Expr::List(x) => {
            executable.operations.push(Operation::CreateList);
            for item in x.items() {
                build_program(item, executable);
                executable.operations.push(Operation::ListAppend);
            }
        }
        Expr::BinOp(x) => {
            build_program(x.lhs().unwrap(), executable);
            build_program(x.rhs().unwrap(), executable);
            executable.operations.push(match x.operator().unwrap() {
                rnix::ast::BinOpKind::Concat => todo!(),
                rnix::ast::BinOpKind::Update => todo!(),
                rnix::ast::BinOpKind::Add => Operation::Add,
                rnix::ast::BinOpKind::Sub => Operation::Sub,
                rnix::ast::BinOpKind::Mul => Operation::Mul,
                rnix::ast::BinOpKind::Div => Operation::Div,
                rnix::ast::BinOpKind::And => Operation::And,
                rnix::ast::BinOpKind::Equal => Operation::Equal,
                rnix::ast::BinOpKind::Implication => Operation::Implication,
                rnix::ast::BinOpKind::Less => Operation::Less,
                rnix::ast::BinOpKind::LessOrEq => Operation::LessOrEqual,
                rnix::ast::BinOpKind::More => Operation::More,
                rnix::ast::BinOpKind::MoreOrEq => Operation::MoreOrEqual,
                rnix::ast::BinOpKind::NotEqual => Operation::NotEqual,
                rnix::ast::BinOpKind::Or => Operation::Or,
            })
        }
        Expr::Paren(x) => build_program(x.expr().unwrap(), executable),
        Expr::AttrSet(x) => {
            assert!(x.inherits().next().is_none());
            executable.operations.push(Operation::CreateAttrset);
            for keyvalue in x.attrpath_values() {
                let path = keyvalue.attrpath().unwrap();
                let value = keyvalue.value().unwrap();
                build_program(value, executable);
                let mut attrs = path.attrs();
                let name = attrs.next().unwrap().to_string();
                assert!(attrs.next().is_none());
                executable
                    .operations
                    .push(Operation::Push(UnpackedValue::new_string(name).pack()));
                executable.operations.push(Operation::SetAttr);
            }
        }
        Expr::UnaryOp(_) => todo!(),
        Expr::Ident(x) => executable.operations.push(Operation::Load(
            x.ident_token().unwrap().green().text().to_string(),
        )),
        Expr::With(_) => todo!(),
        Expr::HasAttr(_) => todo!(),
        _ => todo!(),
    }
}

#[repr(C)]
struct Scope {
    values: HashMap<String, Value>,
    previous: *mut Scope,
}

unsafe extern "C" fn scope_lookup(
    mut scope: *const Scope,
    name: *const u8,
    name_len: usize,
) -> Value {
    let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(name, name_len));
    loop {
        match (*scope).values.get(name) {
            Some(value) => return value.clone(),

            None => {
                scope = (*scope).previous;
                if scope.is_null() {
                    panic!("identifier {name} not found")
                }
            }
        }
    }
}

unsafe extern "C" fn scope_function_scope(
    previous: *mut Scope,
    name: *const u8,
    name_len: usize,
    value: Value,
) -> *mut Scope {
    let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(name, name_len));
    println!("creating function scope {previous:?} {name} {name_len} {value:?}");
    Box::leak(Box::new(Scope {
        values: {
            let mut map = HashMap::new();
            map.insert(name.to_string(), value);
            map
        },
        previous,
    }))
}

#[derive(Debug)]
struct Executable {
    len: usize,
    code: extern "C" fn(Value, Value) -> Value,
}

impl Executable {
    fn run(&self, scope: *mut Scope, arg: Value) -> Value {
        unsafe {
            asm!("mov r15, {}", in(reg) scope);
        }

        (self.code)(Value::NULL, arg)
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        unsafe {
            if nix::libc::munmap(self.code as *mut c_void, self.len) < 0 {
                panic!("munmap failed");
            }
        }
    }
}

unsafe extern "C" fn create_function_value(
    executable: *const Executable,
    scope: *mut Scope,
) -> Value {
    Rc::increment_strong_count(executable);
    UnpackedValue::Function(HeapValue::new(Function {
        call: (*executable).code,
        executable: Some(Rc::from_raw(executable)),
        builtin_closure: None,
        parent_scope: scope,
    }))
    .pack()
}

unsafe extern "C" fn list_create() -> Value {
    UnpackedValue::List(HeapValue::new(vec![])).pack()
}

unsafe extern "C" fn list_append_value(list: &mut Vec<Value>, value: Value) {
    list.push(value);
}

unsafe extern "C" fn asm_panic(msg: *const i8) {
    panic!("[JITPANIC] {}", CStr::from_ptr(msg).to_string_lossy());
}

impl Program {
    fn new() -> Self {
        Self { operations: vec![] }
    }

    fn compile(self, param: Option<String>) -> Result<Executable, IcedError> {
        println!("{self:?}");

        let code = {
            let mut asm = CodeAssembler::new(64)?;

            asm.push(rbp)?;
            asm.mov(rbp, rsp)?;

            // callee saved registers
            asm.push(rbx)?;
            asm.push(r12)?;
            asm.push(r13)?;
            asm.push(r14)?;
            // invariant: r15 contains our scope (set by the caller)
            // asm.push(r15)?;

            // keep track of whether we need to align the stack when calling
            // and also verify the operations are valid and won't consume
            // more values on the stack than they should
            let mut stack_values = 0;

            macro_rules! call {
                (reg $reg: ident) => {
                    if stack_values % 2 != 0 {
                        asm.sub(rsp, 8)?;
                    }

                    asm.call($reg)?;

                    if stack_values % 2 != 0 {
                        asm.add(rsp, 8)?;
                    }
                };
                (rust $function: expr) => {
                    #[allow(clippy::fn_to_numeric_cast)]
                    asm.mov(rax, $function as u64)?;
                    call!(reg rax)
                };
            }

            if let Some(param) = param.map(String::leak) {
                // rdi = Current function Value (if a function)
                // rsi = Argument Value (if a function)

                asm.mov(
                    rdi,
                    qword_ptr(
                        rdi + offset_of!(HeapValue<Function>, value)
                            + offset_of!(Function, parent_scope)
                            - ValueKind::Function as i32,
                    ),
                )?;
                asm.mov(rcx, rsi)?;
                asm.mov(rsi, param.as_ptr() as u64)?;
                asm.mov(rdx, param.len() as u64)?;

                call!(rust scope_function_scope);
                asm.mov(r15, rax)?;
            }

            macro_rules! impl_binary_operator {
                ($(int => $if_int: expr,)? $(bool => $if_bool: expr,)? fallback = $other: expr) => {{
                    asm.pop(rsi)?;
                    asm.pop(rdi)?;
                    stack_values -= 2;

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
                        asm.shl(register, VALUE_TAG_WIDTH as i32)?;
                        asm.or(register, ValueKind::Integer as i32)?;

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
            }

            for op in self.operations.into_iter() {
                match op {
                    Operation::Push(value) => {
                        asm.mov(rax, value.0)?;
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::PushFunction(exec) => {
                        // leak
                        let raw = Rc::into_raw(Rc::new(exec));
                        asm.mov(rdi, raw as u64)?;
                        asm.mov(rsi, r15)?;
                        call!(rust create_function_value);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::Add => impl_binary_operator!(
                        int => { asm.add(rdi, rsi)?; rdi },
                        fallback = Value::add
                    ),
                    Operation::Sub => impl_binary_operator!(
                        int => { asm.sub(rdi, rsi)?; rdi },
                        fallback = Value::sub
                    ),
                    Operation::Mul => impl_binary_operator!(
                        int => { asm.imul_2(rdi, rsi)?; rdi },
                        fallback = Value::mul
                    ),
                    Operation::Div => impl_binary_operator!(
                        int => {
                            asm.xor(rdx, rdx)?;
                            asm.mov(rax, rdi)?;
                            asm.idiv(rsi)?;
                            rax
                        },
                        fallback = Value::div
                    ),
                    Operation::And => impl_binary_operator!(
                        bool => asm.and(rdi, rsi)?,
                        fallback = Value::and
                    ),
                    Operation::CreateAttrset => {
                        call!(rust Attrset::create);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::SetAttr => {
                        assert!(stack_values >= 3);
                        asm.pop(rdx)?;
                        asm.pop(rsi)?;
                        asm.mov(rdi, qword_ptr(rsp))?;
                        stack_values -= 2;

                        let mut not_an_attrset = asm.create_label();
                        asm.mov(rcx, rdi)?;
                        asm.and(rcx, VALUE_TAG_MASK as i32)?;
                        asm.cmp(cl, ValueKind::Attrset as u32)?;
                        asm.jne(not_an_attrset)?;
                        asm.cmp(rdi, 0)?; // attrset tag also applies to null
                        asm.je(not_an_attrset)?;

                        asm.add(
                            rdi,
                            offset_of!(HeapValue<Attrset>, value) as i32
                                - ValueKind::Attrset as i32,
                        )?;

                        call!(rust Attrset::set_attr);

                        let mut end = asm.create_label();
                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_attrset)?;

                        asm.mov(rdi, c"setattr called on non-attrset value".as_ptr() as u64)?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                    }
                    Operation::GetAttrConsume => {
                        assert!(stack_values >= 2);
                        asm.pop(rsi)?;
                        asm.pop(rdi)?;
                        stack_values -= 2;

                        asm.mov(rcx, rdi)?;
                        asm.and(rcx, VALUE_TAG_MASK as i32)?;
                        asm.cmp(cl, ValueKind::Attrset as u32)?;
                        let mut not_an_attrset = asm.create_label();
                        asm.jne(not_an_attrset)?;
                        asm.cmp(rdi, 0)?; // attrset tag also applies to null
                        asm.je(not_an_attrset)?;

                        asm.add(
                            rdi,
                            offset_of!(HeapValue<Attrset>, value) as i32
                                - ValueKind::Attrset as i32,
                        )?;

                        call!(rust Attrset::get_attr);
                        asm.push(rax)?;

                        let mut end = asm.create_label();
                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_attrset)?;

                        asm.mov(rdi, c"getattr called on non-attrset value".as_ptr() as u64)?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::Apply => {
                        assert!(stack_values >= 2);
                        asm.pop(rbx)?;
                        asm.pop(rsi)?;
                        stack_values -= 2;

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
                                rbx + offset_of!(HeapValue<Function>, value) as i32
                                    + offset_of!(Function, call) as i32
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
                        let name = String::leak(name);
                        asm.mov(rdi, r15)?;
                        asm.mov(rsi, name.as_ptr() as u64)?;
                        asm.mov(rdx, name.len() as u64)?;
                        call!(rust scope_lookup);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::CreateList => {
                        call!(rust list_create);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::ListAppend => {
                        assert!(stack_values >= 2);
                        asm.pop(rsi)?;
                        asm.mov(rdi, qword_ptr(rsp))?;
                        stack_values -= 1;

                        let mut end = asm.create_label();
                        let mut not_a_list = asm.create_label();

                        asm.mov(rax, rdi)?;
                        asm.and(rax, VALUE_TAG_MASK as i32)?;
                        asm.cmp(al, ValueKind::List as u32)?;
                        asm.jne(not_a_list)?;

                        asm.add(
                            rdi,
                            offset_of!(HeapValue<Vec<Value>>, value) as i32
                                - ValueKind::List as i32,
                        )?;
                        call!(rust list_append_value);
                        asm.jmp(end)?;

                        asm.set_label(&mut not_a_list)?;

                        asm.mov(rdi, c"append called on non-list value".as_ptr() as u64)?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                    }
                    _ => todo!(),
                }
            }

            asm.pop(rax)?;

            // callee saved registers
            // asm.pop(r15)?;
            asm.pop(r14)?;
            asm.pop(r13)?;
            asm.pop(r12)?;
            asm.pop(rbx)?;

            asm.leave()?;
            asm.ret()?;

            asm.assemble(0)?
        };

        let decoder = iced_x86::Decoder::new(64, &code, 0);
        let mut formatter = iced_x86::IntelFormatter::new();
        let mut output = String::new();
        for instruction in decoder {
            output.clear();
            formatter.format(&instruction, &mut output);
            println!("\t{output}")
        }

        unsafe {
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
            if nix::libc::mprotect(code_mem as *mut c_void, code.len(), PROT_EXEC | PROT_READ) < 0 {
                panic!("mprotect failed");
            }
            Ok(Executable {
                code: std::mem::transmute(code_mem),
                len: code.len(),
            })
        }
    }
}

const VALUE_TAG_WIDTH: u8 = 3;
const VALUE_TAG_MASK: u64 = 0b111;

#[repr(u8)]
#[derive(Debug, PartialEq, Eq)]
enum ValueKind {
    Attrset = 0, // also used as a tag for null
    Function,
    Integer,
    Double,
    String,
    Path,
    List,
    Bool,
    Null, // all zeroes: an attrset which is a null pointer
}

#[repr(C)]
struct HeapValue<T = ()> {
    // NOTE: Currently unused
    // TODO: Implement reference counted memory management
    refcount: u64,
    value: T,
}

impl<T> HeapValue<T> {
    fn new(value: T) -> *mut HeapValue<T> {
        Box::leak(Box::new(Self { refcount: 1, value }))
    }
}

impl<T> Deref for HeapValue<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for HeapValue<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

struct Attrset {
    map: HashMap<String, Value>,
}

impl Attrset {
    extern "C" fn create() -> Value {
        UnpackedValue::Attrset(HeapValue::new(Attrset {
            map: HashMap::new(),
        }))
        .pack()
    }

    extern "C" fn set_attr(&mut self, value: Value, name: Value) {
        let UnpackedValue::String(name) = name.unpack() else {
            panic!("SetAttr called with non-string name")
        };
        let name = unsafe { (*name).as_str() };
        self.map.insert(name.to_string(), value);
    }

    extern "C" fn get_attr(&mut self, name: Value) -> Value {
        let UnpackedValue::String(name) = name.unpack() else {
            panic!("GetAttr called with non-string name")
        };
        self.map.get(unsafe { (*name).as_str() }).cloned().unwrap()
    }
}

#[repr(C)]
struct Function {
    call: unsafe extern "C" fn(Value, Value) -> Value,
    executable: Option<Rc<Executable>>,
    builtin_closure: Option<Box<dyn FnMut(Value) -> Value>>,
    parent_scope: *const Scope,
}

enum UnpackedValue {
    Integer(i64),
    Bool(bool),
    List(*mut HeapValue<Vec<Value>>),
    Attrset(*mut HeapValue<Attrset>),
    String(*mut HeapValue<String>),
    Function(*mut HeapValue<Function>),
    Null,
}

impl Display for ValueKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueKind::Integer => f.write_str("integer"),
            ValueKind::Double => f.write_str("float"),
            ValueKind::Bool => f.write_str("boolean"),
            ValueKind::String => f.write_str("string"),
            ValueKind::Path => f.write_str("path"),
            ValueKind::List => f.write_str("list"),
            ValueKind::Attrset => f.write_str("set"),
            ValueKind::Null => f.write_str("null"),
            Self::Function => f.write_str("function"),
        }
    }
}

unsafe extern "C" fn call_builtin(fun: Value, value: Value) -> Value {
    let this = (fun.0 & !VALUE_TAG_MASK) as *mut HeapValue<Function>;
    println!(
        "call_builtin HeapValue:{:?} arg:{value:?}",
        this as *const _
    );
    ((*this).builtin_closure.as_mut().unwrap_unchecked())(value)
}

impl UnpackedValue {
    fn new_string(value: String) -> UnpackedValue {
        UnpackedValue::String(Box::leak(Box::new(HeapValue { refcount: 1, value })))
    }

    fn new_attrset(values: HashMap<String, Value>) -> UnpackedValue {
        UnpackedValue::Attrset(Box::leak(Box::new(HeapValue {
            refcount: 1,
            value: Attrset { map: values },
        })))
    }

    fn new_function(function: impl FnMut(Value) -> Value + 'static) -> UnpackedValue {
        UnpackedValue::Function(HeapValue::new(Function {
            call: call_builtin,
            executable: None,
            builtin_closure: Some(Box::new(function)),
            parent_scope: std::ptr::null_mut(),
        }))
    }

    fn kind(&self) -> ValueKind {
        match self {
            UnpackedValue::Integer(_) => ValueKind::Integer,
            UnpackedValue::Bool(_) => ValueKind::Bool,
            UnpackedValue::List(_) => ValueKind::List,
            UnpackedValue::Attrset(_) => ValueKind::Attrset,
            UnpackedValue::String(_) => ValueKind::String,
            UnpackedValue::Function(_) => ValueKind::Function,
            UnpackedValue::Null => ValueKind::Null,
        }
    }

    fn pack(self) -> Value {
        macro_rules! pack_ptr {
            ($ptr: ident, $kind: ident) => {{
                println!(
                    "HeapValue<> PACKED {:x} = {}",
                    ($ptr as u64),
                    ValueKind::$kind
                );
                assert!(($ptr as u64).trailing_zeros() >= VALUE_TAG_WIDTH.into());
                Value(($ptr as u64) | (ValueKind::$kind as u64))
            }};
        }
        match self {
            Self::Integer(x) => unsafe {
                Value(
                    (std::mem::transmute::<i64, u64>(x) << VALUE_TAG_WIDTH)
                        | (ValueKind::Integer as u64),
                )
            },
            Self::Bool(x) => Value(if x {
                0b1000 | (ValueKind::Bool as u64)
            } else {
                ValueKind::Bool as u64
            }),
            Self::String(ptr) => pack_ptr!(ptr, String),
            Self::List(ptr) => pack_ptr!(ptr, List),
            Self::Attrset(ptr) => pack_ptr!(ptr, Attrset),
            Self::Function(ptr) => pack_ptr!(ptr, Function),
            Self::Null => Value::NULL,
        }
    }

    fn fmt_attrset_display(
        map: &HashMap<String, Value>,
        depth: usize,
        f: &mut impl Write,
        debug: bool,
    ) -> std::fmt::Result {
        writeln!(f, "{{")?;
        for (key, value) in map.iter() {
            for _ in 0..=depth {
                write!(f, "  ")?;
            }
            write!(f, "{key} = ")?;
            if debug {
                value.unpack().fmt_debug_rec(depth + 1, f)?
            } else {
                value.unpack().fmt_display_rec(depth + 1, f)?
            }
            writeln!(f, ";")?;
        }
        for _ in 0..depth {
            write!(f, "  ")?;
        }
        write!(f, "}}")
    }

    fn fmt_list_display(
        list: &[Value],
        depth: usize,
        f: &mut impl Write,
        debug: bool,
    ) -> std::fmt::Result {
        if list.len() > 2 {
            writeln!(f, "[")?;
        } else {
            write!(f, "[ ")?;
        }
        let mut first = true;
        for value in list.iter() {
            if list.len() > 2 {
                for _ in 0..=depth {
                    write!(f, "  ")?;
                }
            }
            if debug {
                value.unpack().fmt_debug_rec(depth + 1, f)?
            } else {
                value.unpack().fmt_display_rec(depth + 1, f)?
            }
            if list.len() > 2 {
                writeln!(f)?;
            } else {
                write!(f, " ")?;
            }
        }
        if list.len() > 2 {
            for _ in 0..depth {
                write!(f, "  ")?;
            }
        }
        write!(f, "]")
    }

    fn fmt_display_rec(&self, depth: usize, f: &mut impl Write) -> std::fmt::Result {
        match self {
            Self::Integer(value) => {
                write!(f, "{value}")
            }
            Self::Bool(value) => {
                write!(f, "{value}")
            }
            Self::String(value) => {
                if depth == 0 {
                    write!(f, "{}", unsafe { (**value).as_str() })
                } else {
                    write!(f, "{:?}", unsafe { (**value).as_str() })
                }
            }
            Self::List(value) => Self::fmt_list_display(unsafe { &**value }, depth, f, false),
            Self::Attrset(value) => {
                Self::fmt_attrset_display(unsafe { &(**value).map }, depth, f, false)
            }
            Self::Function(_) => write!(f, "<function>"),
            UnpackedValue::Null => write!(f, "null"),
        }
    }

    fn fmt_debug_rec(&self, depth: usize, f: &mut impl Write) -> std::fmt::Result {
        match self {
            Self::Integer(value) => {
                write!(f, "{value}")
            }
            Self::Bool(value) => {
                write!(f, "{value}")
            }
            Self::String(value) => {
                let value = unsafe { (**value).as_str() };
                write!(f, "{value:?}")
            }
            Self::List(value) => Self::fmt_list_display(unsafe { &**value }, depth, f, true),
            Self::Attrset(value) => {
                Self::fmt_attrset_display(unsafe { &(**value).map }, depth, f, true)
            }
            Self::Function(_) => write!(f, "<function>"),
            Self::Null => write!(f, "null"),
        }
    }
}

impl Display for UnpackedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_display_rec(0, f)
    }
}

impl Debug for UnpackedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_debug_rec(0, f)
    }
}

impl Clone for UnpackedValue {
    // TODO: refcnt
    fn clone(&self) -> Self {
        macro_rules! clone_ptr {
            ($arg: ident, $kind: ident) => {{
                unsafe {
                    (**$arg).refcount += 1;
                };
                Self::$kind(*$arg)
            }};
        }
        match self {
            Self::Integer(value) => Self::Integer(*value),
            Self::Bool(value) => Self::Bool(*value),
            Self::Attrset(ptr) => clone_ptr!(ptr, Attrset),
            Self::String(ptr) => clone_ptr!(ptr, String),
            Self::List(ptr) => clone_ptr!(ptr, List),
            Self::Function(ptr) => clone_ptr!(ptr, Function),
            Self::Null => Self::Null,
        }
    }
}

// https://github.com/SerenityOS/serenity/blob/1d29f9081fc243407e4902713d3c7fc0ee69650c/Userland/Libraries/LibJS/Runtime/Value.h
#[repr(transparent)]
struct Value(u64);

impl Value {
    const NULL: Value = Value(0);

    fn kind(&self) -> ValueKind {
        if self.0 == 0 {
            return ValueKind::Null;
        }
        let value = self.0 & VALUE_TAG_MASK;
        assert!(value <= ValueKind::Bool as u64);
        unsafe { std::mem::transmute(value as u8) }
    }

    fn unpack(&self) -> UnpackedValue {
        macro_rules! unpack_ptr {
            ($kind: ident) => {
                UnpackedValue::$kind((self.0 & !VALUE_TAG_MASK) as *mut _)
            };
        }
        match self.kind() {
            ValueKind::Integer => {
                UnpackedValue::Integer(unsafe { std::mem::transmute(self.0 >> VALUE_TAG_WIDTH) })
            }
            ValueKind::Double => todo!(),
            ValueKind::Bool => UnpackedValue::Bool(self.0 == 0b1000 | (ValueKind::Bool as u64)),
            ValueKind::String => unpack_ptr!(String),
            ValueKind::Path => todo!(),
            ValueKind::List => unpack_ptr!(List),
            ValueKind::Attrset => unpack_ptr!(Attrset),
            ValueKind::Function => unpack_ptr!(Function),
            ValueKind::Null => UnpackedValue::Null,
        }
    }

    #[inline]
    fn leaking_copy(&self) -> Value {
        Value(self.0)
    }

    extern "C" fn add(self, other: Value) -> Value {
        match (self.unpack(), other.unpack()) {
            (UnpackedValue::Integer(a), UnpackedValue::Integer(b)) => {
                UnpackedValue::Integer(a + b).pack()
            }
            (_, _) => todo!(),
        }
    }

    extern "C" fn sub(self, other: Value) -> Value {
        match (self.unpack(), other.unpack()) {
            (UnpackedValue::Integer(a), UnpackedValue::Integer(b)) => {
                UnpackedValue::Integer(a - b).pack()
            }
            (_, _) => todo!(),
        }
    }

    extern "C" fn mul(self, other: Value) -> Value {
        match (self.unpack(), other.unpack()) {
            (UnpackedValue::Integer(a), UnpackedValue::Integer(b)) => {
                UnpackedValue::Integer(a * b).pack()
            }
            (_, _) => todo!(),
        }
    }

    extern "C" fn div(self, other: Value) -> Value {
        match (self.unpack(), other.unpack()) {
            (UnpackedValue::Integer(a), UnpackedValue::Integer(b)) => {
                UnpackedValue::Integer(a / b).pack()
            }
            (_, _) => todo!(),
        }
    }

    extern "C" fn and(self, other: Value) -> Value {
        match (self.unpack(), other.unpack()) {
            (UnpackedValue::Bool(a), UnpackedValue::Bool(b)) => UnpackedValue::Bool(a && b).pack(),
            (_, _) => panic!(
                "&& is not supported between value of type {} and {}",
                self.kind(),
                other.kind()
            ),
        }
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        // TODO: refcnt
        Self(self.0)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.unpack().fmt_display_rec(0, f)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.unpack())
    }
}

fn create_root_scope() -> Scope {
    let mut values = HashMap::new();
    values.insert(
        "builtins".to_string(),
        UnpackedValue::new_attrset({
            let mut builtins = HashMap::new();

            macro_rules! extract_typed {
                ($builtin_name: literal, $what: ident($value: expr)) => {
                    match &$value.unpack() {
                        UnpackedValue::$what(ptr) => unsafe { &(**ptr).value },
                        other => panic!(
                            concat!("builtin ", $builtin_name, " expected {} but got {}"),
                            ValueKind::$what,
                            other.kind()
                        ),
                    }
                };
            }

            builtins.insert(
                "map".to_string(),
                UnpackedValue::new_function(|fun| {
                    let mapper = extract_typed!("map", Function(fun));
                    UnpackedValue::new_function(move |list| {
                        let list = extract_typed!("map", List(list));
                        UnpackedValue::List(HeapValue::new(
                            list.iter()
                                .cloned()
                                .map(|x| unsafe { (mapper.call)(fun.leaking_copy(), x) })
                                .collect(),
                        ))
                        .pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "trace".to_string(),
                UnpackedValue::new_function(|message| {
                    println!("trace: {message}");
                    UnpackedValue::new_function(|value| value).pack()
                })
                .pack(),
            );

            builtins
        })
        .pack(),
    );
    values.insert("true".to_string(), UnpackedValue::Bool(true).pack());
    values.insert("false".to_string(), UnpackedValue::Bool(false).pack());
    values.insert("null".to_string(), UnpackedValue::Null.pack());
    Scope {
        values,
        previous: std::ptr::null_mut(),
    }
}

fn main() {
    let fake_ptr = 0b10101000 as *mut u8 as *mut HeapValue<Attrset>;
    assert!(matches!(
        UnpackedValue::Attrset(fake_ptr).pack().unpack(),
        UnpackedValue::Attrset(ptr) if ptr == fake_ptr
    ));

    let result =
        rnix::Root::parse(&std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap());
    let root = result.tree();
    let mut root_block = Program { operations: vec![] };
    build_program(root.expr().unwrap(), &mut root_block);
    let compiled = root_block.compile(None).unwrap();
    let result = compiled.run(&mut create_root_scope() as *mut Scope, Value::NULL);
    println!("{result}");
}
