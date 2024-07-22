#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::fn_to_numeric_cast)]
#![allow(clippy::type_complexity)]

use core::{arch::asm, panic};
use std::{
    cell::UnsafeCell,
    collections::BTreeMap,
    ffi::{CStr, CString},
    fmt::{Debug, Display, Write},
    mem::{offset_of, MaybeUninit},
    ops::{Deref, DerefMut},
    os::raw::c_void,
    path::{Path, PathBuf},
    rc::Rc,
};

use iced_x86::{code_asm::*, Formatter};
use nix::libc::{MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};
use rnix::ast::{Attr, Attrpath, Expr, HasEntry, InterpolPart};

#[derive(Debug)]
enum Operation {
    Push(Value),
    // This is different from Push because this also sets the parent scope of the function
    PushFunction(Executable),
    CreateAttrset,
    InheritAttrs(bool, Vec<String>),
    SetAttrpath(usize, Program),
    GetAttrConsume,
    HasAttrpath(usize),
    PushList,
    ListAppend(Program),
    Concat,
    Update,
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
    ScopePush,
    ScopeInherit(Option<Program>, Vec<String>),
    ScopeSet(String, Program),
    // identifier lookup
    Load(String),
    ScopePop,
    IfElse(Program, Program),
    Invert,
    Negate,
}

#[derive(Debug)]
struct Program {
    operations: Vec<Operation>,
}

fn create_program(here: &Path, expr: Expr) -> Program {
    let mut program = Program::new();
    build_program(here, expr, &mut program);
    program
}

fn attr_assert_const(attr: Attr) -> String {
    match attr {
        Attr::Ident(x) => x.ident_token().unwrap().green().text().to_string(),
        Attr::Dynamic(_) => todo!(),
        Attr::Str(_) => todo!(),
    }
}

fn build_attrpath(here: &Path, attrpath: Attrpath, program: &mut Program) -> usize {
    let mut n = 0;
    for attr in attrpath.attrs().collect::<Vec<_>>().into_iter().rev() {
        match attr {
            Attr::Ident(x) => program.operations.push(Operation::Push(
                UnpackedValue::new_string(x.ident_token().unwrap().green().text().to_string())
                    .pack(),
            )),
            Attr::Dynamic(x) => build_program(here, x.expr().unwrap(), program),
            Attr::Str(_) => todo!(),
        };
        n += 1;
    }
    n
}

fn build_program(here: &Path, expr: Expr, program: &mut Program) {
    match expr {
        Expr::Apply(x) => {
            build_program(here, x.argument().unwrap(), program);
            build_program(here, x.lambda().unwrap(), program);
            program.operations.push(Operation::Apply);
        }
        // TODO:
        Expr::Assert(_) => (),
        Expr::Error(_) => todo!(),
        Expr::IfElse(x) => {
            build_program(here, x.condition().unwrap(), program);
            let if_true = create_program(here, x.body().unwrap());
            let if_false = create_program(here, x.else_body().unwrap());
            program
                .operations
                .push(Operation::IfElse(if_true, if_false))
        }
        Expr::Select(x) => {
            build_program(here, x.expr().unwrap(), program);
            for attr in x.attrpath().unwrap().attrs() {
                program.operations.push(Operation::Push(
                    UnpackedValue::new_string(attr.to_string()).pack(),
                ));
                program.operations.push(Operation::GetAttrConsume)
            }
        }
        Expr::Str(x) => {
            let mut parts = x.normalized_parts();
            match &mut parts[..] {
                [InterpolPart::Literal(value)] => program.operations.push(Operation::Push(
                    UnpackedValue::new_string(std::mem::take(value)).pack(),
                )),
                [] => program.operations.push(Operation::Push(
                    UnpackedValue::new_string(String::new()).pack(),
                )),
                other => todo!(),
            }
        }
        Expr::Path(x) => {
            let mut parts = x.parts().collect::<Vec<_>>();
            match &mut parts[..] {
                [InterpolPart::Literal(value)] => program.operations.push(Operation::Push(
                    UnpackedValue::new_path(here.join(value.to_string())).pack(),
                )),
                [] => program.operations.push(Operation::Push(
                    UnpackedValue::new_path(PathBuf::new()).pack(),
                )),
                other => todo!(),
            }
        }
        Expr::Literal(x) => match x.kind() {
            rnix::ast::LiteralKind::Float(_) => todo!(),
            rnix::ast::LiteralKind::Integer(x) => {
                program.operations.push(Operation::Push(
                    UnpackedValue::Integer(x.value().unwrap()).pack(),
                ));
            }
            rnix::ast::LiteralKind::Uri(_) => todo!(),
        },
        Expr::Lambda(x) => {
            let param = x.param().unwrap();
            let param = match param {
                rnix::ast::Param::Pattern(pat) => Parameter::Pattern {
                    entries: pat
                        .pat_entries()
                        .map(|x| {
                            (
                                x.ident().unwrap().to_string(),
                                x.default()
                                    .map(|x| create_program(here, x))
                                    .map(|x| x.compile(None).unwrap()),
                            )
                        })
                        .collect(),
                    binding: pat.pat_bind().map(|x| x.ident().unwrap().to_string()),
                    ignore_unmatched: pat.ellipsis_token().is_some(),
                },
                rnix::ast::Param::IdentParam(ident) => {
                    Parameter::Ident(ident.ident().unwrap().to_string())
                }
            };
            let body = x.body().unwrap();
            let mut function_executable = Program::new();
            build_program(here, body, &mut function_executable);
            let compiled = function_executable.compile(Some(param)).unwrap();
            program.operations.push(Operation::PushFunction(compiled))
        }
        Expr::LegacyLet(_) => todo!(),
        Expr::LetIn(x) => {
            program.operations.push(Operation::ScopePush);
            for inherit in x.inherits() {
                let source = inherit
                    .from()
                    .and_then(|f| f.expr())
                    .map(|source| create_program(here, source));
                program.operations.push(Operation::ScopeInherit(
                    source,
                    inherit.attrs().map(attr_assert_const).collect(),
                ));
            }
            for entry in x.attrpath_values() {
                let mut it = entry.attrpath().unwrap().attrs();
                let name = attr_assert_const(it.next().unwrap());
                let value = entry.value().unwrap();
                let value_program = create_program(here, value);
                program
                    .operations
                    .push(Operation::ScopeSet(name, value_program));
            }
            build_program(here, x.body().unwrap(), program);
            program.operations.push(Operation::ScopePop);
        }
        Expr::List(x) => {
            program.operations.push(Operation::PushList);
            for item in x.items() {
                let value_program = create_program(here, item);
                program
                    .operations
                    .push(Operation::ListAppend(value_program));
            }
        }
        Expr::BinOp(x) => {
            build_program(here, x.lhs().unwrap(), program);
            build_program(here, x.rhs().unwrap(), program);
            program.operations.push(match x.operator().unwrap() {
                rnix::ast::BinOpKind::Concat => Operation::Concat,
                rnix::ast::BinOpKind::Update => Operation::Update,
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
        Expr::Paren(x) => build_program(here, x.expr().unwrap(), program),
        Expr::AttrSet(x) => {
            program.operations.push(Operation::CreateAttrset);
            for inherit in x.inherits() {
                let has_source = if let Some(source) = inherit.from().and_then(|f| f.expr()) {
                    build_program(here, source, program);
                    true
                } else {
                    false
                };
                program.operations.push(Operation::InheritAttrs(
                    has_source,
                    inherit.attrs().map(attr_assert_const).collect(),
                ));
            }
            for keyvalue in x.attrpath_values() {
                let path = keyvalue.attrpath().unwrap();
                let value = create_program(here, keyvalue.value().unwrap());
                let n = build_attrpath(here, path, program);
                program.operations.push(Operation::SetAttrpath(n, value));
            }
        }
        Expr::UnaryOp(x) => {
            build_program(here, x.expr().unwrap(), program);
            program.operations.push(match x.operator().unwrap() {
                rnix::ast::UnaryOpKind::Invert => Operation::Invert,
                rnix::ast::UnaryOpKind::Negate => Operation::Negate,
            })
        }
        Expr::Ident(x) => program.operations.push(Operation::Load(
            x.ident_token().unwrap().green().text().to_string(),
        )),
        Expr::With(_) => todo!(),
        Expr::HasAttr(x) => {
            build_program(here, x.expr().unwrap(), program);
            let mut parts = 0;
            for attr in x.attrpath().unwrap().attrs() {
                let name = attr_assert_const(attr);
                program
                    .operations
                    .push(Operation::Push(UnpackedValue::new_string(name).pack()));
                parts += 1;
            }
            program.operations.push(Operation::HasAttrpath(parts))
        }
        _ => todo!(),
    }
}

#[derive(Debug)]
enum LazyValueImpl {
    Evaluated(Value),
    LazyJIT((*mut Scope, Rc<Executable>)),
    LazyBuiltin(MaybeUninit<Box<dyn FnOnce() -> Value>>),
}

#[derive(Debug, Clone)]
struct LazyValue(Rc<UnsafeCell<LazyValueImpl>>);

impl LazyValue {
    fn from_value(value: Value) -> LazyValue {
        Self(Rc::new(UnsafeCell::new(LazyValueImpl::Evaluated(value))))
    }

    fn from_jit(scope: *mut Scope, rc: Rc<Executable>) -> LazyValue {
        Self(Rc::new(UnsafeCell::new(LazyValueImpl::LazyJIT((
            scope, rc,
        )))))
    }

    fn from_closure(fun: impl FnOnce() -> Value + 'static) -> LazyValue {
        Self(Rc::new(UnsafeCell::new(LazyValueImpl::LazyBuiltin(
            MaybeUninit::new(Box::new(fun)),
        ))))
    }

    fn map(&self, fun: impl FnOnce(Value) -> Value + 'static) -> LazyValue {
        let rc = self.clone();
        Self::from_closure(move || fun(rc.evaluate().clone()))
    }

    #[inline]
    fn evaluate(&self) -> &Value {
        let inner = unsafe { &mut *self.0.get() };
        match inner {
            LazyValueImpl::Evaluated(value) => value,
            LazyValueImpl::LazyJIT((scope, executable)) => {
                println!("evaluating lazy JIT executable");
                let result = (**executable).run(*scope, &Value::NULL);
                unsafe { Rc::decrement_strong_count(executable) };
                *inner = LazyValueImpl::Evaluated(result);
                println!("evaluated lazy JIT executable");
                match inner {
                    LazyValueImpl::Evaluated(x) => x,
                    _ => unreachable!(),
                }
            }
            LazyValueImpl::LazyBuiltin(value) => {
                let result =
                    unsafe { (std::mem::replace(value, MaybeUninit::uninit()).assume_init())() };
                *inner = LazyValueImpl::Evaluated(result);
                match inner {
                    LazyValueImpl::Evaluated(x) => x,
                    _ => unreachable!(),
                }
            }
        }
    }

    #[inline]
    fn as_maybe_evaluated(&self) -> Option<&Value> {
        match unsafe { &*self.0.get() } {
            LazyValueImpl::Evaluated(value) => Some(value),
            _ => None,
        }
    }
}

struct LazyMap {
    values: BTreeMap<String, LazyValue>,
}

impl From<Value> for LazyValue {
    fn from(value: Value) -> Self {
        Self::from_value(value)
    }
}

impl From<UnpackedValue> for LazyValue {
    fn from(value: UnpackedValue) -> Self {
        Self::from_value(value.pack())
    }
}

impl LazyMap {
    fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<Value> {
        self.values
            .get_mut(key)
            .map(|value| value.evaluate().clone())
    }

    fn get_lazy_or_insert_with(
        &mut self,
        key: String,
        inserter: impl FnOnce() -> LazyValue,
    ) -> &LazyValue {
        self.values.entry(key).or_insert_with(inserter)
    }

    fn get_lazy(&self, key: &str) -> Option<&LazyValue> {
        self.values.get(key)
    }

    fn set(&mut self, key: String, value: impl Into<LazyValue>) {
        self.values.insert(key, value.into());
    }

    fn iter(&self) -> impl Iterator<Item = (&str, &LazyValue)> {
        self.values.iter().map(|x| (x.0.as_str(), x.1))
    }
}

#[derive(Debug)]
struct Executable {
    len: usize,
    code: extern "C" fn(Value, Value) -> Value,
}

impl Executable {
    #[inline(always)]
    fn run(&self, scope: *mut Scope, arg: &Value) -> Value {
        unsafe {
            asm!("mov r15, {}", in(reg) scope, out("r15") _);
            (self.code)(Value::NULL, arg.leaking_copy())
        }
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

#[repr(C)]
struct Scope {
    values: LazyMap,
    previous: *mut Scope,
}

impl Scope {
    unsafe fn lookup_lazy(mut scope: *mut Scope, name: &str) -> LazyValue {
        loop {
            match dbg!((*dbg!(scope)).values.get_lazy(dbg!(name))) {
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

    unsafe fn lookup(scope: *mut Scope, name: &str) -> Value {
        Scope::lookup_lazy(scope, name).evaluate().clone()
    }
}

unsafe extern "C" fn scope_lookup(scope: *mut Scope, name: *const u8, name_len: usize) -> Value {
    let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(name, name_len));
    Scope::lookup(scope, name)
}

unsafe extern "C" fn scope_function_scope(
    previous: *mut Scope,
    value: Value,
    parameter: *const Parameter,
) -> *mut Scope {
    match &*parameter {
        Parameter::Ident(name) => {
            println!("creating function scope {previous:?} {name} {value:?}");
            Box::leak(Box::new(Scope {
                values: {
                    let mut map = LazyMap::new();
                    map.set(name.to_string(), LazyValue::from_value(value));
                    map
                },
                previous,
            }))
        }
        Parameter::Pattern {
            entries,
            binding,
            ignore_unmatched,
        } => {
            let UnpackedValue::Attrset(heapvalue) = value.unpack() else {
                panic!("cannot unpack non-attrset value in pattern parameter");
            };
            let mut scope = LazyMap::new();
            let mut found_keys = 0;
            for (key, default) in entries.iter() {
                let value = (*heapvalue).get_lazy(key);
                found_keys += value.is_some() as usize;
                let Some(value) = value.cloned().or_else(|| {
                    default
                        .as_ref()
                        .map(|x| LazyValue::from_value(x.run(previous, &Value::NULL)))
                }) else {
                    panic!("missing pattern entry");
                };
                scope.set(key.to_string(), value);
            }
            if !ignore_unmatched {
                assert_eq!(found_keys, (*heapvalue).values.len());
            }
            if let Some(binding) = binding {
                scope.set(binding.to_string(), value);
            }
            // scope.get("hello");
            let mut s = String::new();
            UnpackedValue::fmt_attrset_display(&scope, 0, &mut s, true).unwrap();
            println!("{s}");
            Box::leak(Box::new(Scope {
                values: scope,
                previous,
            }))
        }
    }
}

unsafe extern "C" fn scope_create(previous: *mut Scope) -> *mut Scope {
    Box::leak(Box::new(Scope {
        values: LazyMap::new(),
        previous,
    }))
}

unsafe extern "C" fn scope_set(
    scope: *mut Scope,
    key: *const u8,
    key_len: usize,
    value: *const Executable,
) {
    let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key, key_len));
    Rc::increment_strong_count(value);
    (*scope).values.set(
        name.to_string(),
        LazyValue::from_jit(scope, Rc::from_raw(value)),
    )
}

unsafe extern "C" fn scope_inherit_from(
    scope: *mut Scope,
    from: *const Executable,
    what: *const String,
    whatn: usize,
) {
    let what = std::slice::from_raw_parts(what, whatn);
    for key in what {
        Rc::increment_strong_count(from);
        let from = Rc::from_raw(from);
        (*scope).values.set(
            key.to_string(),
            LazyValue::from_closure(move || {
                let value = from.run(scope, &Value::NULL).unpack();
                let UnpackedValue::Attrset(attrs) = value else {
                    panic!("scope_inherit_from: not an attrset");
                };
                (*attrs).get(key).unwrap()
            }),
        )
    }
}

unsafe extern "C" fn scope_inherit_parent(
    scope: *mut Scope,
    from: *mut Scope,
    what: *const String,
    whatn: usize,
) {
    let what = std::slice::from_raw_parts(what, whatn);
    for key in what {
        (*scope)
            .values
            .set(key.to_string(), Scope::lookup(from, key).clone())
    }
}

unsafe extern "C" fn attrset_create() -> Value {
    UnpackedValue::Attrset(HeapValue::new(LazyMap::new())).pack()
}

unsafe extern "C" fn attrset_set(
    map: &mut LazyMap,
    scope: *mut Scope,
    value: *const Executable,
    name: Value,
) {
    let UnpackedValue::String(name) = name.unpack() else {
        panic!("SetAttr called with non-string name")
    };
    let name = unsafe { (*name).as_str() };
    Rc::increment_strong_count(value);
    map.set(
        name.to_string(),
        LazyValue::from_jit(scope, Rc::from_raw(value)),
    );
}

unsafe extern "C" fn attrset_inherit_from(
    attrset: &mut LazyMap,
    from: &mut LazyMap,
    what: *const String,
    whatn: usize,
) {
    let what = std::slice::from_raw_parts(what, whatn);
    for key in what {
        attrset.set(key.to_string(), (*from).get_lazy(key).unwrap().clone())
    }
}

unsafe extern "C" fn attrset_inherit_parent(
    attrset: &mut LazyMap,
    from: *mut Scope,
    what: *const String,
    whatn: usize,
) {
    let what = std::slice::from_raw_parts(what, whatn);
    for key in what {
        println!("inherit {key}");
        attrset.set(key.to_string(), Scope::lookup_lazy(from, key));
        println!("inherited {key}");
    }
}

unsafe extern "C" fn attrset_get(map: &mut LazyMap, name: Value) -> Value {
    let UnpackedValue::String(name) = name.unpack() else {
        panic!("GetAttr called with non-string name")
    };
    map.get(unsafe { (*name).as_str() }).unwrap()
}

unsafe extern "C" fn attrset_get_or_insert_attrset(map: &mut LazyMap, name: Value) -> Value {
    let UnpackedValue::String(name) = name.unpack() else {
        panic!("GetAttr called with non-string name")
    };
    map.get_lazy_or_insert_with(unsafe { (*name).to_string() }, || {
        UnpackedValue::new_attrset(LazyMap::new()).into()
    })
    .evaluate()
    .clone()
}

unsafe extern "C" fn attrset_hasattr(map: &mut LazyMap, name: Value) -> Value {
    let UnpackedValue::String(name) = name.unpack() else {
        panic!("HasAttr called with non-string name")
    };
    UnpackedValue::Bool(map.get_lazy(unsafe { (*name).as_str() }).is_some()).pack()
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

unsafe extern "C" fn list_append_value(
    list: &mut Vec<LazyValue>,
    scope: *mut Scope,
    executable: *const Executable,
) {
    unsafe {
        Rc::increment_strong_count(executable);
        list.push(LazyValue::from_jit(scope, Rc::from_raw(executable)));
    }
}

unsafe extern "C" fn list_concat(a: Value, b: Value) -> Value {
    if let (UnpackedValue::List(a), UnpackedValue::List(b)) = (a.unpack(), b.unpack()) {
        UnpackedValue::new_list((*a).iter().chain((*b).iter()).cloned().collect()).pack()
    } else {
        panic!("concat attempted on non-list operands")
    }
}

unsafe extern "C" fn asm_panic(msg: *const i8) {
    panic!("[JITPANIC] {}", CStr::from_ptr(msg).to_string_lossy());
}

enum Parameter {
    Ident(String),
    Pattern {
        // FIXME: make this option an Option<Program> instead of compiling it during parsing
        entries: Vec<(String, Option<Executable>)>,
        binding: Option<String>,
        ignore_unmatched: bool,
    },
}

impl Program {
    fn new() -> Self {
        Self { operations: vec![] }
    }

    fn compile(self, param: Option<Parameter>) -> Result<Executable, IcedError> {
        let debug_header = format!("{self:?}");

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
                    asm.mov(rax, $function as u64)?;
                    call!(reg rax)
                };
            }

            if let Some(param) = param {
                // FIXME: leak
                let param = Box::leak(Box::new(param)) as *const Parameter;
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
                asm.mov(rdx, param as u64)?;

                call!(rust scope_function_scope);
                asm.mov(r15, rax)?;
            }

            macro_rules! impl_binary_operator {
                ($(int => $(retag($tag: ident))? $if_int: block,)? $(bool => $if_bool: expr,)? fallback = $other: expr) => {{
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

            // TODO: something like this but maybe use r14 as a temporary register?
            //       (so that one does not forget and just overwrite it)
            // macro_rules! check {
            //     (Attrset $reg: ident else => $else: ident) => {
            //             asm.mov(rbx, rdi)?;
            //             asm.and(rbx, VALUE_TAG_MASK as i32)?;
            //             asm.cmp(bl, ValueKind::Attrset as u32)?;
            //             asm.jne($else)?;
            //             asm.cmp(rdi, 0)?; // attrset tag also applies to null
            //             asm.je($else)?;
            //     };
            //     ($tag: ident $reg: ident else => $else: ident) => {
            //             asm.mov(rbx, rdi)?;
            //             asm.and(rbx, VALUE_TAG_MASK as i32)?;
            //             asm.cmp(bl, ValueKind::$tag as u32)?;
            //             asm.jne($else)?;
            //     }
            // }

            for op in self.operations.into_iter() {
                match op {
                    Operation::Push(value) => {
                        asm.mov(rax, value.0)?;
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::PushFunction(exec) => {
                        // FIXME: leak
                        let raw = Rc::into_raw(Rc::new(exec));
                        asm.mov(rdi, raw as u64)?;
                        asm.mov(rsi, r15)?;
                        call!(rust create_function_value);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::Add => impl_binary_operator!(
                        int => retag(Integer) { asm.add(rdi, rsi)?; rdi },
                        fallback = Value::add
                    ),
                    Operation::Sub => impl_binary_operator!(
                        int => retag(Integer) { asm.sub(rdi, rsi)?; rdi },
                        fallback = Value::sub
                    ),
                    Operation::Mul => impl_binary_operator!(
                        int => retag(Integer) { asm.imul_2(rdi, rsi)?; rdi },
                        fallback = Value::mul
                    ),
                    Operation::Div => impl_binary_operator!(
                        int => retag(Integer) {
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
                    Operation::Or => impl_binary_operator!(
                        bool => asm.or(rdi, rsi)?,
                        fallback = Value::or
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
                    Operation::CreateAttrset => {
                        call!(rust attrset_create);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::SetAttrpath(components, value_program) => {
                        // FIXME: leak
                        let raw = Rc::into_raw(Rc::new(value_program.compile(None)?));

                        assert!(stack_values > components);
                        asm.mov(rdi, qword_ptr(rsp + (8 * components)))?;

                        let mut not_an_attrset = asm.create_label();
                        for _ in 0..(components - 1) {
                            asm.mov(rbx, rdi)?;
                            asm.and(rbx, VALUE_TAG_MASK as i32)?;
                            asm.cmp(bl, ValueKind::Attrset as u32)?;
                            asm.jne(not_an_attrset)?;
                            asm.cmp(rdi, 0)?; // attrset tag also applies to null
                            asm.je(not_an_attrset)?;

                            asm.add(
                                rdi,
                                offset_of!(HeapValue<LazyMap>, value) as i32
                                    - ValueKind::Attrset as i32,
                            )?;

                            asm.pop(rsi)?;
                            stack_values -= 1;
                            call!(rust attrset_get_or_insert_attrset);
                            asm.mov(rdi, rax)?;
                        }

                        asm.mov(rbx, rdi)?;
                        asm.and(rbx, VALUE_TAG_MASK as i32)?;
                        asm.cmp(bl, ValueKind::Attrset as u32)?;
                        asm.jne(not_an_attrset)?;
                        asm.cmp(rdi, 0)?; // attrset tag also applies to null
                        asm.je(not_an_attrset)?;

                        asm.add(
                            rdi,
                            offset_of!(HeapValue<LazyMap>, value) as i32
                                - ValueKind::Attrset as i32,
                        )?;
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
                    Operation::InheritAttrs(from_stack, names) => {
                        let names = Vec::leak(names);
                        let mut not_an_attrset = asm.create_label();
                        let mut end = asm.create_label();
                        if from_stack {
                            assert!(stack_values >= 1);
                            asm.pop(rsi)?;
                            stack_values -= 1;

                            asm.mov(rbx, rsi)?;
                            asm.and(rbx, VALUE_TAG_MASK as i32)?;
                            asm.cmp(bl, ValueKind::Attrset as u32)?;
                            asm.jne(not_an_attrset)?;
                            asm.cmp(rsi, 0)?; // attrset tag also applies to null
                            asm.je(not_an_attrset)?;

                            asm.add(
                                rsi,
                                offset_of!(HeapValue<LazyMap>, value) as i32
                                    - ValueKind::Attrset as i32,
                            )?;
                        } else {
                            asm.mov(rsi, r15)?;
                        }

                        assert!(stack_values >= 1);
                        asm.mov(rdi, qword_ptr(rsp))?;

                        asm.mov(rbx, rdi)?;
                        asm.and(rbx, VALUE_TAG_MASK as i32)?;
                        asm.cmp(bl, ValueKind::Attrset as u32)?;
                        asm.jne(not_an_attrset)?;
                        asm.cmp(rdi, 0)?; // attrset tag also applies to null
                        asm.je(not_an_attrset)?;

                        asm.add(
                            rdi,
                            offset_of!(HeapValue<LazyMap>, value) as i32
                                - ValueKind::Attrset as i32,
                        )?;
                        asm.mov(rdx, names.as_ptr() as u64)?;
                        asm.mov(rcx, names.len() as u64)?;

                        if from_stack {
                            call!(rust attrset_inherit_from);
                        } else {
                            call!(rust attrset_inherit_parent);
                        }
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
                            offset_of!(HeapValue<LazyMap>, value) as i32
                                - ValueKind::Attrset as i32,
                        )?;

                        call!(rust attrset_get);
                        asm.push(rax)?;

                        let mut end = asm.create_label();
                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_attrset)?;

                        asm.mov(rdi, c"getattr called on non-attrset value".as_ptr() as u64)?;
                        call!(rust asm_panic);

                        asm.set_label(&mut end)?;
                        stack_values += 1;
                    }
                    Operation::HasAttrpath(len) => {
                        assert!(stack_values >= 2);
                        assert!(len == 1);
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
                            offset_of!(HeapValue<LazyMap>, value) as i32
                                - ValueKind::Attrset as i32,
                        )?;

                        call!(rust attrset_hasattr);
                        asm.push(rax)?;

                        let mut end = asm.create_label();
                        asm.jmp(end)?;

                        asm.set_label(&mut not_an_attrset)?;

                        asm.mov(rdi, c"hasattr called on non-attrset value".as_ptr() as u64)?;
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
                    Operation::PushList => {
                        call!(rust list_create);
                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    Operation::ListAppend(value_program) => {
                        // FIXME: leak
                        let raw = Rc::into_raw(Rc::new(value_program.compile(None)?));

                        assert!(stack_values >= 1);
                        asm.mov(rdi, qword_ptr(rsp))?;

                        let mut end = asm.create_label();
                        let mut not_a_list = asm.create_label();

                        asm.mov(rax, rdi)?;
                        asm.and(rax, VALUE_TAG_MASK as i32)?;
                        asm.cmp(al, ValueKind::List as u32)?;
                        asm.jne(not_a_list)?;

                        asm.add(
                            rdi,
                            offset_of!(HeapValue<Vec<LazyValue>>, value) as i32
                                - ValueKind::List as i32,
                        )?;
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
                        let name = String::leak(name);
                        // FIXME: leak
                        let raw = Rc::into_raw(Rc::new(value_program.compile(None)?));
                        asm.mov(rdi, r15)?;
                        asm.mov(rsi, name.as_ptr() as u64)?;
                        asm.mov(rdx, name.len() as u64)?;
                        asm.mov(rcx, raw as u64)?;
                        call!(rust scope_set);
                    }
                    Operation::ScopeInherit(from, names) => {
                        let names = Vec::leak(names);
                        let has_from = if let Some(program) = from {
                            // FIXME: leak
                            asm.mov(rsi, Rc::into_raw(Rc::new(program.compile(None)?)) as u64)?;
                            true
                        } else {
                            asm.mov(rsi, qword_ptr(r15 + offset_of!(Scope, previous)))?;
                            false
                        };
                        asm.mov(rdi, r15)?;
                        asm.mov(rdx, names.as_ptr() as u64)?;
                        asm.mov(rcx, names.len() as u64)?;
                        if has_from {
                            call!(rust scope_inherit_from);
                        } else {
                            call!(rust scope_inherit_parent);
                        }
                    }
                    Operation::ScopePop => {
                        // FIXME: make scopes rc or smth
                        asm.mov(r15, qword_ptr(r15 + offset_of!(Scope, previous)))?;
                    }
                    Operation::IfElse(if_true, if_false) => {
                        // FIXME: leak
                        let if_true = Rc::into_raw(Rc::new(if_true.compile(None)?));
                        let if_false = Rc::into_raw(Rc::new(if_false.compile(None)?));

                        assert!(stack_values >= 1);
                        asm.pop(rsi)?;
                        stack_values -= 1;

                        let mut end = asm.create_label();
                        let mut not_a_boolean = asm.create_label();
                        asm.mov(rdx, rsi)?;
                        asm.and(rdx, VALUE_TAG_MASK as i32)?;
                        asm.cmp(dl, ValueKind::Bool as i32)?;
                        asm.jne(not_a_boolean)?;

                        unsafe {
                            asm.mov(rax, (*if_false).code as u64)?;
                            asm.mov(rbx, (*if_true).code as u64)?;
                        }
                        asm.cmp(rsi, ValueKind::Bool as i32)?;
                        asm.cmovne(rax, rbx)?;

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

                        let mut end = asm.create_label();
                        let mut not_an_integer = asm.create_label();

                        asm.mov(rax, rdi)?;
                        asm.and(rax, VALUE_TAG_MASK as i32)?;
                        asm.cmp(rax, ValueKind::Integer as i32)?;
                        asm.jne(not_an_integer)?;

                        asm.shr(rax, 3)?;
                        asm.neg(rax)?;
                        asm.shl(rax, 3)?;
                        asm.or(rax, ValueKind::Integer as i32)?;

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
                    Operation::Concat => {
                        assert!(stack_values >= 2);
                        asm.pop(rdi)?;
                        asm.pop(rsi)?;
                        stack_values -= 2;

                        call!(rust list_concat);

                        asm.push(rax)?;
                        stack_values += 1;
                    }
                    x => {
                        let dbg = CString::new(format!("TODO: Operation::{x:?}"))
                            .unwrap()
                            .into_raw();
                        asm.mov(rdi, dbg as u64)?;
                        call!(rust asm_panic);
                    }
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

        println!("{}", debug_header);
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
    List(*mut HeapValue<Vec<LazyValue>>),
    Attrset(*mut HeapValue<LazyMap>),
    String(*mut HeapValue<String>),
    Function(*mut HeapValue<Function>),
    Path(*mut HeapValue<PathBuf>),
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

    fn new_path(value: PathBuf) -> UnpackedValue {
        UnpackedValue::Path(Box::leak(Box::new(HeapValue { refcount: 1, value })))
    }

    fn new_list(value: Vec<LazyValue>) -> UnpackedValue {
        UnpackedValue::List(HeapValue::new(value))
    }

    fn new_attrset(value: LazyMap) -> UnpackedValue {
        UnpackedValue::Attrset(Box::leak(Box::new(HeapValue { refcount: 1, value })))
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
            UnpackedValue::Path(_) => ValueKind::Path,
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
            Self::Path(ptr) => pack_ptr!(ptr, Path),
            Self::List(ptr) => pack_ptr!(ptr, List),
            Self::Attrset(ptr) => pack_ptr!(ptr, Attrset),
            Self::Function(ptr) => pack_ptr!(ptr, Function),
            Self::Null => Value::NULL,
        }
    }

    fn fmt_attrset_display(
        map: &LazyMap,
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
            if let Some(value) = value.as_maybe_evaluated() {
                if debug {
                    value.unpack().fmt_debug_rec(depth + 1, f)?
                } else {
                    value.unpack().fmt_display_rec(depth + 1, f)?
                }
            } else {
                write!(f, "...")?;
            }
            writeln!(f, ";")?;
        }
        for _ in 0..depth {
            write!(f, "  ")?;
        }
        write!(f, "}}")
    }

    fn fmt_list_display(
        list: &[LazyValue],
        depth: usize,
        f: &mut impl Write,
        debug: bool,
    ) -> std::fmt::Result {
        if list.len() > 2 {
            writeln!(f, "[")?;
        } else {
            write!(f, "[ ")?;
        }
        for value in list.iter() {
            if list.len() > 2 {
                for _ in 0..=depth {
                    write!(f, "  ")?;
                }
            }
            if let Some(value) = value.as_maybe_evaluated() {
                if debug {
                    value.unpack().fmt_debug_rec(depth + 1, f)?
                } else {
                    value.unpack().fmt_display_rec(depth + 1, f)?
                }
            } else {
                write!(f, "...")?;
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
            Self::Path(value) => write!(f, "{}", unsafe { (**value).display() }),
            Self::List(value) => Self::fmt_list_display(unsafe { &**value }, depth, f, false),
            Self::Attrset(value) => {
                Self::fmt_attrset_display(unsafe { &(**value).value }, depth, f, false)
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
            Self::Path(value) => {
                write!(f, "{}", unsafe { (**value).display() })
            }
            Self::List(value) => Self::fmt_list_display(unsafe { &**value }, depth, f, true),
            Self::Attrset(value) => {
                Self::fmt_attrset_display(unsafe { &(**value) }, depth, f, true)
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
            Self::Path(ptr) => clone_ptr!(ptr, Path),
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
    const TRUE: Value = Value(0b1000 | ValueKind::Bool as u64);
    const FALSE: Value = Value(ValueKind::Bool as u64);

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
            ValueKind::Path => unpack_ptr!(Path),
            ValueKind::List => unpack_ptr!(List),
            ValueKind::Attrset => unpack_ptr!(Attrset),
            ValueKind::Function => unpack_ptr!(Function),
            ValueKind::Null => UnpackedValue::Null,
        }
    }

    #[inline]
    unsafe fn leaking_copy(&self) -> Value {
        Value(self.0)
    }

    // TODO: macro all this stuff
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

    extern "C" fn or(self, other: Value) -> Value {
        match (self.unpack(), other.unpack()) {
            (UnpackedValue::Bool(a), UnpackedValue::Bool(b)) => UnpackedValue::Bool(a || b).pack(),
            (_, _) => panic!(
                "|| is not supported between values of type {} and {}",
                self.kind(),
                other.kind()
            ),
        }
    }
}

macro_rules! value_impl_number_comparison {
    ($name: ident, $operator: tt) => {
        extern "C" fn $name(self, other: Value) -> Value {
            match (self.unpack(), other.unpack()) {
                (UnpackedValue::Integer(a), UnpackedValue::Integer(b)) => {
                    UnpackedValue::Bool(a $operator b).pack()
                }
                (_, _) => panic!(
                    concat!(stringify!($operator), " is not supported between values of type {} and {}"),
                    self.kind(), other.kind()
                )
            }
        }
    };
}

impl Value {
    value_impl_number_comparison!(less, <);
    value_impl_number_comparison!(less_or_equal, <=);
    value_impl_number_comparison!(greater_or_equal, >=);
    value_impl_number_comparison!(greater, >);
    value_impl_number_comparison!(equal, ==);
    value_impl_number_comparison!(not_equal, !=);
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
    let mut values = LazyMap::new();

    macro_rules! extract_typed {
        ($builtin_name: literal, $what: ident($value: expr)) => {
            unsafe {&***extract_typed!($builtin_name, HeapValue<$what>($value))}
        };
        ($builtin_name: literal, HeapValue<$what: ident>($value: expr)) => {
            match &$value.unpack() {
                UnpackedValue::$what(ptr) => &(*ptr),
                other => panic!(
                    concat!("builtin ", $builtin_name, " expected {} but got {}"),
                    ValueKind::$what,
                    other.kind()
                ),
            }
        };
    }

    values.set(
        "builtins".to_string(),
        UnpackedValue::new_attrset({
            let mut builtins = LazyMap::new();

            builtins.set(
                "map".to_string(),
                UnpackedValue::new_function(|fun| {
                    let mapper = extract_typed!("map", Function(fun));
                    UnpackedValue::new_function(move |list| {
                        let list = extract_typed!("map", List(list));
                        UnpackedValue::List(HeapValue::new(
                            list.iter()
                                .map(|x| {
                                    let fun = fun.clone();
                                    x.map(move |x| unsafe { (mapper.call)(fun, x) })
                                })
                                .collect(),
                        ))
                        .pack()
                    })
                    .pack()
                }),
            );

            builtins.set(
                "trace".to_string(),
                UnpackedValue::new_function(|message| {
                    println!("trace: {message}");
                    UnpackedValue::new_function(|value| value).pack()
                }),
            );

            builtins
        }),
    );
    values.set("true".to_string(), UnpackedValue::Bool(true));
    values.set("false".to_string(), UnpackedValue::Bool(false));
    values.set("null".to_string(), UnpackedValue::Null);
    values.set(
        "import".to_string(),
        UnpackedValue::new_function(|value| {
            let path = extract_typed!("import", Path(value));
            import(path)
        }),
    );
    values.set(
        "__trace_shallow".to_string(),
        UnpackedValue::new_function(|value: Value| {
            println!("trace (shallow): {value:?}");
            UnpackedValue::new_function(|value| value).pack()
        }),
    );

    Scope {
        values,
        previous: std::ptr::null_mut(),
    }
}

thread_local! {
    static ROOT_SCOPE: Scope = create_root_scope();
}

fn import(path: &Path) -> Value {
    let expr = match std::fs::read_to_string(path) {
        Err(e) if e.to_string().contains("Is a directory") => {
            std::fs::read_to_string(path.join("default.nix"))
        }
        other => other,
    }
    .unwrap();

    let result = rnix::Root::parse(&expr);
    let root = result.tree();
    let mut root_block = Program { operations: vec![] };
    build_program(
        path.parent().unwrap(),
        root.expr().unwrap(),
        &mut root_block,
    );
    let compiled = root_block.compile(None).unwrap();
    ROOT_SCOPE.with(|root| compiled.run(root as *const _ as *mut _, &Value::NULL))
}

fn seq(value: &Value, deep: bool) {
    match value.unpack() {
        UnpackedValue::Integer(_) => (),
        UnpackedValue::Bool(_) => (),
        UnpackedValue::List(value) => {
            let list = unsafe { &mut **value };
            for value in list.iter() {
                let value = value.evaluate();
                if deep {
                    seq(value, deep);
                }
            }
        }
        UnpackedValue::Attrset(value) => {
            let map = unsafe { &mut **value };
            for (key, value) in map.iter() {
                println!("seq: evaluating key {key}");
                let value = value.evaluate();
                if deep {
                    seq(value, deep);
                }
            }
        }
        UnpackedValue::String(_) => (),
        UnpackedValue::Function(_) => (),
        UnpackedValue::Path(_) => (),
        UnpackedValue::Null => (),
    }
}

#[test]
fn pointer_packing_works() {
    let fake_ptr = 0b10101000 as *mut HeapValue<LazyMap>;
    assert!(matches!(
        UnpackedValue::Attrset(fake_ptr).pack().unpack(),
        UnpackedValue::Attrset(ptr) if ptr == fake_ptr
    ));
}

fn main() {
    let value = import(&PathBuf::from(std::env::args().nth(1).unwrap()));
    seq(&value, true);
    println!("{}", value);
}
