#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::fn_to_numeric_cast)]
#![allow(clippy::type_complexity)]
// This is likely to be stalibised soon
#![feature(offset_of_nested)]
// This can be replaced later
#![feature(offset_of_enum)]

use core::panic;
use std::{
    cell::UnsafeCell,
    collections::BTreeMap,
    fmt::{Debug, Display, Write},
    mem::{offset_of, MaybeUninit},
    ops::Deref,
    path::PathBuf,
    rc::Rc,
};

use compiler::Executable;
use dwarf::*;
use rnix::{
    ast::{AstToken, Attr, Attrpath, Expr, HasEntry, InterpolPart},
    TextRange,
};
use rowan::ast::AstNode; // NOTE: Importing this from rowan works but not from rnix...

#[derive(Debug)]
struct SourceFile {
    filename: String,
    content: String,
}

#[derive(Debug, Clone)]
struct SourceSpan {
    // TODO: this doesn't have to be stored per-SourceSpan
    //       (per-Executable instead?)
    file: Rc<SourceFile>,
    range: TextRange,
}

#[derive(Debug)]
enum Operation {
    Push(Value),
    // This is different from Push because this also sets the parent scope of the function
    PushFunction(Parameter, Program),
    CreateAttrset { rec: bool },
    InheritAttrs(Option<Program>, Vec<String>),
    SetAttrpath(usize, Program),
    GetAttrConsume { default: Option<Program> },
    HasAttrpath(usize),
    PushList,
    ListAppend(Program),
    StringCloneToMut,
    // TODO: Make construction Operations like StringAppend, SetAttrpath, InheritAttrs etc.
    //       unchecked at runtime (with optional checking for debugging?).
    StringMutAppend,
    StringToPath,
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
    SourceSpanPush(SourceSpan),
    SourceSpanPop,
}

#[derive(Debug)]
struct Program {
    operations: Vec<Operation>,
}

struct IRCompiler {
    working_directory: PathBuf,
    source_file: Rc<SourceFile>,
}

// TODO: Notes on building attrsets
// The reference nix implementation allows using attrpaths with the same prefix like this:
// {
//     a.a = 1;
//     a.b = 1;
// }
// This is simple when there is no dynamic components on the path.
// Fortunately, it is forbidden to do the same with dynamic components.
// This means that the fastest way to compile an attrset build would be to build out a tree
// structure from the attrpaths and probably compile known attributes statically into a base
// attribute set which would then be filled with values and possible dynamic attributes.
// I believe this would improve attrset building performance during runtime pretty significantly

fn attr_assert_const(attr: Attr) -> String {
    match attr {
        Attr::Ident(x) => x.ident_token().unwrap().green().text().to_string(),
        Attr::Dynamic(_) => todo!(),
        Attr::Str(_) => todo!(),
    }
}

impl IRCompiler {
    fn create_program(&self, expr: Expr) -> Program {
        let mut program = Program::new();
        self.build_program(expr, &mut program);
        program
    }

    fn build_attrpath(&self, attrpath: Attrpath, program: &mut Program) -> usize {
        let mut n = 0;
        for attr in attrpath.attrs().collect::<Vec<_>>().into_iter().rev() {
            match attr {
                Attr::Ident(x) => program.operations.push(Operation::Push(
                    UnpackedValue::new_string(x.ident_token().unwrap().green().text().to_string())
                        .pack(),
                )),
                Attr::Dynamic(x) => self.build_program(x.expr().unwrap(), program),
                Attr::Str(x) => self.build_string(x.normalized_parts(), |x| x, false, program),
            };
            n += 1;
        }
        n
    }

    fn build_string<Content>(
        &self,
        parts: impl IntoIterator<Item = InterpolPart<Content>>,
        content_to_text: impl Fn(&Content) -> &str,
        is_path: bool,
        program: &mut Program,
    ) {
        let mut parts = parts.into_iter();
        match parts.next() {
            None => {
                return program.operations.push(Operation::Push(
                    UnpackedValue::new_string(String::new()).pack(),
                ))
            }
            Some(InterpolPart::Literal(literal)) => program.operations.push(Operation::Push(
                UnpackedValue::new_string({
                    let literal = content_to_text(&literal).to_string();
                    if is_path {
                        // not an absolute path
                        if !literal.starts_with("/")
                            && !self.working_directory.as_os_str().is_empty()
                        {
                            format!("{}/{literal}", self.working_directory.to_str().unwrap())
                        } else {
                            literal
                        }
                    } else {
                        literal
                    }
                })
                .pack(),
            )),
            Some(InterpolPart::Interpolation(interpol)) => {
                self.build_program(interpol.expr().unwrap(), program);
            }
        }
        program.operations.push(Operation::StringCloneToMut);
        for part in parts {
            match part {
                InterpolPart::Literal(literal) => program.operations.push(Operation::Push(
                    UnpackedValue::new_string(content_to_text(&literal).to_string()).pack(),
                )),
                InterpolPart::Interpolation(interpol) => {
                    self.build_program(interpol.expr().unwrap(), program);
                }
            }
            program.operations.push(Operation::StringMutAppend);
        }
        if is_path {
            program.operations.push(Operation::StringToPath);
        }
    }

    fn build_program(&self, expr: Expr, program: &mut Program) {
        program
            .operations
            .push(Operation::SourceSpanPush(SourceSpan {
                file: self.source_file.clone(),
                range: expr.syntax().text_range(),
            }));
        match expr {
            Expr::Apply(x) => {
                self.build_program(x.argument().unwrap(), program);
                self.build_program(x.lambda().unwrap(), program);
                program.operations.push(Operation::Apply);
            }
            // TODO:
            Expr::Assert(x) => self.build_program(x.body().unwrap(), program),
            Expr::Error(_) => todo!(),
            Expr::IfElse(x) => {
                self.build_program(x.condition().unwrap(), program);
                let if_true = self.create_program(x.body().unwrap());
                let if_false = self.create_program(x.else_body().unwrap());
                program
                    .operations
                    .push(Operation::IfElse(if_true, if_false))
            }
            Expr::Select(x) => {
                self.build_program(x.expr().unwrap(), program);
                for attr in x.attrpath().unwrap().attrs() {
                    program.operations.push(Operation::Push(
                        UnpackedValue::new_string(attr.to_string()).pack(),
                    ));
                    program.operations.push(Operation::GetAttrConsume {
                        default: x.default_expr().map(|expr| self.create_program(expr)),
                    })
                }
            }
            Expr::Str(x) => self.build_string(x.normalized_parts(), |x| x, false, program),
            Expr::Path(x) => self.build_string(x.parts(), |x| x.syntax().text(), true, program),
            Expr::Literal(x) => match x.kind() {
                rnix::ast::LiteralKind::Float(_) => todo!(),
                rnix::ast::LiteralKind::Integer(x) => {
                    program.operations.push(Operation::Push(
                        UnpackedValue::Integer(x.value().unwrap() as i32).pack(),
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
                                    x.default().map(|x| self.create_program(x)),
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
                program
                    .operations
                    .push(Operation::PushFunction(param, self.create_program(body)))
            }
            Expr::LegacyLet(_) => todo!(),
            Expr::LetIn(x) => {
                program.operations.push(Operation::ScopePush);
                for inherit in x.inherits() {
                    let source = inherit
                        .from()
                        .and_then(|f| f.expr())
                        .map(|source| self.create_program(source));
                    program.operations.push(Operation::ScopeInherit(
                        source,
                        inherit.attrs().map(attr_assert_const).collect(),
                    ));
                }
                for entry in x.attrpath_values() {
                    let mut it = entry.attrpath().unwrap().attrs();
                    let name = attr_assert_const(it.next().unwrap());
                    let value = entry.value().unwrap();
                    let value_program = self.create_program(value);
                    program
                        .operations
                        .push(Operation::ScopeSet(name, value_program));
                }
                self.build_program(x.body().unwrap(), program);
                program.operations.push(Operation::ScopePop);
            }
            Expr::List(x) => {
                program.operations.push(Operation::PushList);
                for item in x.items() {
                    let value_program = self.create_program(item);
                    program
                        .operations
                        .push(Operation::ListAppend(value_program));
                }
            }
            Expr::BinOp(x) => {
                self.build_program(x.lhs().unwrap(), program);
                self.build_program(x.rhs().unwrap(), program);
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
            Expr::Paren(x) => self.build_program(x.expr().unwrap(), program),
            Expr::AttrSet(x) => {
                program.operations.push(Operation::CreateAttrset {
                    rec: x.rec_token().is_some(),
                });
                for inherit in x.inherits() {
                    let source = inherit
                        .from()
                        .and_then(|f| f.expr())
                        .map(|source| self.create_program(source));
                    program.operations.push(Operation::InheritAttrs(
                        source,
                        inherit.attrs().map(attr_assert_const).collect(),
                    ));
                }
                for keyvalue in x.attrpath_values() {
                    let path = keyvalue.attrpath().unwrap();
                    let value = self.create_program(keyvalue.value().unwrap());
                    let n = self.build_attrpath(path, program);
                    program.operations.push(Operation::SetAttrpath(n, value));
                }
            }
            Expr::UnaryOp(x) => {
                self.build_program(x.expr().unwrap(), program);
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
                self.build_program(x.expr().unwrap(), program);
                let attrpath = x.attrpath().unwrap();
                let n = self.build_attrpath(attrpath, program);
                program.operations.push(Operation::HasAttrpath(n))
            }
            _ => todo!(),
        }
        program.operations.push(Operation::SourceSpanPop);
    }

    pub fn compile(
        working_directory: PathBuf,
        filename: String,
        file_content: String,
        expression: Expr,
    ) -> Program {
        IRCompiler {
            working_directory,
            source_file: Rc::new(SourceFile {
                filename,
                content: file_content,
            }),
        }
        .create_program(expression)
    }
}

type ValueMap = BTreeMap<String, Value>;
type ValueList = Vec<Value>;

enum ScopeStorage {
    // Used for recursive attribute sets
    #[allow(dead_code)] // Used dynamically via JIT
    Attrset,
    Scope(ValueMap),
}

#[repr(C)]
struct Scope {
    values: *const ValueMap,
    previous: *mut Scope,
    storage: ScopeStorage,
}

impl Scope {
    fn from_map(map: ValueMap, previous: *mut Scope) -> *mut Scope {
        unsafe {
            let scope_ptr = Box::leak(Box::new(Scope {
                values: std::ptr::null(),
                storage: ScopeStorage::Scope(map),
                previous,
            })) as *mut Scope;
            (*scope_ptr).values =
                (scope_ptr as *const u8).add(offset_of!(Scope, storage.Scope.0)) as *const _;
            scope_ptr
        }
    }

    fn from_attrs(map: Rc<UnsafeCell<ValueMap>>, previous: *mut Scope) -> *mut Scope {
        Box::leak(Box::new(Scope {
            values: unsafe { (*Rc::<UnsafeCell<ValueMap>>::into_raw(map)).get() },
            previous,
            storage: ScopeStorage::Attrset,
        }))
    }

    unsafe fn lookup(mut scope: *mut Scope, name: &str) -> Value {
        loop {
            let mut scope_debug = String::new();
            UnpackedValue::fmt_attrset_display(&*(*scope).values, 0, &mut scope_debug, true)
                .unwrap();
            // eprintln!("{scope:?} {name} {}", scope_debug);
            match (*scope).get(name) {
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
}

impl Drop for Scope {
    fn drop(&mut self) {
        if matches!(self.storage, ScopeStorage::Attrset) {
            unsafe { Rc::decrement_strong_count(self.values) }
        }
    }
}

impl Deref for Scope {
    type Target = ValueMap;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.values }
    }
}
#[derive(Debug)]
enum Parameter {
    Ident(String),
    Pattern {
        entries: Vec<(String, Option<Program>)>,
        binding: Option<String>,
        ignore_unmatched: bool,
    },
}

mod dwarf;

mod exception;
mod unwind;
use exception::*;

mod compiler;

impl Program {
    fn new() -> Self {
        Self { operations: vec![] }
    }
}

const VALUE_TAG_WIDTH: u8 = 3;
const VALUE_TAG_MASK: u64 = 0b111;
const ATTRSET_TAG_MASK: u64 = 0b111 | (0b11 << 62);
const ATTRSET_TAG_ATTRSET: u64 = ValueKind::Attrset as u64;
const ATTRSET_TAG_LAZY: u64 = ValueKind::Attrset as u64 | (0b11 << 62);
const ATTRSET_TAG_NULL: u64 = ValueKind::Attrset as u64 | (0b10 << 62);

#[repr(u8)]
#[derive(Debug, PartialEq, Eq)]
enum ValueKind {
    // TODO: consolidate this into some Pointer variant that stores
    //       the pointer type in the upper byte (canonical address form x86_64 quirk)
    // 0b11...ValueKind::Attrset = Lazy ptr
    // 0b10...ValueKind::Attrset = NULL
    // 0b00...ValueKind::Attrset = Attrset
    Attrset = 0, // also used as a tag for null
    Function,
    Integer,
    Double,
    String,
    Path,
    List,
    Bool,
    // NOTE: These are no longer tag values, instead these are packed into Attrset tags
    // FIXME: Most Values in the whole program are going to be hidden behind this extra layer of
    //        LazyValue indirection, maybe there is a way to make the potential performance penalty
    //        smaller?
    Lazy, // a pointer to LazyValue
    Null, // all zeroes: an attrset which is a null pointer
}

struct Function {
    call: unsafe extern "C-unwind" fn(Value, Value) -> Value,
    _executable: Option<Rc<Executable>>,
    builtin_closure: Option<Box<dyn FnMut(Value) -> Value>>,
    parent_scope: *const Scope,
}

#[derive(Debug)]
enum LazyValueImpl {
    Evaluated(Value),
    LazyJIT((*mut Scope, Rc<Executable>)),
    LazyBuiltin(MaybeUninit<Box<dyn FnOnce() -> Value>>),
}

#[derive(Debug, Clone)]
struct LazyValue(Rc<UnsafeCell<LazyValueImpl>>);

thread_local! {
    static LAZY_DEPTH: UnsafeCell<usize> = const { UnsafeCell::new(0) };
}

impl LazyValueImpl {
    fn evaluate(&mut self) -> &Value {
        match self {
            LazyValueImpl::Evaluated(value) => value,
            LazyValueImpl::LazyJIT((scope, executable)) => {
                LAZY_DEPTH.with(|x| unsafe {
                    let depth = &mut *x.get();
                    *depth += 1;
                    if *depth > 100 {
                        panic!("max lazy JIT depth exceeded");
                    };
                });
                // eprintln!("evaluating lazy JIT executable");
                let result = (**executable).run(*scope, &Value::NULL).into_evaluated();
                *self = LazyValueImpl::Evaluated(result);
                // eprintln!("evaluated lazy JIT executable");
                LAZY_DEPTH.with(|x| unsafe {
                    let depth = &mut *x.get();
                    *depth -= 1;
                });
                match self {
                    LazyValueImpl::Evaluated(x) => x,
                    _ => unreachable!(),
                }
            }
            LazyValueImpl::LazyBuiltin(value) => {
                let result =
                    unsafe { (std::mem::replace(value, MaybeUninit::uninit()).assume_init())() }
                        .into_evaluated();
                *self = LazyValueImpl::Evaluated(result);
                match self {
                    LazyValueImpl::Evaluated(x) => x,
                    _ => unreachable!(),
                }
            }
        }
    }

    #[inline]
    fn as_maybe_evaluated(&self) -> Option<&Value> {
        match self {
            LazyValueImpl::Evaluated(value) => Some(value),
            _ => None,
        }
    }
}

impl LazyValue {
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

    fn evaluate(&self) -> &Value {
        unsafe { &mut *self.0.get() }.evaluate()
    }

    #[inline]
    fn as_maybe_evaluated(&self) -> Option<&Value> {
        unsafe { (*self.0.get()).as_maybe_evaluated() }
    }
}

#[derive(Clone)]
enum UnpackedValue {
    Integer(i32),
    Bool(bool),
    List(Rc<UnsafeCell<ValueList>>),
    Attrset(Rc<UnsafeCell<ValueMap>>),
    String(Rc<String>),
    Function(Rc<Function>),
    Path(Rc<PathBuf>),
    Lazy(LazyValue),
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
            ValueKind::Lazy => f.write_str("<lazy>"),
            ValueKind::Null => f.write_str("null"),
            Self::Function => f.write_str("function"),
        }
    }
}

impl From<i32> for UnpackedValue {
    fn from(value: i32) -> Self {
        UnpackedValue::Integer(value)
    }
}

impl From<bool> for UnpackedValue {
    fn from(value: bool) -> Self {
        UnpackedValue::Bool(value)
    }
}

impl From<String> for UnpackedValue {
    fn from(value: String) -> Self {
        UnpackedValue::new_string(value)
    }
}

impl From<LazyValue> for UnpackedValue {
    fn from(value: LazyValue) -> Self {
        UnpackedValue::Lazy(value)
    }
}

impl FromIterator<Value> for UnpackedValue {
    fn from_iter<T: IntoIterator<Item = Value>>(iter: T) -> Self {
        UnpackedValue::new_list(iter.into_iter().collect())
    }
}

impl FromIterator<(String, Value)> for UnpackedValue {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        UnpackedValue::new_attrset(iter.into_iter().collect())
    }
}

impl<T: Into<UnpackedValue>> From<T> for Value {
    fn from(value: T) -> Self {
        value.into().pack()
    }
}

impl<T> FromIterator<T> for Value
where
    UnpackedValue: FromIterator<T>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        UnpackedValue::from_iter(iter).pack()
    }
}

unsafe extern "C-unwind" fn call_builtin(fun: Value, value: Value) -> Value {
    let this = (fun.0 & !VALUE_TAG_MASK) as *mut Function;
    // eprintln!(
    //     "call_builtin HeapValue:{:?} arg:{value:?}",
    //     this as *const _
    // );
    ((*this).builtin_closure.as_mut().unwrap_unchecked())(value)
}

impl UnpackedValue {
    fn new_string(value: String) -> UnpackedValue {
        UnpackedValue::String(Rc::new(value))
    }

    fn new_path(value: PathBuf) -> UnpackedValue {
        UnpackedValue::Path(Rc::new(value))
    }

    fn new_list(value: ValueList) -> UnpackedValue {
        UnpackedValue::List(Rc::new(UnsafeCell::new(value)))
    }

    fn new_attrset(value: ValueMap) -> UnpackedValue {
        UnpackedValue::Attrset(Rc::new(UnsafeCell::new(value)))
    }

    fn new_function(function: impl FnMut(Value) -> Value + 'static) -> UnpackedValue {
        UnpackedValue::Function(Rc::new(Function {
            call: call_builtin,
            _executable: None,
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
            UnpackedValue::Lazy(_) => ValueKind::Lazy,
            UnpackedValue::Null => ValueKind::Null,
        }
    }

    fn pack(self) -> Value {
        macro_rules! pack_ptr {
            ($ptr: ident, $kind: ident) => {{
                Value((Rc::into_raw($ptr) as u64) | (ValueKind::$kind as u64))
            }};
        }
        match self {
            Self::Integer(x) => unsafe {
                Value(
                    ((std::mem::transmute::<i32, u32>(x) as u64) << VALUE_TAG_WIDTH)
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
            Self::Lazy(ptr) => Value(Rc::into_raw(ptr.0) as u64 | ATTRSET_TAG_LAZY),
            Self::Null => Value::NULL,
        }
    }

    fn evaluate(self) -> UnpackedValue {
        if let UnpackedValue::Lazy(lazy) = self {
            lazy.evaluate().unpack()
        } else {
            self
        }
    }

    fn fmt_attrset_display(
        map: &ValueMap,
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
                value.clone().unpack().fmt_debug_rec(depth + 1, f)?
            } else {
                value.clone().unpack().fmt_display_rec(depth + 1, f)?
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
        for value in list.iter() {
            if list.len() > 2 {
                for _ in 0..=depth {
                    write!(f, "  ")?;
                }
            }
            if debug {
                value.clone().unpack().fmt_debug_rec(depth + 1, f)?
            } else {
                value.clone().unpack().fmt_display_rec(depth + 1, f)?
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
                    write!(f, "{}", value)
                } else {
                    write!(f, "{:?}", value)
                }
            }
            Self::Path(value) => write!(f, "{}", value.display()),
            Self::List(value) => {
                Self::fmt_list_display(unsafe { &*(value.get()) }, depth, f, false)
            }
            Self::Attrset(value) => {
                Self::fmt_attrset_display(unsafe { &*(value.get()) }, depth, f, false)
            }
            Self::Function(_) => write!(f, "«lambda»"),
            Self::Lazy(x) => {
                if let Some(x) = x.as_maybe_evaluated() {
                    x.clone().unpack().fmt_display_rec(depth, f)
                } else {
                    write!(f, "...")
                }
            }
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
                write!(f, "{:?}", value.as_str())
            }
            Self::Path(value) => {
                write!(f, "{}", value.display())
            }
            Self::List(value) => Self::fmt_list_display(unsafe { &*(value.get()) }, depth, f, true),
            Self::Attrset(value) => {
                Self::fmt_attrset_display(unsafe { &*(value.get()) }, depth, f, true)
            }
            Self::Function(_) => write!(f, "«lambda»"),
            Self::Lazy(x) => {
                if let Some(x) = x.as_maybe_evaluated() {
                    write!(f, "«lazy ")?;
                    x.unpack().fmt_debug_rec(depth, f)?;
                    write!(f, "»")
                } else {
                    write!(f, "«lazy»")
                }
            }
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

// https://github.com/SerenityOS/serenity/blob/1d29f9081fc243407e4902713d3c7fc0ee69650c/Userland/Libraries/LibJS/Runtime/Value.h
#[repr(transparent)]
struct Value(u64);

impl Value {
    const NULL: Value = Value(ATTRSET_TAG_NULL);
    const TRUE: Value = Value(0b1000 | ValueKind::Bool as u64);
    const FALSE: Value = Value(ValueKind::Bool as u64);

    fn kind(&self) -> ValueKind {
        match self.0 & ATTRSET_TAG_MASK {
            ATTRSET_TAG_NULL => return ValueKind::Null,
            ATTRSET_TAG_LAZY => return ValueKind::Lazy,
            _ => (),
        }
        let value = self.0 & VALUE_TAG_MASK;
        assert!(value <= ValueKind::Bool as u64);
        unsafe { std::mem::transmute(value as u8) }
    }

    fn unpack(&self) -> UnpackedValue {
        macro_rules! unpack_ptr {
            ($kind: ident) => {
                UnpackedValue::$kind(unsafe {
                    let raw = (self.0 & !VALUE_TAG_MASK) as *mut _;
                    Rc::increment_strong_count(raw);
                    Rc::from_raw(raw)
                })
            };
        }
        match self.kind() {
            ValueKind::Integer => UnpackedValue::Integer(unsafe {
                std::mem::transmute((self.0 >> VALUE_TAG_WIDTH) as u32)
            }),
            ValueKind::Double => todo!(),
            ValueKind::Bool => UnpackedValue::Bool(self.0 == 0b1000 | (ValueKind::Bool as u64)),
            ValueKind::String => unpack_ptr!(String),
            ValueKind::Path => unpack_ptr!(Path),
            ValueKind::List => unpack_ptr!(List),
            ValueKind::Attrset => unpack_ptr!(Attrset),
            ValueKind::Function => unpack_ptr!(Function),
            // FIXME: this pointer may not be in x86_64 canonical form
            ValueKind::Lazy => unsafe {
                UnpackedValue::Lazy(LazyValue({
                    let raw = (self.0 & !ATTRSET_TAG_MASK) as *const _;
                    Rc::increment_strong_count(raw);
                    Rc::from_raw(raw)
                }))
            },
            ValueKind::Null => UnpackedValue::Null,
        }
    }

    fn into_unpacked(self) -> UnpackedValue {
        macro_rules! unpack_ptr {
            ($kind: ident) => {
                UnpackedValue::$kind(unsafe { Rc::from_raw((self.0 & !VALUE_TAG_MASK) as *mut _) })
            };
        }
        match self.kind() {
            ValueKind::Integer => UnpackedValue::Integer(unsafe {
                std::mem::transmute((self.0 >> VALUE_TAG_WIDTH) as u32)
            }),
            ValueKind::Double => todo!(),
            ValueKind::Bool => UnpackedValue::Bool(self.0 == 0b1000 | (ValueKind::Bool as u64)),
            ValueKind::String => unpack_ptr!(String),
            ValueKind::Path => unpack_ptr!(Path),
            ValueKind::List => unpack_ptr!(List),
            ValueKind::Attrset => unpack_ptr!(Attrset),
            ValueKind::Function => unpack_ptr!(Function),
            // FIXME: this pointer may not be in x86_64 canonical form
            ValueKind::Lazy => unsafe {
                UnpackedValue::Lazy(LazyValue(Rc::from_raw(
                    (self.0 & !ATTRSET_TAG_MASK) as *const _,
                )))
            },
            ValueKind::Null => UnpackedValue::Null,
        }
    }

    unsafe fn increase_refcount(&self) {
        macro_rules! incref {
            ($x: expr) => {
                unsafe { Rc::increment_strong_count(Rc::as_ptr($x)) }
            };
        }
        match &self.unpack() {
            UnpackedValue::Integer(_) => (),
            UnpackedValue::Bool(_) => (),
            UnpackedValue::List(x) => incref!(x),
            UnpackedValue::Attrset(x) => incref!(x),
            UnpackedValue::String(x) => incref!(x),
            UnpackedValue::Function(x) => incref!(x),
            UnpackedValue::Path(x) => incref!(x),
            UnpackedValue::Lazy(x) => incref!(&x.0),
            UnpackedValue::Null => (),
        }
    }

    fn evaluate(&self) -> &Value {
        if (self.0 & ATTRSET_TAG_MASK) == ATTRSET_TAG_LAZY {
            unsafe { (*((self.0 & !ATTRSET_TAG_MASK) as *mut LazyValueImpl)).evaluate() }
        } else {
            self
        }
    }

    fn evaluate_mut(&mut self) -> &Value {
        if (self.0 & ATTRSET_TAG_MASK) == ATTRSET_TAG_LAZY {
            unsafe {
                LazyValue(Rc::from_raw((self.0 & !ATTRSET_TAG_MASK) as *mut _))
                    .evaluate()
                    .clone_into(self)
            }
        }
        self
    }

    fn into_evaluated(self) -> Value {
        if (self.0 & ATTRSET_TAG_MASK) == ATTRSET_TAG_LAZY {
            unsafe {
                (*((self.0 & !ATTRSET_TAG_MASK) as *mut LazyValueImpl))
                    .evaluate()
                    .clone()
            }
        } else {
            self
        }
    }

    fn lazy_map(self, fun: impl FnOnce(Value) -> Value + 'static) -> Value {
        UnpackedValue::Lazy(LazyValue::from_closure(|| fun(self))).pack()
    }

    #[inline]
    unsafe fn leaking_copy(&self) -> Value {
        Value(self.0)
    }

    pub fn is_true(&self) -> bool {
        self.0 == Self::TRUE.0
    }

    pub fn is_false(&self) -> bool {
        self.0 == Self::FALSE.0
    }
}

macro_rules! value_impl_binary_operator {
    (
        $name: ident,
        $op: tt,
        $(
            $kind: ident($a: ident, $b: ident) => $expr: expr
        ),*
    ) => {
        extern "C-unwind" fn $name(self, other: Value) -> Value {
            match (self.into_unpacked(), other.into_unpacked()) {
                $((UnpackedValue::$kind($a), UnpackedValue::$kind($b)) => $expr.into(),)*
                (a, b) => throw!(
                    concat!(stringify!($op), " is not supported between values of type {} and {}"),
                    a.kind(), b.kind()
                )
            }
        }
    };
}

macro_rules! value_impl_binary_operators {
    {
        $($op: tt $name: ident($a: ident, $b: ident) {
            $($kind: ident => $expr: expr),*
        };)*
    } => {
        $(
            value_impl_binary_operator!(
                $name,
                $op,
                $($kind($a, $b) => $expr),*
            );
        )*
    }
}

impl Value {
    value_impl_binary_operators! {
        +add(a, b) {
            Integer => a + b,
            String => {
                    let mut result = String::with_capacity(a.len() + b.len());
                    result += &a;
                    result += &b;
                    result
            }
        };
        -sub(a, b) { Integer => a - b };
        *mul(a, b) { Integer => a * b };
        /div(a, b) { Integer => a / b };
        <less(a, b) { Integer => a < b, String => a < b };
        <=less_or_equal(a, b) { Integer => a <= b, String => a <= b };
        >=greater_or_equal(a, b) { Integer => a >= b, String => a >= b };
        >greater(a, b) { Integer => a > b, String => a > b };
        ==equal(a, b) { Integer => a == b, String => a == b };
        !=not_equal(a, b) { Integer => a != b, String => a != b };
        &&and(a, b) { Bool => a && b };
        ||or(a, b) { Bool => a || b };
        ->implication(a, b) { Bool => if a { b } else { true } };
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        unsafe {
            self.increase_refcount();
        }
        Value(self.0)
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

mod builtins;
use builtins::{import, seq};

#[test]
fn string_packing_works() {
    assert!(matches!(
        UnpackedValue::new_string("hello".to_string()).pack().unpack(),
        UnpackedValue::String(ptr) if *ptr == "hello"
    ))
}

#[test]
fn scope_from_map_get_works() {
    unsafe {
        let s = Scope::from_map(
            {
                let mut m = ValueMap::new();
                m.insert("a".to_string(), UnpackedValue::Integer(10).pack());
                m
            },
            std::ptr::null_mut(),
        );
        assert_eq!(
            (*s).get("a").map(|x| x.0),
            Some(UnpackedValue::Integer(10).pack().0)
        );
    }
}

fn main() {
    match catch_nix_unwind(move || {
        let value = import(PathBuf::from(std::env::args().nth(1).unwrap()));
        seq(&value, true);
        value
    }) {
        Ok(value) => println!("{value:?}"),
        Err(exception) => {
            eprintln!("\x1b[31;1merror:\x1b[0m {}", exception.message);
            eprintln!("backtrace (most recent first):");
            for (i, span) in exception.stacktrace.into_iter().enumerate() {
                let source = &span.file.content;

                let span_line_start = source[..span.range.start().into()]
                    .rfind('\n')
                    .map(|x| x + 1)
                    .unwrap_or(0);
                let span_line_end = source[span.range.end().into()..]
                    .find('\n')
                    .map(|x| x + Into::<usize>::into(span.range.end()))
                    .unwrap_or(source.len());

                let nlines = source[span_line_start..span_line_end]
                    .chars()
                    .filter(|c| *c == '\n')
                    .count()
                    + 1;
                let lineno = source[..span.range.start().into()]
                    .chars()
                    .filter(|x| *x == '\n')
                    .count()
                    + 1;
                let colno = Into::<usize>::into(span.range.start()) - span_line_start + 1;
                let final_line_index = source[..span.range.end().into()]
                    .rfind('\n')
                    .map(|x| x + 1)
                    .unwrap_or(0);
                let end_colno = Into::<usize>::into(span.range.end()) - final_line_index;
                let end_offset = if final_line_index == span_line_start {
                    end_colno - colno + 1
                } else {
                    end_colno
                };

                eprintln!("\t{i}: {}:{}:{}", span.file.filename, lineno, colno);
                let mut is_highlighted = false;
                for (i, mut line) in source[span_line_start..span_line_end].lines().enumerate() {
                    eprint!("\t{:>6}|", i + lineno);
                    if i == 0 {
                        eprint!("{}\x1b[33m", &line[..colno - 1]);
                        line = &line[colno - 1..];
                        is_highlighted = true;
                    }

                    if is_highlighted {
                        eprint!("\x1b[33m");
                    }

                    if i == nlines - 1 {
                        eprint!("{}\x1b[0m", &line[..end_offset]);
                        line = &line[end_offset..];
                        is_highlighted = false;
                    }

                    eprintln!("{}\x1b[0m", line);
                }
                eprintln!()
            }
        }
    }
}
