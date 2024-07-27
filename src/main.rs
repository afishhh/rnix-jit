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
    collections::{BTreeMap, HashMap, HashSet},
    env::VarError,
    fmt::{Debug, Display, Write},
    mem::{offset_of, MaybeUninit},
    ops::Deref,
    path::{Path, PathBuf},
    rc::Rc,
};

use compiler::Executable;
use dwarf::*;
use regex::Regex;
use rnix::ast::{AstToken, Attr, Attrpath, Expr, HasEntry, InterpolPart};

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

fn build_attrpath(here: &Path, attrpath: Attrpath, program: &mut Program) -> usize {
    let mut n = 0;
    for attr in attrpath.attrs().collect::<Vec<_>>().into_iter().rev() {
        match attr {
            Attr::Ident(x) => program.operations.push(Operation::Push(
                UnpackedValue::new_string(x.ident_token().unwrap().green().text().to_string())
                    .pack(),
            )),
            Attr::Dynamic(x) => build_program(here, x.expr().unwrap(), program),
            Attr::Str(x) => build_string(here, x.normalized_parts(), |x| x, false, program),
        };
        n += 1;
    }
    n
}

fn build_string<Content>(
    here: &Path,
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
                    if !literal.starts_with("/") && !here.as_os_str().is_empty() {
                        format!("{}/{literal}", here.to_str().unwrap())
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
            build_program(here, interpol.expr().unwrap(), program);
        }
    }
    program.operations.push(Operation::StringCloneToMut);
    for part in parts {
        match part {
            InterpolPart::Literal(literal) => program.operations.push(Operation::Push(
                UnpackedValue::new_string(content_to_text(&literal).to_string()).pack(),
            )),
            InterpolPart::Interpolation(interpol) => {
                build_program(here, interpol.expr().unwrap(), program);
            }
        }
        program.operations.push(Operation::StringMutAppend);
    }
    if is_path {
        program.operations.push(Operation::StringToPath);
    }
}

fn build_program(here: &Path, expr: Expr, program: &mut Program) {
    match expr {
        Expr::Apply(x) => {
            build_program(here, x.argument().unwrap(), program);
            build_program(here, x.lambda().unwrap(), program);
            program.operations.push(Operation::Apply);
        }
        // TODO:
        Expr::Assert(x) => build_program(here, x.body().unwrap(), program),
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
                program.operations.push(Operation::GetAttrConsume {
                    default: x.default_expr().map(|expr| create_program(here, expr)),
                })
            }
        }
        Expr::Str(x) => build_string(here, x.normalized_parts(), |x| x, false, program),
        Expr::Path(x) => build_string(here, x.parts(), |x| x.syntax().text(), true, program),
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
                                x.default().map(|x| create_program(here, x)),
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
                .push(Operation::PushFunction(param, create_program(here, body)))
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
            program.operations.push(Operation::CreateAttrset {
                rec: x.rec_token().is_some(),
            });
            for inherit in x.inherits() {
                let source = inherit
                    .from()
                    .and_then(|f| f.expr())
                    .map(|source| create_program(here, source));
                program.operations.push(Operation::InheritAttrs(
                    source,
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
            let attrpath = x.attrpath().unwrap();
            let n = build_attrpath(here, attrpath, program);
            program.operations.push(Operation::HasAttrpath(n))
        }
        _ => todo!(),
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

fn dwarf_align(data: &mut Vec<u8>, start: usize) {
    let real_length = data.len() - start;
    let aligned_length = real_length.next_multiple_of(8);
    data[start..start + 4].copy_from_slice(&((aligned_length - 4) as u32).to_le_bytes());
    for _ in real_length..aligned_length {
        DW_CFA_nop(data);
    }
}

mod sysvunwind;
use sysvunwind::*;

const NIX_EXCEPTION_CLASS_SLICE: &[u8; 8] = b"FSH\0NIX\0";
const NIX_EXCEPTION_CLASS: u64 = u64::from_le_bytes(*NIX_EXCEPTION_CLASS_SLICE);

unsafe extern "C" fn nix_exception_cleanup(
    code: _Unwind_Reason_Code,
    _object: *const _Unwind_Exception,
) {
    println!("Nix exception deleted: {code:?}")
}

#[no_mangle]
unsafe extern "C" fn eh_personality(
    version: std::ffi::c_int,
    actions: std::ffi::c_int,
    _exception: *const _Unwind_Exception,
    _context: *const _Unwind_Context,
) -> _Unwind_Reason_Code {
    assert_eq!(version, 1);
    println!(
        "EH PERSONALITY CALLED SEARCH:{} CLEANUP:{} -> _URC_CONTINUE_UNWIND",
        actions & _UA_SEARCH_PHASE > 0,
        actions & _UA_CLEANUP_PHASE > 0
    );
    if actions & _UA_CLEANUP_PHASE > 0 {
        // TODO: Cleanup stack
        //       This requires saving more information in unwind data.
        //       (we need to know when the stack pointer was when the unwind happened)
    }
    _URC_CONTINUE_UNWIND
}

#[derive(Debug)]
struct Fde {
    initial_location: u64,
    address_range: u64,
}

fn create_eh_frame(
    fdes: impl IntoIterator<Item = Fde>,
    code_alignment_factor: u64,
    data_alignment_factor: i64,
    cie_instruction_builder: impl FnOnce(&mut Vec<u8>),
    fde_instruction_builder: impl Fn(usize, &Fde, &mut Vec<u8>),
) -> Vec<u8> {
    let mut data = vec![];

    // CIE
    {
        // length
        data.extend([0u8; 4]);
        // CIE_id
        data.extend(0u32.to_le_bytes());
        // version
        data.push(1);
        // augmentation
        data.extend(b"zP\0");
        // code_alignment_factor
        uleb128(&mut data, code_alignment_factor);
        // data_alignment_factor
        sleb128(&mut data, data_alignment_factor);
        // return_address_register
        data.push(16);
        // leb128(&mut data, /* "Return adddress RA" */ 16);
        // optional CIE augmentation section
        uleb128(&mut data, 9);
        data.push(0x00); // absolute pointer
        data.extend((eh_personality as u64).to_le_bytes());
        // initial_instructions
        cie_instruction_builder(&mut data);
    }

    dwarf_align(&mut data, 0);
    // dbg!(&data, data.len());

    // FDEs
    {
        for (i, fde) in fdes.into_iter().enumerate() {
            // eprintln!("fde {i} starts at address offset {}", data.len());
            let fde_start = data.len();
            // length
            data.extend([0u8; 4]);
            // CIE_pointer
            data.extend((data.len() as u32).to_le_bytes());
            // initial_location
            data.extend(fde.initial_location.to_le_bytes());
            // address_range
            data.extend(fde.address_range.to_le_bytes());
            // fde augmentation section
            uleb128(&mut data, 0);
            // call frame instructions
            let fde_program_start = data.len();
            fde_instruction_builder(i, &fde, &mut data);
            // eprintln!("program for fde {i} is {:?}", &data[fde_program_start..]);
            dwarf_align(&mut data, fde_start);
            // eprintln!("fde {:?}", &data[fde_start..])
        }
    }

    // libgcc detects the end of an .eh_frame section via the presence of a zero-length FDE
    // see https://github.com/llvm/llvm-project/blob/1055c5e1d316164c70e0c9f016411a28f3b4792e/llvm/lib/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.cpp#L128
    data.extend([0, 0, 0, 0]);

    data
}

// This will be useful when supporting non-libgcc unwinders
// (i.e. libunwind which doesn't accept whole .eh_frame sections but single FDEs)
#[allow(dead_code)]
fn walk_eh_frame_fdes(eh_frame: &[u8], callback: impl Fn(usize)) {
    let mut offset = 0;
    while offset < eh_frame.len() {
        let current = &eh_frame[offset..];
        let len = u32::from_le_bytes(current[0..4].try_into().unwrap()) as usize;
        if len == 0 {
            offset += 4;
            break;
        }
        let cie_pointer = u32::from_le_bytes(current[4..8].try_into().unwrap());
        if cie_pointer == 0 {
            // is a cie
        } else {
            callback(offset);
        }
        offset += len + 4;
    }
    assert_eq!(offset, eh_frame.len());
}

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
                unsafe { Rc::decrement_strong_count(executable) };
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

impl<T: Into<UnpackedValue>> From<T> for Value {
    fn from(value: T) -> Self {
        value.into().pack()
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
                    write!(f, "Lazy(")?;
                    x.unpack().fmt_display_rec(depth, f)?;
                    write!(f, ")")
                } else {
                    write!(f, "...")
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
}

macro_rules! value_impl_binary_operator {
    (
        $name: ident,
        $(
            $kind: ident($a: ident, $b: ident) => $expr: expr
        ),*
    ) => {
        extern "C" fn $name(self, other: Value) -> Value {
            match (self.unpack(), other.unpack()) {
                $((UnpackedValue::$kind($a), UnpackedValue::$kind($b)) => $expr.into(),)*
                (a, b) => panic!(
                    concat!(stringify!($operator), " is not supported between values of type {} and {}"),
                    a.kind(), b.kind()
                )
            }
        }
    };
}

macro_rules! value_impl_binary_operators {
    {
        $($name: ident($a: ident, $b: ident) {
            $($kind: ident => $expr: expr),*
        };)*
    } => {
        $(
            value_impl_binary_operator!(
                $name,
                $($kind($a, $b) => $expr),*
            );
        )*
    }
}

impl Value {
    value_impl_binary_operators! {
        add(a, b) {
            Integer => a + b,
            String => {
                    let mut result = String::with_capacity(a.len() + b.len());
                    result += &a;
                    result += &b;
                    result
            }
        };
        sub(a, b) { Integer => a - b };
        mul(a, b) { Integer => a * b };
        div(a, b) { Integer => a / b };
        less(a, b) { Integer => a < b, String => a < b };
        less_or_equal(a, b) { Integer => a <= b, String => a <= b };
        greater_or_equal(a, b) { Integer => a >= b, String => a >= b };
        greater(a, b) { Integer => a > b, String => a > b };
        equal(a, b) { Integer => a == b, String => a == b };
        not_equal(a, b) { Integer => a != b, String => a != b };
        and(a, b) { Bool => a && b };
        or(a, b) { Bool => a || b };
        implication(a, b) { Bool => if a { b } else { true } };
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

fn create_root_scope() -> *mut Scope {
    let mut values = ValueMap::new();

    macro_rules! extract_typed {
        ($builtin_name: literal, Attrset($($tt: tt)*)) => {
            unsafe { &*extract_typed!(@impl $builtin_name, Attrset($($tt)*)).get() }
        };
        ($builtin_name: literal, List($($tt: tt)*)) => {
            unsafe { &*extract_typed!(@impl $builtin_name, List($($tt)*)).get() }
        };
        ($builtin_name: literal, $what: ident($($tt: tt)*)) => {
            extract_typed!(@impl $builtin_name, $what($($tt)*))
        };
        (@impl $builtin_name: literal, $what: ident($($value: tt)*)) => {
            match extract_typed!(@evaluate_value $($value)*).unpack() {
                UnpackedValue::$what(ptr) => ptr,
                other => panic!(
                    concat!("builtin ", $builtin_name, " expected {} but got {}"),
                    ValueKind::$what,
                    other.kind()
                ),
            }
        };
        (@evaluate_value move $value: expr) => {
            $value.into_evaluated()
        };
        (@evaluate_value mut $value: expr) => {
            $value.evaluate_mut()
        };
        (@evaluate_value ref $value: expr) => {
            $value.evaluate()
        };
    }

    macro_rules! call_value {
        ($builtin_name: literal, $value: expr $(, $args: expr)+) => {{
            let mut current = $value;
            $(
                let fun = extract_typed!($builtin_name, Function(mut current));
                current = unsafe { (fun.call)(current, $args) };
            )+
            current
        }};
    }

    let map_function = UnpackedValue::new_function(|fun| {
        UnpackedValue::new_function(move |list| {
            let list = extract_typed!("map", List(move list));
            UnpackedValue::new_list(
                list.iter()
                    .map(|x| {
                        let fun = fun.clone();
                        x.clone()
                            .lazy_map(move |x| call_value!("map", fun.clone(), x))
                    })
                    .collect(),
            )
            .pack()
        })
        .pack()
    })
    .pack();

    let to_string_function = UnpackedValue::new_function(|value| match value.evaluate().unpack() {
        UnpackedValue::Integer(_) => todo!(),
        UnpackedValue::Bool(v) => if v { "1" } else { "0" }.to_string().into(),
        UnpackedValue::List(_) => todo!(),
        UnpackedValue::Attrset(_) => todo!(),
        UnpackedValue::String(_) => value,
        UnpackedValue::Function(_) => todo!(),
        UnpackedValue::Path(p) => p.display().to_string().into(),
        UnpackedValue::Lazy(_) => unreachable!(),
        UnpackedValue::Null => String::new().into(),
    })
    .pack();

    let throw_function = UnpackedValue::new_function(|message| {
        let message = extract_typed!("throw", String(move message));
        let exception = _Unwind_Exception::new(NIX_EXCEPTION_CLASS, nix_exception_cleanup);
        println!("builtins.throw {message}");
        let result = unsafe { _Unwind_RaiseException(&exception as *const _Unwind_Exception) };
        eprintln!("_Unwind_RaiseException failed: {result:?}");
        std::process::abort();
    })
    .pack();

    // TODO: Why is this also in the top-level scope? Backwards compatibility?
    values.insert("map".to_string(), map_function.clone());
    values.insert("toString".to_string(), to_string_function.clone());
    values.insert("throw".to_string(), throw_function.clone());

    values.insert(
        "builtins".to_string(),
        UnpackedValue::new_attrset({
            let mut builtins = ValueMap::new();

            builtins.insert("map".to_string(), map_function);
            builtins.insert("toString".to_string(), to_string_function);
            builtins.insert("throw".to_string(), throw_function);

            builtins.insert(
                "filter".to_string(),
                UnpackedValue::new_function(|filter| {
                    UnpackedValue::new_function(move |list| {
                        UnpackedValue::new_list(
                            extract_typed!("filter", List(move list))
                                .iter()
                                .filter(|x| {
                                    extract_typed!(
                                        "filter",
                                        Bool(move call_value!("filter", filter.clone(), (*x).clone()))
                                    )
                                })
                                .cloned()
                                .collect(),
                        )
                        .pack()
                    })
                    .pack()
                })
                .pack(),
            );

            macro_rules! insert_is_builtin {
                ($name: literal, $type: ident) => {
                    builtins.insert(
                        $name.to_string(),
                        UnpackedValue::new_function(|value| {
                            UnpackedValue::Bool(match value.into_evaluated().unpack() {
                                UnpackedValue::$type(..) => true,
                                UnpackedValue::Lazy(_) => unreachable!(),
                                _ => false,
                            })
                            .pack()
                        })
                        .pack(),
                    );
                };
            }

            insert_is_builtin!("isPath", Path);
            insert_is_builtin!("isString", String);
            insert_is_builtin!("isInt", Integer);
            builtins.insert(
                "isFloat".to_string(),
                UnpackedValue::new_function(|_v| UnpackedValue::Bool(false).pack()).pack(),
            );
            insert_is_builtin!("isBool", Bool);
            insert_is_builtin!("isAttrs", Attrset);
            insert_is_builtin!("isList", List);

            builtins.insert(
                "seq".to_string(),
                UnpackedValue::new_function(|seqd| {
                    seq(&seqd, false);
                    UnpackedValue::new_function(|value| value).pack()
                })
                .pack(),
            );

            builtins.insert(
                "deepSeq".to_string(),
                UnpackedValue::new_function(|seqd| {
                    seq(&seqd, true);
                    UnpackedValue::new_function(|value| value).pack()
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

            builtins.insert(
                "elem".to_string(),
                UnpackedValue::new_function(|needle| {
                    UnpackedValue::new_function(move |list| {
                        let list = extract_typed!("elem", List(move list));
                        for value in list.iter() {
                            // TODO: less cloning
                            if value.evaluate().clone().equal(needle.clone()).0 == Value::TRUE.0 {
                                return Value::TRUE;
                            }
                        }
                        Value::FALSE
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "head".to_string(),
                UnpackedValue::new_function(|list| extract_typed!("elemAt", List(move list))[0].clone())
                    .pack(),
            );

            builtins.insert(
                "tail".to_string(),
                UnpackedValue::new_function(|list| extract_typed!("elemAt", List(move list)).last().unwrap().clone())
                    .pack(),
            );

            builtins.insert(
                "elemAt".to_string(),
                UnpackedValue::new_function(|xs| {
                    UnpackedValue::new_function(move |n| {
                        let xs = extract_typed!("elemAt", List(ref xs));
                        let n = extract_typed!("elemAt", Integer(move n));
                        xs.get(n as usize).unwrap().clone()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "all".to_string(),
                UnpackedValue::new_function(|predicate| {
                    UnpackedValue::new_function(move |list| {
                        let list = extract_typed!("all", List(move list));
                        #[allow(clippy::redundant_closure)]
                        // NOTE: It doesn't seem like Value::and implements FnMut?
                        list.iter()
                            .map(|element| {
                                call_value!("all", predicate.clone(), element.clone())
                            })
                            .reduce(|a, b| Value::and(a, b))
                            .unwrap_or(Value::TRUE)
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "any".to_string(),
                UnpackedValue::new_function(|predicate| {
                    UnpackedValue::new_function(move |list| {
                        let list = extract_typed!("any", List(move list));
                        #[allow(clippy::redundant_closure)]
                        for value in list.iter() {
                            let UnpackedValue::Bool(value) =
                                call_value!("any", predicate.clone(), value.clone()).into_unpacked()
                            else {
                                panic!("any predicate returned non-boolean")
                            };
                            if value {
                                return Value::TRUE;
                            }
                        }
                        Value::FALSE
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "mapAttrs".to_string(),
                UnpackedValue::new_function(|mapper| {
                    UnpackedValue::new_function(move |attrs| {
                        let attrs = extract_typed!("mapAttrs", Attrset(move attrs));
                        let mut result = ValueMap::new();
                        for (key, value) in attrs {
                            result.insert(
                                key.to_string(),
                                call_value!(
                                    "mapAttrs",
                                    mapper.clone(),
                                    UnpackedValue::new_string(key.clone()).pack(),
                                    value.clone()
                                ),
                            );
                        }
                        UnpackedValue::new_attrset(result).pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "listToAttrs".to_string(),
                UnpackedValue::new_function(|list| {
                    let list = extract_typed!("listToAttrs", List(move list));
                    UnpackedValue::new_attrset(
                        list.iter()
                            .map(|item| {
                                let attrset = extract_typed!("listToAttrs", Attrset(ref item));
                                let name = extract_typed!(
                                    "listToAttrs",
                                    String(ref attrset.get("name").unwrap())
                                );
                                let value = attrset.get("value").unwrap();
                                (Rc::unwrap_or_clone(name), value.clone())
                            })
                            .collect(),
                    )
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "getAttr".to_string(),
                UnpackedValue::new_function(|name| {
                    let name = extract_typed!("getAttr", String(move  name));
                    UnpackedValue::new_function(move |attrset| {
                        let attrset = extract_typed!("getAttr", Attrset(move attrset));
                        attrset.get(name.as_str()).unwrap().clone()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "hasAttr".to_string(),
                UnpackedValue::new_function(|name| {
                    let name = extract_typed!("hasAttr", String(move name));
                    UnpackedValue::new_function(move |attrset| {
                        let attrset = extract_typed!("hasAttr", Attrset(move attrset));
                        UnpackedValue::Bool(attrset.contains_key(name.as_str())).pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "intersectAttrs".to_string(),
                UnpackedValue::new_function(|e1| {
                    let e1 = extract_typed!("intersectAttrs", Attrset(move e1));
                    UnpackedValue::new_function(move |e2| {
                        let e2 = extract_typed!("intersectAttrs", Attrset(move e2));
                        UnpackedValue::new_attrset(if e2.len() > e1.len() {
                            e1.keys()
                                .filter_map(|key| {
                                    e1.get(key).map(|value| (key.clone(), value.clone()))
                                })
                                .collect()
                        } else {
                            e2.iter()
                                .filter(|&(key, _)| e1.contains_key(key))
                                .map(|(key, value)| (key.clone(), value.clone()))
                                .collect()
                        })
                        .pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "removeAttrs".to_string(),
                UnpackedValue::new_function(|attrs| {
                    let attrs = extract_typed!("removeAttrs", Attrset(move attrs));
                    UnpackedValue::new_function(move |names| {
                        let names = extract_typed!("removeAttrs", List(move names))
                            .iter()
                            .map(|value| extract_typed!("removeAttrs", String(ref value)))
                            .collect::<HashSet<_>>();
                        UnpackedValue::new_attrset(
                            attrs
                                .iter()
                                .filter(|(key, _)| names.contains(*key))
                                .map(|(key, value)| (key.clone(), value.clone()))
                                .collect(),
                        )
                        .pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "zipAttrsWith".to_string(),
                UnpackedValue::new_function(|mapper| {
                    UnpackedValue::new_function(move |sets| {
                        UnpackedValue::new_attrset(
                            extract_typed!("zipAttrsWith", List(move sets))
                                .iter()
                                .fold(HashMap::<String, Vec<Value>>::new(), |mut acc, value| {
                                    for (key, value) in
                                        extract_typed!("zipAttrsWith", Attrset(ref value))
                                    {
                                        acc.entry(key.to_string())
                                            .or_default()
                                            .push(value.clone())
                                    }
                                    acc
                                })
                                .into_iter()
                                .map(|(key, values)| {
                                    (
                                        key.clone(),
                                        call_value!(
                                            "zipAttrsWith",
                                            mapper.clone(),
                                            UnpackedValue::new_string(key).pack(),
                                            UnpackedValue::new_list(values).pack()
                                        ),
                                    )
                                })
                                .collect(),
                        )
                        .pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "getEnv".to_string(),
                UnpackedValue::new_function(|name| {
                    let name = extract_typed!("getEnv", String(move name));
                    UnpackedValue::new_string(match std::env::var(name.as_str()) {
                        Ok(value) => value,
                        Err(VarError::NotPresent) => String::new(),
                        Err(VarError::NotUnicode(_)) => panic!("getEnv var is not unicode"),
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "attrNames".to_string(),
                UnpackedValue::new_function(|attrset| {
                    let attrset = extract_typed!("attrNames", Attrset(move attrset));

                    UnpackedValue::new_list(
                        attrset
                            .keys()
                            .cloned()
                            .map(UnpackedValue::new_string)
                            .map(UnpackedValue::pack)
                            .collect(),
                    )
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "attrValues".to_string(),
                UnpackedValue::new_function(|attrset| {
                    let attrset = extract_typed!("attrValues", Attrset(move attrset));

                    UnpackedValue::new_list(attrset.values().cloned().collect()).pack()
                })
                .pack(),
            );

            builtins.insert(
                "catAttrs".to_string(),
                UnpackedValue::new_function(|name| {
                    let name = extract_typed!("catAttrs", String(move name));

                    UnpackedValue::new_function(move |attrlist| {
                        let mut result = vec![];
                        let list = extract_typed!("catAttrs", List(move attrlist));
                        for value in list.iter() {
                            let attrset = extract_typed!("catAttrs", Attrset(ref value));
                            if let Some(value) = attrset.get(name.as_str()) {
                                result.push(value.clone());
                            }
                        }
                        UnpackedValue::new_list(result).pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "groupBy".to_string(),
                UnpackedValue::new_function(|f| {
                    UnpackedValue::new_function(move |list| {
                        let mut result = BTreeMap::new();
                        let list = extract_typed!("groupBy", List(move list));
                        for value in list.iter() {
                            let key = extract_typed!(
                                "groupBy",
                                String(move call_value!("groupBy", f.clone(), value.clone()))
                            );
                            result
                                .entry(Rc::unwrap_or_clone(key))
                                .or_insert_with(Vec::new)
                                .push(value.clone());
                        }
                        UnpackedValue::new_attrset(
                            result
                                .into_iter()
                                .map(|(k, v)| (k, UnpackedValue::new_list(v).pack()))
                                .collect(),
                        )
                        .pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "genericClosure".to_string(),
                UnpackedValue::new_function(|_args| todo!("Implement builtins.genericClosure"))
                    .pack(),
            );

            // TODO: merge this with Value::operator methods
            macro_rules! binary_operator {
                ($name: literal $(, int => $op_int: tt)? $(, bool => $op_bool: tt)?) => {
                    builtins.insert(
                        $name.to_string(),
                        UnpackedValue::new_function(|a| {
                            UnpackedValue::new_function(move |b| match (a.unpack(), b.into_unpacked()) {
                                $(
                                    (UnpackedValue::Integer(a), UnpackedValue::Integer(b)) => {
                                        (a $op_int b).into()
                                    }
                                )?
                                $(
                                    (UnpackedValue::Bool(a), UnpackedValue::Bool(b)) => {
                                        (a $op_bool b).into()
                                    }
                                )?
                                (a, b) => panic!(concat!("builtins.", $name, " not supported between values of type {} and {}"), a.kind(), b.kind())
                            }).pack()
                        })
                        .pack(),
                    );
                }
            }

            binary_operator!("add", int => +);
            binary_operator!("sub", int => -);
            binary_operator!("mul", int => *);
            binary_operator!("div", int => /);
            binary_operator!("lessThan", int => <);
            binary_operator!("bitAnd", int => &);
            binary_operator!("bitOr", int => |);
            binary_operator!("bitXor", int => ^);

            builtins.insert(
                "length".to_string(),
                UnpackedValue::new_function(|list| {
                    let list = extract_typed!("length", List(move list));
                    UnpackedValue::Integer(list.len() as i32).pack()
                })
                .pack(),
            );

            builtins.insert(
                "foldl'".to_string(),
                UnpackedValue::new_function(|op| {
                    UnpackedValue::new_function(move |nul| {
                        let op = op.clone();
                        UnpackedValue::new_function(move |list| {
                            let mut acc = nul.clone();
                            let list = extract_typed!("foldl'", List(move list));
                            for value in list.iter() {
                                acc = call_value!("foldl'", op.clone(), acc, value.clone());
                            }
                            acc
                        }).pack()
                    }).pack()
                }).pack(),
            );

            builtins.insert(
                "partition".to_string(),
                UnpackedValue::new_function(|pred| {
                    UnpackedValue::new_function(move |list| {
                        let list = extract_typed!("partition", List(move list));
                        let (right, wrong) = list.iter().cloned().partition(|value|
                            extract_typed!("partition", Bool(move call_value!("partition", pred.clone(), value.clone()))
                            ));
                        UnpackedValue::new_attrset({
                            let mut result = ValueMap::new();
                            result.insert("right".to_string(), UnpackedValue::new_list(right).pack());
                            result.insert("wrong".to_string(), UnpackedValue::new_list(wrong).pack());
                            result
                        }).pack()
                    }).pack()
                }).pack(),
            );

            builtins.insert(
                "sort".to_string(),
                UnpackedValue::new_function(|_comparator| {
                    UnpackedValue::new_function(move |_list| {
                        // let list = extract_typed!("sort", List(list));
                        // let result: Vec<_> = list.iter().map(Value::unpack).collect();
                        todo!("builtins.sort");
                        // UnpackedValue::new_list(result.into_iter().map(UnpackedValue::pack).collect()).pack()
                    }).pack()
                }).pack()
            );


            builtins.insert(
                "concatLists".to_string(),
                UnpackedValue::new_function(|lists| {
                    let lists = extract_typed!("concatLists", List(move lists));
                    UnpackedValue::new_list(
                        lists
                            .iter()
                            .flat_map(|list| extract_typed!("concatLists", List(ref list)).iter())
                            .cloned()
                            .collect(),
                    )
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "concatMap".to_string(),
                UnpackedValue::new_function(|mapper| {
                    UnpackedValue::new_function(move |lists| {
                        let lists = extract_typed!("concatMap", List(move lists));
                        UnpackedValue::new_list(
                            lists
                                .iter()
                                .flat_map(|list| extract_typed!("concatLists", List(ref list)).iter())
                                .cloned()
                                .map(|value| call_value!("concatMap", mapper.clone(), value))
                                .collect(),
                        )
                        .pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "concatStringsSep".to_string(),
                UnpackedValue::new_function(|sep| {
                    let sep = extract_typed!("concatStringsSep", String(move sep));
                    UnpackedValue::new_function(move |strings| {
                        let strings = extract_typed!("concatStringsSep", List(move strings));
                        UnpackedValue::new_string({
                            let mut result = String::new();
                            let mut it = strings.iter();
                            if let Some(first) = it.next() {
                                result +=
                                    extract_typed!("concatStringsSep", String(ref first)).as_str();
                            }
                            for string in it {
                                result += sep.as_str();
                                result +=
                                    extract_typed!("concatStringsSep", String(ref string)).as_str();
                            }
                            result
                        })
                        .pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "stringLength".to_string(),
                UnpackedValue::new_function(|string| {
                    let string = extract_typed!("stringLength", String(move string));
                    UnpackedValue::Integer(string.len() as i32).pack()
                })
                .pack(),
            );

            builtins.insert(
                "substring".to_string(),
                UnpackedValue::new_function(|start| {
                    UnpackedValue::new_function(move |len| {
                        let start = start.clone();
                        UnpackedValue::new_function(move |string| {
                            let start = extract_typed!("substring", Integer(ref start)) as usize;
                            let len = extract_typed!("substring", Integer(ref len));
                            let string = extract_typed!("substring", String(move string));

                            UnpackedValue::new_string(if start >= string.len() {
                                String::new()
                            } else if len == -1 {
                                string[start..].to_string()
                            } else {
                                string[start..std::cmp::min(start + len as usize, string.len())].to_string()
                            })
                            .pack()
                        })
                        .pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "replaceStrings".to_string(),
                UnpackedValue::new_function(|from| {
                    UnpackedValue::new_function(move |to| {
                        let from = from.clone();
                        UnpackedValue::new_function(move |s| {
                            let from = extract_typed!("replaceStrings", List(ref from));
                            let to = extract_typed!("replaceStrings", List(ref to));
                            let s = extract_typed!("replaceStrings", String(move s));
                            let mut result = Rc::unwrap_or_clone(s);

                            for (i, pattern) in from.iter().enumerate() {
                                let pattern = extract_typed!("replaceStrings", String(ref pattern));
                                // FIXME: make this actually efficient
                                if result.contains(pattern.as_str()) {
                                    result = result.replace(
                                        pattern.as_str(),
                                        extract_typed!("replaceStrings", String(ref to[i])).as_str(),
                                    );
                                }
                            }

                            UnpackedValue::new_string(result).pack()
                        })
                        .pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "match".to_string(),
                UnpackedValue::new_function(|regex| {
                    let regex = Regex::new(extract_typed!("match", String(move regex)).as_str()).unwrap();
                    UnpackedValue::new_function(move |string| {
                        let string = extract_typed!("match", String(move string));
                        regex.is_match(string.as_str()).into()
                    }).pack()
                }).pack()
            );

            builtins.insert(
                "split".to_string(),
                UnpackedValue::new_function(|regex| {
                    let regex = Regex::new(extract_typed!("split", String(move regex)).as_str()).unwrap();
                    UnpackedValue::new_function(move |string| {
                        let string = extract_typed!("split", String(move string));
                        UnpackedValue::new_list(regex.split(&string).map(|part| UnpackedValue::new_string(part.to_string()).pack()).collect()).pack()
                    }).pack()
                }).pack()
            );

            builtins.insert(
                "pathExists".to_string(),
                UnpackedValue::new_function(|path| {
                    let path = extract_typed!("pathExists", Path(move path));
                    UnpackedValue::Bool(
                        path.try_exists().unwrap()
                    )
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "readFile".to_string(),
                UnpackedValue::new_function(|path| {
                    let path = extract_typed!("readFile", Path(move path));
                    UnpackedValue::new_string(
                        std::fs::read_to_string(&*path).unwrap()
                    )
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "readFileType".to_string(),
                UnpackedValue::new_function(|path| {
                    let path = extract_typed!("readFileType", Path(move path));
                    let file_type = path.metadata().unwrap().file_type();
                    UnpackedValue::new_string(
                        if file_type.is_dir() {
                            "directory"
                        } else if file_type.is_file() {
                            "regular"
                        } else if file_type.is_symlink() {
                            "symlink"
                        } else {
                            "unknown"
                        }
                        .to_string(),
                    )
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "genList".to_string(),
                UnpackedValue::new_function(|generator| {
                    UnpackedValue::new_function(move |length| {
                        let mut result = vec![];
                        for i in 0..extract_typed!("genList", Integer(move length)) {
                            result.push(call_value!(
                                "genList",
                                generator.clone(),
                                UnpackedValue::Integer(i).pack()
                            ))
                        }
                        UnpackedValue::new_list(result).pack()
                    })
                    .pack()
                })
                .pack(),
            );

            builtins.insert(
                "currentSystem".to_string(),
                UnpackedValue::new_string("x86_64-linux".to_string()).pack(),
            );

            builtins.insert(
                "hasContext".to_string(),
                UnpackedValue::new_function(|_string| {
                    Value::FALSE
                })
                .pack(),
            );

            builtins.insert(
                "addErrorContext".to_string(),
                UnpackedValue::new_function(|_context| {
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
    values.insert(
        "import".to_string(),
        UnpackedValue::new_function(|value| {
            let path = extract_typed!("import", Path(move value));
            import(&path)
        })
        .pack(),
    );

    values.insert(
        "__trace_shallow".to_string(),
        UnpackedValue::new_function(|value: Value| {
            println!("trace (shallow): {value:?}");
            UnpackedValue::new_function(|value| value).pack()
        })
        .pack(),
    );

    Scope::from_map(values, std::ptr::null_mut())
}

thread_local! {
    static ROOT_SCOPE: *mut Scope = create_root_scope();
}

fn import(path: &Path) -> Value {
    let (here, expr) = match std::fs::read_to_string(path) {
        Err(e) if e.to_string().contains("Is a directory") => {
            std::fs::read_to_string(path.join("default.nix")).map(|s| (path, s))
        }
        other => other.map(|s| (path.parent().unwrap(), s)),
    }
    .unwrap();

    let result = rnix::Root::parse(&expr);
    let root = result.tree();
    let mut root_block = Program { operations: vec![] };
    build_program(here, root.expr().unwrap(), &mut root_block);
    let compiled = compiler::compile(root_block, None).unwrap();
    ROOT_SCOPE.with(|root| compiled.run(*root, &Value::NULL))
}

fn seq(value: &Value, deep: bool) {
    match value.unpack() {
        UnpackedValue::Integer(_) => (),
        UnpackedValue::Bool(_) => (),
        UnpackedValue::List(value) => {
            let list = unsafe { &mut *value.get() };
            for value in list.iter() {
                let value = value.evaluate();
                if deep {
                    seq(value, deep);
                }
            }
        }
        UnpackedValue::Attrset(value) => {
            let map = unsafe { &mut *value.get() };
            for (_, value) in map.iter_mut() {
                let value = value.evaluate();
                if deep {
                    seq(value, deep);
                }
            }
        }
        UnpackedValue::String(_) => (),
        UnpackedValue::Path(_) => (),
        UnpackedValue::Function(_) => (),
        UnpackedValue::Lazy(value) => seq(value.evaluate(), deep),
        UnpackedValue::Null => (),
    }
}

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

fn dwarf_testing() {
    let function = compiler::compile(
        Program {
            operations: vec![
                Operation::Push(Value::NULL),
                Operation::Push(
                    UnpackedValue::new_function(|_| {
                        let exception =
                            _Unwind_Exception::new(NIX_EXCEPTION_CLASS, nix_exception_cleanup);
                        let result = unsafe {
                            _Unwind_RaiseException(&exception as *const _Unwind_Exception)
                        };
                        eprintln!("_Unwind_RaiseException: {result:?}");
                        std::process::abort()
                    })
                    .pack(),
                ),
                Operation::Apply,
            ],
        },
        None,
    )
    .unwrap();
    let executable = compiler::compile(
        Program {
            operations: vec![
                Operation::Push(Value::NULL),
                Operation::Push(UnpackedValue::Function(Rc::new(Function {
                    call: function.code(),
                    _executable: None,
                    builtin_closure: None,
                    parent_scope: std::ptr::null_mut(),
                })).pack()),
                Operation::Apply,
            ],
        },
        None,
    )
    .unwrap();
    // std::panic::catch_unwind(|| {
    let result = executable.run(std::ptr::null_mut(), &Value::NULL);
    println!("{result:?}");
    std::process::abort();
    // })
    // .unwrap_or(())
}

fn main() {
    dwarf_testing();
    let value = import(&PathBuf::from(std::env::args().nth(1).unwrap()));
    seq(&value, true);
    println!("{}", value);
}
