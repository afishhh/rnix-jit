use std::{
    cell::UnsafeCell,
    collections::BTreeMap,
    fmt::{Debug, Display, Write as FmtWrite},
    mem::MaybeUninit,
    path::PathBuf,
    rc::Rc,
};

use crate::{compiler::Executable, throw, Scope};

pub(crate) const VALUE_TAG_WIDTH: u8 = 3;
pub(crate) const VALUE_TAG_MASK: u64 = 0b111;
pub(crate) const ATTRSET_TAG_MASK: u64 = 0b111 | (0b11 << 62);
pub(crate) const ATTRSET_TAG_ATTRSET: u64 = ValueKind::Attrset as u64;
pub(crate) const ATTRSET_TAG_LAZY: u64 = ValueKind::Attrset as u64 | (0b11 << 62);
pub(crate) const ATTRSET_TAG_NULL: u64 = ValueKind::Attrset as u64 | (0b10 << 62);

#[repr(u8)]
#[derive(Debug, PartialEq, Eq)]
pub enum ValueKind {
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

pub(crate) struct Function {
    pub(crate) call: unsafe extern "C-unwind" fn(Value, Value) -> Value,
    pub(crate) _executable: Option<Rc<Executable>>,
    pub(crate) builtin_closure: Option<Box<dyn FnMut(Value) -> Value>>,
    pub(crate) parent_scope: *const Scope,
}

#[derive(Debug)]
enum LazyValueImpl {
    Evaluated(Value),
    LazyJIT((*mut Scope, Rc<Executable>)),
    LazyBuiltin(MaybeUninit<Box<dyn FnOnce() -> Value>>),
}

#[derive(Debug, Clone)]
pub struct LazyValue(Rc<UnsafeCell<LazyValueImpl>>);

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
    pub(crate) fn from_jit(scope: *mut Scope, rc: Rc<Executable>) -> LazyValue {
        Self(Rc::new(UnsafeCell::new(LazyValueImpl::LazyJIT((
            scope, rc,
        )))))
    }

    pub fn from_closure(fun: impl FnOnce() -> Value + 'static) -> LazyValue {
        Self(Rc::new(UnsafeCell::new(LazyValueImpl::LazyBuiltin(
            MaybeUninit::new(Box::new(fun)),
        ))))
    }

    pub fn evaluate(&self) -> &Value {
        unsafe { &mut *self.0.get() }.evaluate()
    }

    #[inline]
    pub fn as_maybe_evaluated(&self) -> Option<&Value> {
        unsafe { (*self.0.get()).as_maybe_evaluated() }
    }
}

pub(crate) type ValueMap = BTreeMap<String, Value>;
pub(crate) type ValueList = Vec<Value>;

#[derive(Clone)]
pub enum UnpackedValue {
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
    pub fn new_string(value: String) -> UnpackedValue {
        UnpackedValue::String(Rc::new(value))
    }

    pub fn new_path(value: PathBuf) -> UnpackedValue {
        UnpackedValue::Path(Rc::new(value))
    }

    pub fn new_list(value: ValueList) -> UnpackedValue {
        UnpackedValue::List(Rc::new(UnsafeCell::new(value)))
    }

    pub fn new_attrset(value: ValueMap) -> UnpackedValue {
        UnpackedValue::Attrset(Rc::new(UnsafeCell::new(value)))
    }

    pub fn new_function(function: impl FnMut(Value) -> Value + 'static) -> UnpackedValue {
        UnpackedValue::Function(Rc::new(Function {
            call: call_builtin,
            _executable: None,
            builtin_closure: Some(Box::new(function)),
            parent_scope: std::ptr::null_mut(),
        }))
    }

    pub fn kind(&self) -> ValueKind {
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

    pub fn pack(self) -> Value {
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

    pub(crate) fn into_evaluated(self) -> UnpackedValue {
        if let UnpackedValue::Lazy(lazy) = self {
            lazy.evaluate().unpack()
        } else {
            self
        }
    }

    pub(crate) fn fmt_attrset_display(
        map: &ValueMap,
        depth: usize,
        f: &mut impl FmtWrite,
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

    pub(crate) fn fmt_list_display(
        list: &[Value],
        depth: usize,
        f: &mut impl FmtWrite,
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

    pub(crate) fn fmt_display_rec(&self, depth: usize, f: &mut impl FmtWrite) -> std::fmt::Result {
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

    pub(crate) fn fmt_debug_rec(&self, depth: usize, f: &mut impl FmtWrite) -> std::fmt::Result {
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
pub struct Value(pub(crate) u64);

impl Value {
    pub const NULL: Value = Value(ATTRSET_TAG_NULL);
    pub const TRUE: Value = Value(0b1000 | ValueKind::Bool as u64);
    pub const FALSE: Value = Value(ValueKind::Bool as u64);

    pub(crate) const fn from_bool(value: bool) -> Value {
        if value {
            return Value::TRUE;
        } else {
            return Value::FALSE;
        }
    }

    pub fn kind(&self) -> ValueKind {
        match self.0 & ATTRSET_TAG_MASK {
            ATTRSET_TAG_NULL => return ValueKind::Null,
            ATTRSET_TAG_LAZY => return ValueKind::Lazy,
            _ => (),
        }
        let value = self.0 & VALUE_TAG_MASK;
        assert!(value <= ValueKind::Bool as u64);
        unsafe { std::mem::transmute(value as u8) }
    }

    pub fn unpack(&self) -> UnpackedValue {
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

    pub(crate) fn into_unpacked(self) -> UnpackedValue {
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

    pub(crate) fn evaluate(&self) -> &Value {
        if (self.0 & ATTRSET_TAG_MASK) == ATTRSET_TAG_LAZY {
            unsafe { (*((self.0 & !ATTRSET_TAG_MASK) as *mut LazyValueImpl)).evaluate() }
        } else {
            self
        }
    }

    pub(crate) fn evaluate_mut(&mut self) -> &Value {
        if (self.0 & ATTRSET_TAG_MASK) == ATTRSET_TAG_LAZY {
            unsafe {
                LazyValue(Rc::from_raw((self.0 & !ATTRSET_TAG_MASK) as *mut _))
                    .evaluate()
                    .clone_into(self)
            }
        }
        self
    }

    pub(crate) fn into_evaluated(self) -> Value {
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

    pub(crate) fn lazy_map(self, fun: impl FnOnce(Value) -> Value + 'static) -> Value {
        UnpackedValue::Lazy(LazyValue::from_closure(|| fun(self))).pack()
    }

    #[inline]
    pub(crate) unsafe fn leaking_copy(&self) -> Value {
        Value(self.0)
    }

    pub fn is_true(&self) -> bool {
        self.0 == Self::TRUE.0
    }

    pub fn is_false(&self) -> bool {
        self.0 == Self::FALSE.0
    }

    pub fn equal(self, other: Value) -> Value {
        (self == other).into()
    }

    pub fn not_equal(self, other: Value) -> Value {
        (self != other).into()
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self.unpack(), other.unpack()) {
            (
                UnpackedValue::Integer(_) | UnpackedValue::Bool(_) | UnpackedValue::Null,
                UnpackedValue::Integer(_) | UnpackedValue::Bool(_) | UnpackedValue::Null,
            ) => self.0 == other.0,
            (UnpackedValue::String(a), UnpackedValue::String(b)) => a == b,
            (a, b) if a.kind() != b.kind() => false,
            (a, b) => throw!(
                "== is not supported between values of type {} and {}",
                a.kind(),
                b.kind()
            ),
        }
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
        pub(crate) extern "C-unwind" fn $name(self, other: Value) -> Value {
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
