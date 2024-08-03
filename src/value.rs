use std::{
    cell::UnsafeCell,
    collections::BTreeMap,
    fmt::{Debug, Display, Write as FmtWrite},
    mem::MaybeUninit,
    path::PathBuf,
    rc::Rc,
};

use crate::{compiler::Executable, runnable::Runnable, throw, Scope};

pub struct Function {
    pub(crate) runnable: *const Runnable,
    pub(crate) parent_scope: *mut Scope,
}

impl Function {
    pub(crate) fn new(runnable: Rc<Runnable>, parent_scope: *mut Scope) -> Self {
        Self {
            runnable: Rc::into_raw(runnable),
            parent_scope,
        }
    }

    #[inline(always)]
    pub fn run(&self, arg: Value) -> Value {
        unsafe { (*self.runnable).run(self.parent_scope, arg) }
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        unsafe { Rc::from_raw(self.runnable) };
    }
}

#[derive(Debug)]
pub(crate) enum LazyValueImpl {
    Evaluated(Value),
    LazyRunnable((*mut Scope, Rc<Runnable>)),
    LazyBuiltin(MaybeUninit<Box<dyn FnOnce() -> Value>>),
}

#[derive(Debug, Clone)]
pub struct LazyValue(pub(crate) Rc<UnsafeCell<LazyValueImpl>>);

thread_local! {
    static LAZY_DEPTH: UnsafeCell<usize> = const { UnsafeCell::new(0) };
}

impl LazyValueImpl {
    fn evaluate(&mut self) -> &Value {
        match self {
            LazyValueImpl::Evaluated(value) => value,
            LazyValueImpl::LazyRunnable((scope, runnable)) => {
                LAZY_DEPTH.with(|x| unsafe {
                    let depth = &mut *x.get();
                    *depth += 1;
                    if *depth > 100 {
                        panic!("max lazy JIT depth exceeded");
                    };
                });
                let result = unsafe { runnable.run(*scope, Value::NULL) }.into_evaluated();
                *self = LazyValueImpl::Evaluated(result);
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
    pub(crate) fn from_jit(scope: *mut Scope, rc: Rc<Runnable<Executable>>) -> LazyValue {
        Self(Rc::new(UnsafeCell::new(LazyValueImpl::LazyRunnable((
            scope,
            unsafe { Rc::from_raw(Runnable::erase(Rc::into_raw(rc))) },
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

pub type ValueMap = BTreeMap<String, Value>;
pub type ValueList = Vec<Value>;

#[derive(Clone)]
pub enum UnpackedValue {
    Integer(i32),
    Double(f64),
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

impl From<f64> for UnpackedValue {
    fn from(value: f64) -> Self {
        UnpackedValue::Double(value)
    }
}

impl From<bool> for UnpackedValue {
    fn from(value: bool) -> Self {
        UnpackedValue::Bool(value)
    }
}

impl<'a> From<&'a str> for UnpackedValue {
    fn from(value: &'a str) -> Self {
        UnpackedValue::new_string(value.to_string())
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
        let rc = Rc::new(Runnable::from_closure(function));
        let rc = unsafe { Rc::from_raw(Runnable::erase(Rc::into_raw(rc))) };
        UnpackedValue::Function(Rc::new(Function::new(rc, std::ptr::null_mut())))
    }

    pub fn kind(&self) -> ValueKind {
        match self {
            UnpackedValue::Integer(_) => ValueKind::Integer,
            UnpackedValue::Double(_) => ValueKind::Double,
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
            ($ptr: expr, $kind: ident) => {{
                unsafe {
                    Value::from_raw_parts(
                        ValueKind::$kind,
                        (Rc::into_raw($ptr) as u64) & 0xFFFF_FFFF_FFFF,
                    )
                }
            }};
        }
        match self {
            Self::Integer(x) => unsafe {
                Value::from_raw_parts(ValueKind::Integer, std::mem::transmute::<_, u32>(x).into())
            },
            Self::Double(x) => {
                let bits = x.to_bits();
                debug_assert!(bits & NAN_BITS_MASK != SIGNALLING_NAN_BITS);
                Value(bits)
            }
            Self::Bool(x) => unsafe { Value::from_raw_parts(ValueKind::Bool, x as u64) },
            Self::String(ptr) => pack_ptr!(ptr, String),
            Self::Path(ptr) => pack_ptr!(ptr, Path),
            Self::List(ptr) => pack_ptr!(ptr, List),
            Self::Attrset(ptr) => pack_ptr!(ptr, Attrset),
            Self::Function(ptr) => pack_ptr!(ptr, Function),
            Self::Lazy(ptr) => pack_ptr!(ptr.0, Lazy),
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
            Self::Double(value) => {
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
            Self::Double(value) => {
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

// 1111111111110000000000000000000000000000000000000000000000000000
// |--------------| x86_64 canonical address unused space (16 bits)
// |----------| IEEE 764 double precision floating point required NaN bits (12 bits)
// The four unused bits is where we will store our tag.
// The tag 0b0000 is reserved for double since that is the preferred NaN representation.

// Heap (pointer) types:
// Attrset
// Function
// String
// Path
// List
// Lazy
//
// Pointers have to be sign extended when extracted from the value. (see x86_64 canonical address form)
//
// Value types:
// Double
// Integer
// Bool
// Null

#[cfg(target_arch = "x86_64")]
pub(crate) const QUIET_NAN_BITS: u64 =
    0b0111111111110000000000000000000000000000000000000000000000000000;
pub(crate) const SIGNALLING_NAN_BITS: u64 =
    0b0111111111100000000000000000000000000000000000000000000000000000;
pub(crate) const NAN_BITS_MASK: u64 = QUIET_NAN_BITS;

#[cfg(not(target_arch = "x86_64"))]
compile_error!("QUIET_NAN_BITS and SIGNALLING_NAN_BITS not defined for this architecture.");

pub(crate) const VALUE_TAG_MASK: u64 =
    0b1111111111111111000000000000000000000000000000000000000000000000;
pub(crate) const VALUE_PAYLOAD_MASK: u64 = !VALUE_TAG_MASK;
// A quiet NaN on *most* platforms.
pub(crate) const CANONICAL_NAN_BITS: u64 =
    0b0111111111110000000000000000000000000000000000000000000000000000;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    // NOTE: Quiet NaN
    Double = 0b0000,
    Attrset = 0b0001,
    Function = 0b0010,
    String = 0b0011,
    Path = 0b0100,
    List = 0b0101,
    Integer = 0b0110,
    Bool = 0b0111,
    Lazy = 0b1000,
    Null = 0b1001,
}

impl ValueKind {
    #[inline(always)]
    pub(crate) const fn as_nan_bits(self) -> u64 {
        SIGNALLING_NAN_BITS | ((self as u64) << 48)
    }

    /// as_nan(self) >> 48
    #[inline(always)]
    pub(crate) const fn as_shifted(self) -> u16 {
        (self.as_nan_bits() >> 48) as u16
    }
}

#[repr(transparent)]
pub struct Value(u64);

impl Value {
    pub const NULL: Value = unsafe { Value::from_raw_parts(ValueKind::Null, 0) };
    pub const TRUE: Value = unsafe { Value::from_raw_parts(ValueKind::Bool, 1) };
    pub const FALSE: Value = unsafe { Value::from_raw_parts(ValueKind::Bool, 0) };

    #[inline(always)]
    pub(crate) const unsafe fn from_raw(value: u64) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub(crate) const fn into_raw(self) -> u64 {
        self.0
    }

    #[inline(always)]
    pub(crate) const fn as_raw(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    const unsafe fn from_raw_parts(kind: ValueKind, payload: u64) -> Self {
        // I love not being able to assert in const fns...
        // debug_assert_eq!(payload & !((1 << 49) - 1), 0);
        Self::from_raw(payload | kind.as_nan_bits())
    }

    pub(crate) const fn from_bool(value: bool) -> Value {
        if value {
            Value::TRUE
        } else {
            Value::FALSE
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn as_pointer_bits(&self) -> u64 {
        unsafe {
            std::mem::transmute::<_, u64>((std::mem::transmute::<_, i64>(self.0 << 16)) >> 16)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    compile_error!("Value::as_pointer_bits is not implemented for this architecture");

    pub fn kind(&self) -> ValueKind {
        if self.0 & NAN_BITS_MASK != SIGNALLING_NAN_BITS {
            return ValueKind::Double;
        }
        let kind = (self.0 & (VALUE_TAG_MASK - NAN_BITS_MASK)) >> 48;
        unsafe { std::mem::transmute(kind as u8) }
    }

    #[inline(always)]
    pub fn is(&self, kind: ValueKind) -> bool {
        self.0 & VALUE_TAG_MASK == kind.as_nan_bits()
    }

    pub fn unpack(&self) -> UnpackedValue {
        macro_rules! unpack_ptr {
            ($kind: ident) => {
                UnpackedValue::$kind(unsafe {
                    let raw = self.as_pointer_bits() as *mut _;
                    Rc::increment_strong_count(raw);
                    Rc::from_raw(raw)
                })
            };
        }
        match self.kind() {
            ValueKind::Integer => UnpackedValue::Integer(unsafe {
                std::mem::transmute((self.0 & VALUE_PAYLOAD_MASK) as u32)
            }),
            ValueKind::Double => UnpackedValue::Double(f64::from_bits(self.0)),
            ValueKind::String => unpack_ptr!(String),
            ValueKind::Path => unpack_ptr!(Path),
            ValueKind::List => unpack_ptr!(List),
            ValueKind::Attrset => unpack_ptr!(Attrset),
            ValueKind::Function => unpack_ptr!(Function),
            // FIXME: this pointer may not be in x86_64 canonical form
            ValueKind::Lazy => unsafe {
                let raw = self.as_pointer_bits() as *mut _;
                Rc::increment_strong_count(raw);
                UnpackedValue::Lazy(LazyValue(Rc::from_raw(raw)))
            },
            ValueKind::Bool => UnpackedValue::Bool(self.0 == Value::TRUE.0),
            ValueKind::Null => UnpackedValue::Null,
        }
    }

    pub(crate) fn into_unpacked(self) -> UnpackedValue {
        macro_rules! unpack_ptr {
            ($kind: ident) => {
                UnpackedValue::$kind(unsafe { Rc::from_raw(self.as_pointer_bits() as *mut _) })
            };
        }
        match self.kind() {
            ValueKind::Integer => UnpackedValue::Integer(unsafe {
                std::mem::transmute((self.0 & VALUE_PAYLOAD_MASK) as u32)
            }),
            ValueKind::Double => UnpackedValue::Double(f64::from_bits(self.0)),
            ValueKind::Bool => UnpackedValue::Bool(self.0 == Value::TRUE.0),
            ValueKind::String => unpack_ptr!(String),
            ValueKind::Path => unpack_ptr!(Path),
            ValueKind::List => unpack_ptr!(List),
            ValueKind::Attrset => unpack_ptr!(Attrset),
            ValueKind::Function => unpack_ptr!(Function),
            ValueKind::Lazy => unsafe {
                UnpackedValue::Lazy(LazyValue(Rc::from_raw(self.as_pointer_bits() as *const _)))
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
            UnpackedValue::Double(_) => (),
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
        if self.is(ValueKind::Lazy) {
            unsafe { (*(self.as_pointer_bits() as *mut LazyValueImpl)).evaluate() }
        } else {
            self
        }
    }

    pub(crate) fn evaluate_mut(&mut self) -> &Value {
        if self.is(ValueKind::Lazy) {
            unsafe {
                LazyValue(Rc::from_raw(self.as_pointer_bits() as *mut _))
                    .evaluate()
                    .clone_into(self)
            }
        }
        self
    }

    pub(crate) fn into_evaluated(self) -> Value {
        if self.is(ValueKind::Lazy) {
            unsafe {
                (*(self.as_pointer_bits() as *mut LazyValueImpl))
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

    pub fn is_true(&self) -> bool {
        self.0 == Self::TRUE.0
    }

    pub fn is_false(&self) -> bool {
        self.0 == Self::FALSE.0
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self.evaluate().unpack(), other.evaluate().unpack()) {
            (
                UnpackedValue::Integer(_) | UnpackedValue::Bool(_) | UnpackedValue::Null,
                UnpackedValue::Integer(_) | UnpackedValue::Bool(_) | UnpackedValue::Null,
            ) => self.0 == other.0,
            (UnpackedValue::String(a), UnpackedValue::String(b)) => a == b,
            (UnpackedValue::Path(a), UnpackedValue::Path(b)) => a == b,
            (UnpackedValue::Double(a), UnpackedValue::Double(b)) => a == b,
            (UnpackedValue::Double(a), UnpackedValue::Integer(b)) => a == b.into(),
            (UnpackedValue::Integer(a), UnpackedValue::Double(b)) => f64::from(a) == b,
            (UnpackedValue::Attrset(a), UnpackedValue::Attrset(b)) => unsafe {
                *a.get() == *b.get()
            },
            (UnpackedValue::List(a), UnpackedValue::List(b)) => unsafe { *a.get() == *b.get() },
            (UnpackedValue::Function(_), UnpackedValue::Function(_)) => false,
            (a, b) if a.kind() != b.kind() => false,
            (_, _) => unreachable!(),
        }
    }
}

impl Value {
    pub(crate) extern "C-unwind" fn equal(self, other: Value) -> Value {
        (self == other).into()
    }

    pub(crate) extern "C-unwind" fn not_equal(self, other: Value) -> Value {
        (self != other).into()
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
    // TODO: support mixing Integer and Double operands
    value_impl_binary_operators! {
        +add(a, b) {
            Integer => a + b,
            Double => a + b,
            String => {
                    let mut result = String::with_capacity(a.len() + b.len());
                    result += &a;
                    result += &b;
                    result
            }
        };
        -sub(a, b) { Integer => a - b, Double => a - b };
        *mul(a, b) { Integer => a * b, Double => a * b };
        /div(a, b) { Integer => a / b, Double => a / b };
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

#[macro_export]
macro_rules! value {
    ([$($tt: tt)*]) => {
        $crate::value::UnpackedValue::new_list(#[allow(clippy::vec_init_then_push)] {
            let mut _result = Vec::with_capacity($crate::value!(@listcount 0; $($tt)*));
            $crate::value!(@list _result; $($tt)*);
            _result
        }).pack()
    };
    ({$($tt: tt)*}) => {
        $crate::value::UnpackedValue::new_attrset({
            let mut _result = $crate::value::ValueMap::new();
            $crate::value!(@map _result; $($tt)*);
            _result
        }).pack()
    };
    (null) => { $crate::value::Value::NULL };
    ($expr: expr) => { $crate::value::Value::from($expr) };

    (@key [$key: expr]) => { $key };
    (@key $key: ident) => { stringify!($key).to_string() };

    (@listcount $current: expr; $value: expr$(, $($rest: tt)*)?) => { $crate::value!(@listcount $current + 1; $($($rest)*)?) };
    (@listcount $current: expr;) => { $current };
    (@list $out: ident; $value: tt $(, $($rest: tt)*)?) => {
        $out.push($crate::value!($value));
        $crate::value!(@list $out; $($($rest)*)?)
    };
    (@list $out: ident; $value: expr $(, $($rest: tt)*)?) => {
        $out.push($crate::value::Value::from($value));
        $crate::value!(@list $out; $($($rest)*)?)
    };
    (@list $out: ident;) => { };

    (@map $out: ident; $key: tt = $value: tt; $($rest: tt)*) => {
        $out.insert($crate::value!(@key $key), $crate::value!($value));
        $crate::value!(@map $out; $($rest)*);
    };
    (@map $out: ident; $key: tt = $value: expr; $($rest: tt)*) => {
        $out.insert($crate::value::Value::from(@key $key), $crate::value!($value));
        $crate::value!(@map $out; $($rest)*);
    };
    (@map $out: ident;) => { };
}

#[test]
fn test_value_macro() {
    assert_eq!(value!(20), 20.into());
    assert_eq!(value!("hello".to_string()), "hello".to_string().into());
    assert_eq!(value!(253.11), 253.11.into());

    assert_eq!(
        value!([253.11, 1000]),
        UnpackedValue::new_list(vec![value!(253.11), value!(1000)]).pack()
    );

    assert_eq!(
        value!({
            hello = 10;
            ["computed key".to_string()] = [124, "1245"];
            [9.to_string()] = {
                even = false;
                odd = true;
            };
        }),
        UnpackedValue::new_attrset({
            let mut result = ValueMap::new();
            result.insert("hello".to_string(), 10.into());
            result.insert("computed key".to_string(), value!([124, "1245"]));
            result.insert(
                9.to_string(),
                value!({
                    even = false;
                    odd = true;
                }),
            );
            result
        })
        .pack()
    );
}
