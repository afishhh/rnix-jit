#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::fn_to_numeric_cast)]
#![allow(clippy::type_complexity)]
// This can be replaced later
#![feature(offset_of_enum)]
// Rust's comedic const eval support
#![feature(const_float_bits_conv)]

/// Nothing in this API is stable.
use std::{cell::UnsafeCell, mem::offset_of, ops::Deref, path::PathBuf, rc::Rc};

use dwarf::*;

mod compiler;
mod ir;
#[doc(hidden)]
pub mod perfstats;
pub mod value;
use ir::*;
use value::*;
mod interpreter;
mod runnable;

enum ImplicitScopeStorage {
    Evaluated(Rc<UnsafeCell<ValueMap>>),
    Lazy(Option<Box<dyn FnOnce() -> Rc<UnsafeCell<ValueMap>>>>),
}

static EMPTY_VALUE_MAP: ValueMap = ValueMap::new();

struct ImplicitScope {
    storage: UnsafeCell<ImplicitScopeStorage>,
    previous: Option<Rc<ImplicitScope>>,
}

impl ImplicitScope {
    pub fn evaluate(&self) -> &UnsafeCell<ValueMap> {
        unsafe {
            match &mut *self.storage.get() {
                ImplicitScopeStorage::Evaluated(x) => &*x,
                ImplicitScopeStorage::Lazy(x) => {
                    *self.storage.get() =
                        ImplicitScopeStorage::Evaluated(std::mem::take(x).unwrap_unchecked()());
                    let ImplicitScopeStorage::Evaluated(x) = &*self.storage.get() else {
                        std::hint::unreachable_unchecked();
                    };
                    &*x
                }
            }
        }
    }
}

enum ScopeStorage {
    // Used for recursive attribute sets
    Attrset,
    Scope(ValueMap),
    Unowned,
}

#[repr(C)]
struct Scope {
    values: *const ValueMap,
    previous: *mut Scope,
    storage: ScopeStorage,
    /// The corresponding implicit scope.
    /// This must be copied when creating child scopes.
    implicit_scope: Option<Rc<ImplicitScope>>,
}

impl Scope {
    fn from_map(map: ValueMap, previous: *mut Scope) -> *mut Scope {
        unsafe {
            let scope_ptr = Box::leak(Box::new(Scope {
                values: std::ptr::null(),
                storage: ScopeStorage::Scope(map),
                previous,
                implicit_scope: previous.as_ref().and_then(|x| x.implicit_scope.clone()),
            })) as *mut Scope;
            (*scope_ptr).values =
                (scope_ptr as *const u8).add(offset_of!(Scope, storage.Scope.0)) as *const _;
            scope_ptr
        }
    }

    fn from_attrs(map: Rc<UnsafeCell<ValueMap>>, previous: *mut Scope) -> *mut Scope {
        unsafe {
            Box::leak(Box::new(Scope {
                values: (*Rc::<UnsafeCell<ValueMap>>::into_raw(map)).get(),
                previous,
                storage: ScopeStorage::Attrset,
                implicit_scope: previous.as_ref().and_then(|x| x.implicit_scope.clone()),
            }))
        }
    }

    fn with_new_lazy_implicit(
        previous: *mut Scope,
        evaluate: impl FnOnce() -> Rc<UnsafeCell<ValueMap>> + 'static,
    ) -> *mut Scope {
        unsafe {
            Box::leak(Box::new(Scope {
                values: &EMPTY_VALUE_MAP,
                previous,
                storage: ScopeStorage::Unowned,
                implicit_scope: Some(Rc::new(ImplicitScope {
                    storage: UnsafeCell::new(ImplicitScopeStorage::Lazy(Some(Box::new(evaluate)))),
                    previous: (*previous).implicit_scope.clone(),
                })),
            }))
        }
    }

    // This function does not traverse implicit scopes.
    unsafe fn lookup_local(mut scope: *mut Scope, name: &str) -> Option<Value> {
        loop {
            match (*scope).get(name) {
                Some(value) => return Some(value.clone()),
                None => {
                    scope = (*scope).previous;
                    if scope.is_null() {
                        return None;
                    }
                }
            }
        }
    }

    unsafe fn lookup_implicit(mut scope: *const ImplicitScope, name: &str) -> Option<Value> {
        loop {
            match (*(*scope).evaluate().get()).get(name) {
                Some(value) => return Some(value.clone()),
                None => {
                    scope = &**((*scope).previous.as_ref()?);
                    if scope.is_null() {
                        return None;
                    }
                }
            }
        }
    }

    unsafe fn lookup(scope: *mut Scope, name: &str) -> Value {
        Scope::lookup_local(scope, name).unwrap_or_else(|| {
            (*scope)
                .implicit_scope
                .as_ref()
                .and_then(|implicit| Scope::lookup_implicit(&**implicit, name))
                .unwrap_or_else(|| throw!("identifier {name} not found"))
        })
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

mod builtins;
mod dwarf;
mod exception;
mod unwind;
use exception::*;

pub fn pretty_print_exception(exception: NixException) {
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

pub use builtins::{import, seq};
pub use exception::catch_nix_unwind;

pub fn eval(root: PathBuf, filename: String, code: String) -> Result<Value, NixException> {
    catch_nix_unwind(move || builtins::eval_throwing(root, filename, code))
}
