use std::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop};

use crate::{Scope, Value};

type RunMethodPtr =
    unsafe extern "C-unwind" fn(runnable: *const Runnable, scope: *mut Scope, arg: Value) -> Value;

type DropMethodPtr = fn(runnable: *const Runnable);

#[derive(Debug)]
pub(crate) struct RunnableVTable {
    pub(crate) run: RunMethodPtr,
    pub(crate) drop_in_place: DropMethodPtr,
}

// Basically a simple trait but with a custom vtable
#[repr(C)] // A predictable layout is required
#[derive(Debug)]
pub struct Runnable<T = ()> {
    // Make this !Send
    _marker: PhantomData<*const u8>,
    pub(crate) vtable: RunnableVTable,
    pub(crate) inner: UnsafeCell<ManuallyDrop<T>>,
}

impl<T> Runnable<T> {
    pub(crate) unsafe fn new(vtable: RunnableVTable, value: T) -> Self {
        Self {
            _marker: PhantomData,
            vtable,
            inner: UnsafeCell::new(ManuallyDrop::new(value)),
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn run(&self, scope: *mut Scope, arg: Value) -> Value {
        (self.vtable.run)(Self::erase(self), scope, arg)
    }

    #[inline(always)]
    pub(crate) fn erase(this: *const Runnable<T>) -> *const Runnable<()> {
        unsafe { &*(this as *const Runnable) }
    }
}

impl Runnable {
    pub(crate) unsafe fn upcast<T>(ptr: *const Runnable) -> *const Runnable<T> {
        ptr as *const Runnable<T>
    }
}

impl<T> Drop for Runnable<T> {
    fn drop(&mut self) {
        (self.vtable.drop_in_place)(Self::erase(self))
    }
}

impl<F: FnMut(Value) -> Value + 'static> Runnable<F> {
    pub(crate) fn from_closure(closure: F) -> Runnable<F> {
        unsafe extern "C-unwind" fn run<F: FnMut(Value) -> Value + 'static>(
            this: *const Runnable,
            _scope: *mut Scope,
            arg: Value,
        ) -> Value {
            eprintln!("Runnable::<FnMut(Value) -> Value + 'static>::run() called with arg {arg:?}");
            (*(*Runnable::upcast::<F>(this)).inner.get())(arg)
        }

        unsafe {
            Runnable::new(
                RunnableVTable {
                    run: run::<F>,
                    drop_in_place: |this| {
                        std::ptr::drop_in_place((*Runnable::upcast::<F>(this)).inner.get())
                    },
                },
                closure,
            )
        }
    }
}
