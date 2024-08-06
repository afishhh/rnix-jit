use std::{mem::ManuallyDrop, rc::Rc};

use crate::{
    compiler::{Executable, COMPILER},
    runnable::{Runnable, RunnableVTable},
    Operation, Parameter, Program, Scope, Value,
};

struct Interpreted {
    program: Program,
    parameter: Option<Parameter>,
}

fn eval(scope: *mut Scope, program: &Program) -> Value {
    let mut stack: Vec<Value> = vec![];

    // TODO: Program should be a tree structure.
    todo!();

    stack.pop().unwrap()
}

pub fn into_interpreted(program: Program, parameter: Option<Parameter>) -> Runnable<Interpreted> {
    unsafe extern "C-unwind" fn run(
        this: *const Runnable,
        scope: *mut Scope,
        _arg: Value,
    ) -> Value {
        unsafe {
            let interpreted = &**(*Runnable::upcast::<Interpreted>(this)).inner.get();
            assert!(interpreted.parameter.is_none());
            eval(scope, &interpreted.program)
        }
    }

    unsafe {
        Runnable::new(
            RunnableVTable::with_default_drop::<Interpreted>(run),
            Interpreted { program, parameter },
        )
    }
}

union DynamicallyCompiledInner {
    interpreted: ManuallyDrop<Interpreted>,
    compiled: ManuallyDrop<Rc<Runnable<Executable>>>,
}

struct DynamicallyCompiled {
    times_executed: usize,
    compilation_threshold: usize,
    program: DynamicallyCompiledInner,
}

impl Drop for DynamicallyCompiled {
    fn drop(&mut self) {
        unsafe {
            if self.times_executed >= self.compilation_threshold {
                ManuallyDrop::drop(&mut self.program.compiled)
            } else {
                ManuallyDrop::drop(&mut self.program.interpreted)
            }
        }
    }
}

pub fn into_dynamically_compiled(
    program: Program,
    parameter: Option<Parameter>,
    compilation_threshold: usize,
) -> Rc<Runnable> {
    unsafe extern "C-unwind" fn run(this: *const Runnable, scope: *mut Scope, arg: Value) -> Value {
        unsafe {
            let this = &*Runnable::upcast::<DynamicallyCompiled>(this);
            let inner = &mut **this.inner.get();
            if inner.times_executed >= inner.compilation_threshold {
                // println!(
                //     "Runnable at {:?} executed {} times, JIT compiling",
                //     this as *const _, inner.times_executed
                // );
                let interpreted = ManuallyDrop::take(&mut inner.program.interpreted);
                let compiled = COMPILER
                    .compile(interpreted.program, interpreted.parameter)
                    .unwrap();
                // NOTE: One of the few reasons why this works is because Runnable<Executable>
                //       doesn't actually care what Runnable it's in.
                (*this.vtable.get()).run = compiled.vtable().run;
                compiled.run(scope, arg)
            } else {
                inner.times_executed += 1;
                let program = &inner.program.interpreted.program;
                eval(scope, program)
            }
        }
    }

    unsafe {
        Runnable::new_erased_rc(
            RunnableVTable::with_default_drop::<DynamicallyCompiled>(run),
            DynamicallyCompiled {
                times_executed: 0,
                compilation_threshold,
                program: DynamicallyCompiledInner {
                    interpreted: ManuallyDrop::new(Interpreted { program, parameter }),
                },
            },
        )
    }
}
