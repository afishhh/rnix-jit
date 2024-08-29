use std::{mem::ManuallyDrop, rc::Rc};

use crate::{
    compiler::{Executable, COMPILER},
    runnable::{Runnable, RunnableVTable},
    throw, LazyValue, Parameter, Program, Scope, Value, ValueMap,
};

mod eval;

struct Interpreted {
    program: Rc<Program>,
    parameter: Option<Parameter>,
}

pub fn into_interpreted(program: Rc<Program>, parameter: Option<Parameter>) -> Rc<Runnable<()>> {
    unsafe extern "C-unwind" fn run(
        this: *const Runnable,
        scope: *mut Scope,
        _arg: Value,
    ) -> Value {
        unsafe {
            let interpreted = &**(*Runnable::upcast::<Interpreted>(this)).inner.get();
            assert!(interpreted.parameter.is_none());
            eval::interpret(scope, &interpreted.program, usize::MAX)
        }
    }

    unsafe {
        Runnable::new(
            RunnableVTable::with_default_drop::<Interpreted>(run),
            Interpreted { program, parameter },
        )
        .into_erased_rc()
    }
}

fn create_function_scope(
    previous: *mut Scope,
    parameter: &Parameter,
    mut value: Value,
) -> *mut Scope {
    let scope = Scope::from_map(ValueMap::new(), previous);
    let scope_map = unsafe { &mut *((*scope).values as *mut ValueMap) };
    match parameter {
        Parameter::Ident(name) => {
            scope_map.insert(name.clone(), value);
        }
        Parameter::Pattern {
            entries,
            binding,
            ignore_unmatched,
        } => {
            let x = value.evaluate_mut().unpack_typed_ref_or_throw::<ValueMap>();
            let attrs = unsafe { &*x.get() };
            let mut found_keys = 0;
            for (name, default) in entries {
                let value = match attrs.get(name).cloned() {
                    Some(value) => {
                        found_keys += 1;
                        value
                    }
                    None => match default {
                        Some(fallback) => LazyValue::from_runnable(
                            scope,
                            into_interpreted(fallback.clone(), None),
                        )
                        .into(),
                        None => throw!("missing pattern entry"),
                    },
                };
                scope_map.insert(name.clone(), value);
            }
            if !ignore_unmatched {
                assert_eq!(found_keys, attrs.len());
            }
            if let Some(name) = binding {
                scope_map.insert(name.clone(), value);
            }
        }
    };
    scope
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
    program: Rc<Program>,
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
                    .compile(&interpreted.program, interpreted.parameter.as_ref())
                    .unwrap();
                // NOTE: One of the few reasons why this works is because Runnable<Executable>
                //       doesn't actually care what Runnable it's in.
                (*this.vtable.get()).run = compiled.vtable().run;
                compiled.run(scope, arg)
            } else {
                inner.times_executed += 1;
                let program = &inner.program.interpreted.program;
                let scope = if let Some(param) = &inner.program.interpreted.parameter {
                    create_function_scope(scope, param, arg)
                } else {
                    scope
                };
                eval::interpret(scope, program, inner.compilation_threshold)
            }
        }
    }

    unsafe {
        Runnable::new(
            RunnableVTable::with_default_drop::<DynamicallyCompiled>(run),
            DynamicallyCompiled {
                times_executed: 0,
                compilation_threshold,
                program: DynamicallyCompiledInner {
                    interpreted: ManuallyDrop::new(Interpreted { program, parameter }),
                },
            },
        )
        .into_erased_rc()
    }
}
