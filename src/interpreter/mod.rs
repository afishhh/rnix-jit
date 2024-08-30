use std::rc::Rc;

use crate::{
    compiler::COMPILER,
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

struct DynamicallyCompiled {
    compilation_threshold: usize,
    program: Rc<Program>,
    parameter: Option<Parameter>,
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
            if inner.program.executions.get() >= inner.compilation_threshold {
                eprintln!(
                    "Runnable at {:?} executed {} times, JIT compiling",
                    this as *const _,
                    inner.program.executions.get()
                );
                let compiled = COMPILER
                    .compile(&inner.program, inner.parameter.as_ref())
                    .unwrap();
                *inner.program.compiled.get() = Some(compiled.clone());
                // NOTE: One of the few reasons why this works is because Runnable<Executable>
                //       doesn't actually care what Runnable it's in.
                (*this.vtable.get()).run = compiled.vtable().run;
                compiled.run(scope, arg)
            } else {
                inner
                    .program
                    .executions
                    .set(inner.program.executions.get() + 1);
                let scope = if let Some(param) = &inner.parameter {
                    create_function_scope(scope, param, arg)
                } else {
                    scope
                };
                eval::interpret(scope, &inner.program, inner.compilation_threshold)
            }
        }
    }

    unsafe {
        if let Some(compiled) = &*program.compiled.get() {
            Rc::from_raw(Runnable::erase(Rc::into_raw(compiled.clone())))
        } else {
            Runnable::new(
                RunnableVTable::with_default_drop::<DynamicallyCompiled>(run),
                DynamicallyCompiled {
                    compilation_threshold,
                    program,
                    parameter,
                },
            )
            .into_erased_rc()
        }
    }
}
