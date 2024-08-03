use std::{
    cell::UnsafeCell, ffi::CStr, mem::ManuallyDrop, ops::Deref, path::PathBuf, ptr::NonNull, rc::Rc,
};

use crate::{
    runnable::Runnable, throw, value::LazyValueImpl, AttrsetValue, CreateValueMap, Function,
    LazyValue, Scope, UnpackedValue, Value, ValueList, ValueMap,
};

use super::{CompiledParameter, Executable, COMPILER};

// TODO: use extract_typed! more in this module

pub unsafe extern "C-unwind" fn scope_lookup(
    scope: *mut Scope,
    name: *const u8,
    name_len: usize,
) -> Value {
    let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(name, name_len));
    Scope::lookup(scope, name)
}

pub unsafe extern "C-unwind" fn create_function_scope(
    parameter: *const CompiledParameter,
    previous: *mut Scope,
    mut value: Value,
) -> *mut Scope {
    match &*parameter {
        CompiledParameter::Ident(name) => {
            // println!("creating function scope {previous:?} {name} {value:?}");
            let values = {
                let mut map = ValueMap::new();
                map.insert(name.to_string(), value);
                map
            };
            Scope::from_map(values, previous)
        }
        CompiledParameter::Pattern {
            entries,
            binding,
            ignore_unmatched,
        } => {
            let UnpackedValue::Attrset(heapvalue) = value.evaluate_mut().unpack() else {
                panic!("cannot unpack non-attrset value in pattern parameter");
            };
            let scope = Scope::from_map(ValueMap::new(), previous);
            let scope_map = &mut *((*scope).values as *mut ValueMap);
            let mut found_keys = 0;
            for (key, default) in entries.iter() {
                let value = (*heapvalue.get()).get(key);
                found_keys += value.is_some() as usize;
                let Some(value) = value.cloned().or_else(|| {
                    default
                        .as_ref()
                        .map(|x| LazyValue::from_jit(scope, x.clone()).into())
                }) else {
                    panic!("missing pattern entry");
                };
                scope_map.insert(key.to_string(), value);
            }
            if !ignore_unmatched {
                assert_eq!(found_keys, (*heapvalue.get()).len());
            }
            if let Some(binding) = binding {
                scope_map.insert(binding.to_string(), value);
            }
            // scope.get("hello");
            let mut s = String::new();
            UnpackedValue::fmt_attrset_display(scope_map, 0, &mut s, true).unwrap();
            // println!("{s}");
            scope
        }
    }
}

// TODO: JIT this instead
pub unsafe fn value_map_create(
    mut scope: *mut Scope,
    create: &CreateValueMap,
) -> (Rc<UnsafeCell<ValueMap>>, *mut Scope) {
    let result = Rc::new(UnsafeCell::new(ValueMap::new()));

    if create.rec {
        scope = Scope::from_attrs(result.clone(), scope);
    }

    let lazy_parents = create
        .parents
        .iter()
        .cloned()
        // TODO: HACK: DO NOT DO THIS
        .map(|x| unsafe { COMPILER.compile(x, None) }.unwrap())
        .map(|x| LazyValue::from_jit(scope, x))
        .collect::<Vec<_>>();

    {
        let result_mut = &mut *result.get();

        let to_value = |key: &str, value: &AttrsetValue| match value {
            AttrsetValue::Load => Scope::lookup(scope, key),
            AttrsetValue::Inherit(idx) => {
                let key = key.to_string();
                Value::from(lazy_parents[*idx].clone()).lazy_map(move |x| {
                    if let UnpackedValue::Attrset(set) = x.into_evaluated().into_unpacked() {
                        (*set.get()).get(&key).unwrap().clone()
                    } else {
                        throw!("Cannot inherit from non attribute set value")
                    }
                })
            }
            AttrsetValue::Childset(child) => {
                UnpackedValue::Attrset(value_map_create(scope, child).0).pack()
            }
            AttrsetValue::Program(program) => {
                // TODO: HACK: DO NOT DO THIS
                LazyValue::from_jit(
                    scope,
                    unsafe { COMPILER.compile(program.clone(), None) }.unwrap(),
                )
                .into()
            }
        };

        for (key, value) in create.constant.iter() {
            let value = to_value(&key, &value);
            result_mut.insert(key.to_string(), value);
        }

        for (key, value) in create.dynamic.iter() {
            // TODO: HACK: DO NOT DO THIS
            let UnpackedValue::String(key) =
                unsafe { COMPILER.compile(key.clone(), None).unwrap() }
                    .run(scope, Value::NULL)
                    .into_unpacked()
            else {
                throw!("Non-string attrset key");
            };
            let value = to_value(&key, &value);
            result_mut.insert(Rc::unwrap_or_clone(key), value);
        }
    }

    (result, scope)
}

pub unsafe extern "C-unwind" fn scope_create(
    previous: *mut Scope,
    create: *const CreateValueMap,
) -> *mut Scope {
    value_map_create(previous, &*create).1
}

pub unsafe fn attrset_create(scope: *mut Scope, create: *const CreateValueMap) -> Value {
    UnpackedValue::Attrset(value_map_create(scope, &*create).0).pack()
}

pub unsafe extern "C-unwind" fn attrset_get(
    values: *const Value,
    components: usize,
    scope: *mut Scope,
    fallback: *const Runnable<Executable>,
) -> Value {
    let mut current = values.add(components).read();
    for i in 0..components {
        let UnpackedValue::String(name) = values.add(i).read().into_unpacked() else {
            throw!("attribute name is not a string")
        };
        current = match current.into_evaluated().into_unpacked() {
            UnpackedValue::Attrset(x) => {
                let map = &*x.get();
                match map.get(name.as_str()).cloned() {
                    Some(x) => x,
                    None => {
                        if fallback.is_null() {
                            throw!("{map:?}.{name:?} does not exist")
                        } else {
                            return LazyValue::from_jit(scope, unsafe {
                                Rc::increment_strong_count(fallback);
                                Rc::from_raw(fallback)
                            })
                            .into();
                        }
                    }
                }
            }
            _ => throw!("not an attribute set"),
        };
    }

    current
}

unsafe fn attrset_hasattrpath_impl(map: &ValueMap, names: &[&String]) -> Value {
    let (current, rest) = names.split_first().unwrap();
    match (
        map.get(*current).map(Value::evaluate).map(Value::unpack),
        rest,
    ) {
        (Some(_), []) => Value::TRUE,
        (Some(UnpackedValue::Attrset(next)), rest) => attrset_hasattrpath_impl(&*next.get(), rest),
        (_, _) => Value::FALSE,
    }
}

pub unsafe extern "C-unwind" fn attrset_hasattrpath(
    map: &ValueMap,
    names: *const &String,
    names_len: usize,
) -> Value {
    let names = std::slice::from_raw_parts(names, names_len);
    attrset_hasattrpath_impl(map, names)
}

pub unsafe extern "C-unwind" fn attrset_update(left: &ValueMap, right: &ValueMap) -> Value {
    UnpackedValue::new_attrset({
        let mut result = left.clone();
        for (key, value) in right.iter() {
            result.insert(key.to_string(), value.clone());
        }
        result
    })
    .pack()
}

pub unsafe extern "C-unwind" fn create_function_value(
    runnable: *const Runnable,
    scope: *mut Scope,
) -> Value {
    UnpackedValue::Function(Rc::new(Function::new(
        {
            Rc::increment_strong_count(runnable);
            Rc::from_raw(runnable)
        },
        scope,
    )))
    .pack()
}

pub unsafe extern "C-unwind" fn list_create() -> Value {
    UnpackedValue::new_list(Vec::new()).pack()
}

pub unsafe extern "C-unwind" fn list_append_value(
    list: &mut ValueList,
    scope: *mut Scope,
    executable: *const Runnable<Executable>,
) {
    list.push(
        UnpackedValue::Lazy(LazyValue::from_jit(scope, unsafe {
            Rc::increment_strong_count(executable);
            Rc::from_raw(executable)
        }))
        .pack(),
    );
}

pub unsafe extern "C-unwind" fn list_concat(a: Value, b: Value) -> Value {
    if let (UnpackedValue::List(a), UnpackedValue::List(b)) = (a.unpack(), b.unpack()) {
        UnpackedValue::new_list(
            (*a.get())
                .iter()
                .chain((*b.get()).iter())
                .cloned()
                .collect(),
        )
        .pack()
    } else {
        panic!("concat attempted on non-list operands")
    }
}

pub unsafe extern "C-unwind" fn string_mut_append(from: NonNull<String>, to: &mut String) {
    assert_ne!(from.as_ptr(), to as *mut _);
    *to += from.as_ref();
    Rc::decrement_strong_count(from.as_ptr());
}

pub unsafe extern "C-unwind" fn value_into_evaluated(a: *mut UnsafeCell<LazyValueImpl>) -> Value {
    LazyValue(Rc::from_raw(a)).evaluate().clone()
}

pub unsafe extern "C-unwind" fn value_ref(a: ManuallyDrop<Value>) -> Value {
    a.deref().clone()
}

pub unsafe extern "C-unwind" fn value_string_to_mut(a: Value) -> Value {
    macro_rules! clonerc {
        ($x: expr) => {
            if Rc::strong_count(&$x) > 1 {
                Rc::new((*$x).clone())
            } else {
                $x
            }
        };
    }
    match a.into_unpacked() {
        UnpackedValue::String(x) => UnpackedValue::String(clonerc!(x)),
        UnpackedValue::Path(x) => UnpackedValue::new_string(x.to_str().unwrap().to_string()),
        UnpackedValue::Lazy(x) => return value_string_to_mut(x.evaluate().clone()),
        other => other,
    }
    .pack()
}

pub unsafe extern "C-unwind" fn value_string_to_path(a: Value) -> Value {
    match a.into_unpacked() {
        UnpackedValue::String(x) => UnpackedValue::new_path(PathBuf::from(Rc::unwrap_or_clone(x))),
        UnpackedValue::Lazy(x) => return value_string_to_path(x.evaluate().clone()),
        other => other,
    }
    .pack()
}

pub unsafe extern "C-unwind" fn asm_panic(msg: *const i8) {
    panic!("[JITPANIC] {}", CStr::from_ptr(msg).to_string_lossy());
}

pub unsafe extern "C-unwind" fn asm_panic_with_value(msg: *const i8, value: Value) {
    println!("[JITPANIC] {}", CStr::from_ptr(msg).to_string_lossy());
    println!(
        "[JITPANIC] Value passed to asm_panic_with_value: {} {value:?}",
        value.as_raw()
    );
    panic!("jitpanic")
}

pub unsafe extern "C-unwind" fn asm_throw(msg: *const i8) {
    throw!("{}", CStr::from_ptr(msg).to_string_lossy());
}
