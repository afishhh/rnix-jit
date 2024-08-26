use std::{
    cell::UnsafeCell, ffi::CStr, mem::ManuallyDrop, ops::Deref, path::PathBuf, ptr::NonNull, rc::Rc,
};

use crate::{
    runnable::Runnable, throw, value::LazyValueImpl, Function, LazyValue, Scope, SourceSpan,
    UnpackedValue, Value, ValueList, ValueMap, ValueUnpackable,
};

use super::{CompiledParameter, Executable};

// TODO: use extract_typed! more in this module

pub unsafe extern "C-unwind" fn scope_lookup(scope: *mut Scope, name: *const String) -> Value {
    Scope::lookup(scope, &*name)
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
            let heapvalue = ValueMap::unpack_from_ref_or_throw(value.evaluate_mut());
            let scope = Scope::from_map(ValueMap::new(), previous);
            let scope_map = &mut *((*scope).values as *mut ValueMap);
            let mut found_keys = 0;
            for (key, default) in entries.iter() {
                let value = (*heapvalue.get()).get(key);
                found_keys += value.is_some() as usize;
                let Some(value) = value.cloned().or_else(|| {
                    default
                        .as_ref()
                        .map(|x| LazyValue::from_runnable(scope, x.clone()).into())
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

pub unsafe extern "C" fn create_lazy_value(scope: *mut Scope, runnable: *const Runnable) -> Value {
    LazyValue::from_runnable(scope, {
        Rc::increment_strong_count(runnable);
        Rc::from_raw(runnable)
    })
    .into()
}

pub unsafe extern "C-unwind" fn create_lazy_attrset_access(
    attrset: ManuallyDrop<Value>,
    keyv: Value,
) -> Value {
    let attrset = ManuallyDrop::into_inner(attrset.clone());
    LazyValue::new_attribute_access(attrset, keyv).into()
}

pub unsafe extern "C" fn map_create() -> *mut ValueMap {
    Rc::into_raw(Rc::new(ValueMap::new())) as *mut _
}

pub unsafe extern "C" fn map_create_recursive_scope(
    map: *mut ValueMap,
    previous: *mut Scope,
) -> *mut Scope {
    Scope::from_attrs(Rc::from_raw(map as *const _), previous)
}

pub unsafe extern "C" fn map_insert_constant(map: *mut ValueMap, key: *const String, value: Value) {
    (*map).insert((*key).clone(), value);
}

pub unsafe extern "C-unwind" fn map_insert_dynamic(map: *mut ValueMap, key: Value, value: Value) {
    let x = String::unpack_from_or_throw(key.into_evaluated());
    (*map).insert(Rc::unwrap_or_clone(x), value);
}

pub unsafe extern "C" fn map_into_attrset(map: *mut ValueMap) -> Value {
    UnpackedValue::Attrset(Rc::from_raw(map as *const _)).pack()
}

pub unsafe extern "C" fn map_into_scope(map: *mut ValueMap, previous: *mut Scope) -> *mut Scope {
    Scope::from_attrs(Rc::from_raw(map as *const _), previous)
}

pub unsafe extern "C-unwind" fn scope_create_with(
    previous: *mut Scope,
    namespace: *const Runnable,
) -> *mut Scope {
    let namespace_rc = Rc::from_raw(namespace);
    Scope::with_new_lazy_implicit(previous, move || {
        ValueMap::unpack_from_or_throw(namespace_rc.run(previous, Value::NULL))
    })
}

pub unsafe extern "C-unwind" fn attrset_get(
    values: *const Value,
    components: usize,
    scope: *mut Scope,
    fallback: *const Runnable<Executable>,
) -> Value {
    let mut current = values.add(components).read();
    for i in 0..components {
        let name = String::unpack_from_or_throw(values.add(i).read());
        let map = ValueMap::unpack_from_or_throw(current.into_evaluated());
        let map = &*map.get();
        current = match map.get(&*name).cloned() {
            Some(x) => x,
            None => {
                if fallback.is_null() {
                    throw!("{map:?}.{name:?} does not exist")
                } else {
                    return LazyValue::from_runnable(scope, unsafe {
                        Rc::increment_strong_count(fallback);
                        Rc::from_raw(fallback)
                    })
                    .into();
                }
            }
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
        UnpackedValue::Lazy(LazyValue::from_runnable(scope, unsafe {
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
