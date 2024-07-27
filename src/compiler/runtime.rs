use std::{
    cell::UnsafeCell, ffi::CStr, mem::ManuallyDrop, ops::Deref, path::PathBuf, ptr::NonNull, rc::Rc,
};

use crate::{Function, LazyValue, Scope, ScopeStorage, UnpackedValue, Value, ValueList, ValueMap};

use super::{CompiledParameter, Executable};

pub unsafe extern "C-unwind" fn scope_lookup(
    scope: *mut Scope,
    name: *const u8,
    name_len: usize,
) -> Value {
    let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(name, name_len));
    Scope::lookup(scope, name)
}

pub unsafe extern "C-unwind" fn create_function_scope(
    previous: *mut Scope,
    mut value: Value,
    parameter: *const CompiledParameter,
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
            let mut scope_map = &mut *((*scope).values as *mut ValueMap);
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

pub unsafe extern "C" fn scope_create(previous: *mut Scope) -> *mut Scope {
    Scope::from_map(ValueMap::new(), previous)
}

pub unsafe extern "C" fn scope_create_rec(
    attrset: *mut UnsafeCell<ValueMap>,
    previous: *mut Scope,
) -> *mut Scope {
    Scope::from_attrs(Rc::from_raw(attrset), previous)
}

pub unsafe extern "C" fn scope_set(
    scope: *mut Scope,
    key: *const u8,
    key_len: usize,
    value: *const Executable,
) {
    let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(key, key_len));
    Rc::increment_strong_count(value);
    match &mut (*scope).storage {
        ScopeStorage::Attrset => panic!("scope_set called on Attrset Scope"),
        ScopeStorage::Scope(values) => values.insert(
            name.to_string(),
            UnpackedValue::Lazy(LazyValue::from_jit(scope, Rc::from_raw(value))).pack(),
        ),
    };
}

pub unsafe extern "C" fn map_inherit_from(
    scope: *mut Scope,
    map: &mut ValueMap,
    from: *const Executable,
    what: *const String,
    whatn: usize,
) {
    let what = std::slice::from_raw_parts(what, whatn);

    for key in what {
        Rc::increment_strong_count(from);
        let from = Rc::from_raw(from);
        map.insert(
            key.to_string(),
            UnpackedValue::Lazy(LazyValue::from_closure(move || {
                let value = from.run(scope, &Value::NULL).unpack().evaluate();
                let UnpackedValue::Attrset(attrs) = value else {
                    panic!("map_inherit_from: not an attrset");
                };
                (*attrs.get())
                    .get(key)
                    .unwrap_or_else(|| panic!("inherit({:?}) {:?} missing", (*attrs.get()), key))
                    .clone()
            }))
            .pack(),
        );
    }
}

pub unsafe extern "C" fn scope_inherit_parent(
    scope: *mut Scope,
    from: *mut Scope,
    what: *const String,
    whatn: usize,
) {
    let what = std::slice::from_raw_parts(what, whatn);
    let values = match &mut (*scope).storage {
        ScopeStorage::Attrset => panic!("scope_inherit_from called on Attrset Scope"),
        ScopeStorage::Scope(values) => values,
    };
    for key in what {
        values.insert(key.to_string(), Scope::lookup(from, key));
    }
}

pub unsafe extern "C" fn attrset_create() -> Value {
    UnpackedValue::new_attrset(ValueMap::new()).pack()
}

pub unsafe extern "C" fn attrset_set(
    map: &mut ValueMap,
    scope: *mut Scope,
    value: *const Executable,
    name: Value,
) {
    let UnpackedValue::String(name) = name.unpack() else {
        panic!("SetAttr called with non-string name")
    };
    Rc::increment_strong_count(value);
    map.insert(
        Rc::unwrap_or_clone(name),
        UnpackedValue::Lazy(LazyValue::from_jit(scope, Rc::from_raw(value))).pack(),
    );
}

pub unsafe extern "C" fn attrset_inherit_parent(
    attrset: &mut ValueMap,
    from: *mut Scope,
    what: *const String,
    whatn: usize,
) {
    let what = std::slice::from_raw_parts(what, whatn);
    for key in what {
        attrset.insert(key.to_string(), Scope::lookup(from, key));
    }
}

pub unsafe extern "C" fn attrset_get(
    map: &mut ValueMap,
    name: Value,
    scope: *mut Scope,
    fallback: *const Executable,
) -> Value {
    let UnpackedValue::String(name) = name.into_unpacked() else {
        panic!("GetAttr called with non-string name")
    };
    map.get(name.as_str()).cloned().unwrap_or_else(|| {
        if fallback.is_null() {
            panic!("{map:?}.{name:?} does not exist")
        } else {
            LazyValue::from_jit(scope, unsafe {
                Rc::increment_strong_count(fallback);
                Rc::from_raw(fallback)
            })
        }
        .into()
    })
}

pub unsafe extern "C" fn attrset_get_or_insert_attrset(map: &mut ValueMap, name: Value) -> Value {
    let UnpackedValue::String(name) = name.unpack() else {
        panic!("GetAttr called with non-string name")
    };
    map.entry(Rc::unwrap_or_clone(name))
        .or_insert_with(|| UnpackedValue::new_attrset(ValueMap::new()).pack())
        .clone()
}

unsafe fn attrset_hasattrpath_impl(map: &ValueMap, names: &[&String]) -> Value {
    let (current, rest) = names.split_last().unwrap();
    match (
        map.get(*current).map(Value::evaluate).map(Value::unpack),
        rest,
    ) {
        (Some(_), []) => Value::TRUE,
        (Some(UnpackedValue::Attrset(next)), rest) => attrset_hasattrpath_impl(&*next.get(), rest),
        (_, _) => Value::FALSE,
    }
}

pub unsafe extern "C" fn attrset_hasattrpath(
    map: &ValueMap,
    names: *const &String,
    names_len: usize,
) -> Value {
    let names = std::slice::from_raw_parts(names, names_len);
    attrset_hasattrpath_impl(map, names)
}

pub unsafe extern "C" fn attrset_update(left: &ValueMap, right: &ValueMap) -> Value {
    UnpackedValue::new_attrset({
        let mut result = left.clone();
        for (key, value) in right.iter() {
            result.insert(key.to_string(), value.clone());
        }
        result
    })
    .pack()
}

pub unsafe extern "C" fn create_function_value(
    executable: *const Executable,
    scope: *mut Scope,
) -> Value {
    Rc::increment_strong_count(executable);
    UnpackedValue::Function(Rc::new(Function {
        call: (*executable).code(),
        _executable: Some(Rc::from_raw(executable)),
        builtin_closure: None,
        parent_scope: scope,
    }))
    .pack()
}

pub unsafe extern "C" fn list_create() -> Value {
    UnpackedValue::new_list(Vec::new()).pack()
}

pub unsafe extern "C" fn list_append_value(
    list: &mut ValueList,
    scope: *mut Scope,
    executable: *const Executable,
) {
    unsafe {
        Rc::increment_strong_count(executable);
        list.push(UnpackedValue::Lazy(LazyValue::from_jit(scope, Rc::from_raw(executable))).pack());
    }
}

pub unsafe extern "C" fn list_concat(a: Value, b: Value) -> Value {
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

pub unsafe extern "C" fn string_mut_append(from: NonNull<String>, to: &mut String) {
    assert_ne!(from.as_ptr(), to as *mut _);
    *to += from.as_ref();
    Rc::decrement_strong_count(from.as_ptr());
}

pub unsafe extern "C" fn value_into_evaluated(a: Value) -> Value {
    a.into_evaluated()
}

pub unsafe extern "C" fn value_ref(a: ManuallyDrop<Value>) -> Value {
    a.deref().clone()
}

pub unsafe extern "C" fn value_string_to_mut(a: Value) -> Value {
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

pub unsafe extern "C" fn value_string_to_path(a: Value) -> Value {
    match a.into_unpacked() {
        UnpackedValue::String(x) => UnpackedValue::new_path(PathBuf::from(Rc::unwrap_or_clone(x))),
        UnpackedValue::Lazy(x) => return value_string_to_path(x.evaluate().clone()),
        other => other,
    }
    .pack()
}

pub unsafe extern "C" fn asm_panic(msg: *const i8) {
    panic!("[JITPANIC] {}", CStr::from_ptr(msg).to_string_lossy());
}

pub unsafe extern "C-unwind" fn asm_panic_with_value(msg: *const i8, value: Value) {
    println!("[JITPANIC] {}", CStr::from_ptr(msg).to_string_lossy());
    println!(
        "[JITPANIC] Value passed to asm_panic_with_value: {} {value:?}",
        value.0
    );
    panic!("jitpanic")
}
