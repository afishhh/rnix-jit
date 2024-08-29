use std::{mem::MaybeUninit, path::PathBuf, rc::Rc};

use crate::{
    catch_nix_unwind, interpreter::into_dynamically_compiled, throw, Function, LazyValue, Program,
    Scope, UnpackedValue, Value, ValueKind, ValueList, ValueMap,
};

unsafe fn has_or_getattrpath(
    mut current: Value,
    attrpath: &mut [MaybeUninit<Value>],
    maybe_default_if_getattr: Option<Option<Value>>,
) -> Value {
    let mut it = attrpath.iter_mut().rev();
    for v in &mut it {
        let UnpackedValue::Attrset(attrset) = current.evaluate().unpack() else {
            it.for_each(|v| v.assume_init_drop());
            match maybe_default_if_getattr {
                Some(Some(x)) => return x,
                Some(None) => throw!("{current} is not an attribute set"),
                None => return Value::FALSE,
            }
        };
        let attrset = &*attrset.get();

        let UnpackedValue::String(a) = v.assume_init_read().into_evaluated().into_unpacked() else {
            throw!("Attribute path component is not a string")
        };
        current = match attrset.get(a.as_str()) {
            Some(x) => x.clone(),
            None => {
                it.for_each(|v| v.assume_init_drop());
                match maybe_default_if_getattr {
                    Some(Some(x)) => return x,
                    Some(None) => throw!("{current} does not have attribute {a}"),
                    None => return Value::FALSE,
                }
            }
        }
    }

    if maybe_default_if_getattr.is_some() {
        current
    } else {
        Value::TRUE
    }
}

pub fn interpret(mut scope: *mut Scope, program: &Program, compilation_threshold: usize) -> Value {
    let mut stack: Vec<Value> = vec![];
    let mut map_stack: Vec<(Vec<LazyValue>, *mut ValueMap)> = vec![];
    let mut source_span_stack = vec![];

    let result = catch_nix_unwind(|| {
        for operation in program.operations.iter() {
            macro_rules! value_eager_binop {
                ($op: ident) => {{
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(Value::$op(a.into_evaluated(), b.into_evaluated()));
                }};
            }

            macro_rules! value_lazy_boolop {
                ($rhs: expr, shortcircuit on $svalue: expr => $sresult: expr) => {{
                    let result = {
                        let lhs = stack.pop().unwrap().into_evaluated();
                        if lhs.as_raw() == Value::from_bool($svalue).into_raw() {
                            Value::from_bool($sresult)
                        } else if lhs.as_raw() == Value::from_bool(!$svalue).into_raw() {
                            let rhs = interpret(scope, $rhs, compilation_threshold);
                            if !rhs.is(ValueKind::Bool) {
                                throw!("second operand of boolean operator is not a boolean");
                            }
                            rhs
                        } else {
                            throw!("first operand of boolean operator is not a boolean")
                        }
                    };
                    stack.push(result)
                }};
            }

            macro_rules! make_lazy {
                ($program: expr) => {
                    LazyValue::from_runnable(
                        scope,
                        into_dynamically_compiled($program, None, compilation_threshold),
                    )
                };
            }

            match operation {
                crate::Operation::Push(x) => stack.push(x.clone()),
                crate::Operation::PushFunction(param, program) => {
                    stack.push(
                        UnpackedValue::Function(Rc::new(Function::new(
                            into_dynamically_compiled(
                                program.clone(),
                                Some(param.clone()),
                                compilation_threshold,
                            ),
                            scope,
                        )))
                        .pack(),
                    );
                }
                crate::Operation::MapBegin { parents, rec } => {
                    let map = Rc::into_raw(Rc::new(ValueMap::new())) as *mut _;
                    if *rec {
                        unsafe {
                            Rc::increment_strong_count(map);
                            scope = Scope::from_attrs(Rc::from_raw(map as *mut _), scope);
                        }
                    };
                    map_stack.push((
                        parents
                            .iter()
                            .map(|program| make_lazy!(program.clone()))
                            .collect(),
                        map,
                    ))
                }
                crate::Operation::ConstantAttrPop(name) => unsafe {
                    (*map_stack.last().unwrap().1).insert(name.clone(), stack.pop().unwrap());
                },
                crate::Operation::ConstantAttrLoad(name) => unsafe {
                    (*map_stack.last().unwrap().1).insert(name.clone(), Scope::lookup(scope, name));
                },
                crate::Operation::ConstantAttrLazy(name, value) => unsafe {
                    (*map_stack.last().unwrap().1)
                        .insert(name.clone(), make_lazy!(value.clone()).into());
                },
                crate::Operation::ConstantAttrInherit(name, parentidx) => unsafe {
                    let (parents, map) = map_stack.last().unwrap();
                    let parent = parents[*parentidx].clone();
                    let name = name.clone();
                    (**map).insert(
                        (*name).clone(),
                        LazyValue::from_closure(move || {
                            parent.evaluate().get_attribute(&name).clone()
                        })
                        .into(),
                    );
                },
                crate::Operation::DynamicAttrPop => unsafe {
                    let value = stack.pop().unwrap();
                    let key = stack
                        .pop()
                        .unwrap()
                        .into_evaluated()
                        .unpack_typed_or_throw::<String>();
                    (*map_stack.last().unwrap().1).insert((*key).clone(), value);
                },
                crate::Operation::DynamicAttrLazy(program) => unsafe {
                    let key = stack
                        .pop()
                        .unwrap()
                        .into_evaluated()
                        .unpack_typed_or_throw::<String>();
                    (*map_stack.last().unwrap().1)
                        .insert((*key).clone(), make_lazy!(program.clone()).into());
                },
                crate::Operation::PushAttrset { parents: _ } => stack.push(
                    UnpackedValue::Attrset(unsafe {
                        Rc::from_raw(map_stack.pop().unwrap().1 as *mut _)
                    })
                    .into(),
                ),
                crate::Operation::GetAttrConsume {
                    components,
                    default,
                } => unsafe {
                    stack.set_len(stack.len() - *components);
                    let current = stack.pop().unwrap();
                    let path = &mut stack.spare_capacity_mut()[1..=*components];
                    let result = has_or_getattrpath(
                        current,
                        path,
                        Some(default.clone().map(|x| make_lazy!(x).into())),
                    );
                    stack.push(result);
                },
                crate::Operation::HasAttrpath(elements) => unsafe {
                    stack.set_len(stack.len() - *elements);
                    let current = stack.pop().unwrap();
                    let path = &mut stack.spare_capacity_mut()[1..=*elements];
                    let result = has_or_getattrpath(current, path, None);
                    stack.push(result);
                },
                crate::Operation::PushList => {
                    stack.push(UnpackedValue::new_list(Vec::new()).pack())
                }
                crate::Operation::ListAppend(item) => {
                    let UnpackedValue::List(list) = stack.last().unwrap().unpack() else {
                        unreachable!()
                    };
                    unsafe { (*list.get()).push(make_lazy!(item.clone()).into()) }
                }
                crate::Operation::StringCloneToMut => {
                    let value = stack.pop();
                    stack.push(value.unwrap().make_mut_string());
                }
                crate::Operation::StringMutAppend => unsafe {
                    let s = stack
                        .pop()
                        .unwrap()
                        .into_evaluated()
                        .unpack_typed_or_throw::<String>();
                    let mut_str = stack.last().unwrap().unpack_typed_ref_or_throw::<String>()
                        as *const _ as *mut String;
                    // TODO: Make string construction use a side storage like attrset construction.
                    #[allow(invalid_reference_casting)]
                    {
                        *mut_str += s.as_str();
                    }
                },
                crate::Operation::StringToPath => {
                    // TODO: See above
                    let string =
                        Rc::unwrap_or_clone(stack.pop().unwrap().unpack_typed_or_throw::<String>());
                    stack.push(UnpackedValue::new_path(PathBuf::from(string)).pack())
                }
                crate::Operation::Concat => unsafe {
                    let rhs = stack
                        .pop()
                        .unwrap()
                        .into_evaluated()
                        .unpack_typed_or_throw::<ValueList>();
                    let lhs = stack
                        .pop()
                        .unwrap()
                        .into_evaluated()
                        .unpack_typed_or_throw::<ValueList>();
                    let rhs = &*rhs.get();
                    let lhs = &*lhs.get();
                    let mut result = Vec::with_capacity(lhs.len() + rhs.len());
                    result.extend(lhs.iter().cloned());
                    result.extend(rhs.iter().cloned());
                    stack.push(UnpackedValue::new_list(result).pack())
                },
                crate::Operation::Update => unsafe {
                    let upper = stack
                        .pop()
                        .unwrap()
                        .into_evaluated()
                        .unpack_typed_or_throw::<ValueMap>();
                    let lower = stack
                        .pop()
                        .unwrap()
                        .into_evaluated()
                        .unpack_typed_or_throw::<ValueMap>();
                    let mut result = (*lower.get()).clone();
                    for (key, value) in (*upper.get()).iter() {
                        result.insert(key.clone(), value.clone());
                    }
                    stack.push(UnpackedValue::new_attrset(result).pack())
                },
                crate::Operation::Add => value_eager_binop!(add),
                crate::Operation::Sub => value_eager_binop!(sub),
                crate::Operation::Mul => value_eager_binop!(mul),
                crate::Operation::Div => value_eager_binop!(div),
                crate::Operation::And(rhs) => {
                    value_lazy_boolop!(rhs, shortcircuit on false => false)
                }
                crate::Operation::Or(rhs) => value_lazy_boolop!(rhs, shortcircuit on true => true),
                crate::Operation::Implication(rhs) => {
                    value_lazy_boolop!(rhs, shortcircuit on false => true)
                }
                crate::Operation::Less => value_eager_binop!(less),
                crate::Operation::LessOrEqual => value_eager_binop!(less_or_equal),
                crate::Operation::MoreOrEqual => value_eager_binop!(greater_or_equal),
                crate::Operation::More => value_eager_binop!(greater),
                crate::Operation::Equal => value_eager_binop!(equal),
                crate::Operation::NotEqual => value_eager_binop!(not_equal),
                crate::Operation::Apply => {
                    let value = stack.pop().unwrap();
                    let argument = stack.pop().unwrap();
                    let function = value.into_evaluated().unpack_typed_or_throw::<Function>();
                    stack.push(function.run(argument));
                }
                crate::Operation::ScopeEnter { parents: _ } => unsafe {
                    scope = Scope::from_attrs(
                        Rc::from_raw(map_stack.pop().unwrap().1 as *mut _),
                        scope,
                    );
                },
                crate::Operation::ScopeWith(what) => {
                    let runnable =
                        into_dynamically_compiled(what.clone(), None, compilation_threshold);
                    scope = Scope::with_new_lazy_implicit(scope, move || {
                        unsafe { runnable.run(scope, Value::NULL).into_evaluated() }
                            .unpack_typed_or_throw::<ValueMap>()
                    });
                }
                crate::Operation::Load(x) => stack.push(unsafe { Scope::lookup(scope, x) }),
                crate::Operation::ScopeLeave => scope = unsafe { (*scope).previous },
                crate::Operation::IfElse(then, otherwise) => {
                    let value = stack.pop().unwrap().into_evaluated();
                    if value.is_true() {
                        stack.push(interpret(scope, then, compilation_threshold));
                    } else if value.is_false() {
                        stack.push(interpret(scope, otherwise, compilation_threshold));
                    } else {
                        throw!("if condition is not a boolean")
                    }
                }
                crate::Operation::Invert => {
                    let x = stack.pop().unwrap();
                    if x.is_true() {
                        stack.push(Value::FALSE);
                    } else if x.is_false() {
                        stack.push(Value::TRUE);
                    } else {
                        throw!("invert attempted on non-boolean value");
                    }
                }
                crate::Operation::Negate => {
                    let x = stack.pop().unwrap();
                    let UnpackedValue::Integer(mut i) = x.into_unpacked() else {
                        throw!("negate attempted on non-integer value")
                    };
                    // HACK: HACK: HACK: Does not need an explanation
                    if i == -2147483648 {
                        i = 2147483647;
                    }
                    stack.push((-i).into());
                }
                crate::Operation::SourceSpanPush(x) => source_span_stack.push(x),
                crate::Operation::SourceSpanPop => {
                    _ = source_span_stack.pop().unwrap();
                }
            }
        }
    });

    if let Err(mut exc) = result {
        if source_span_stack.is_empty() {
            eprintln!("warning: no source span for interpret invocation");
        }
        exc.stacktrace
            .extend(source_span_stack.into_iter().cloned());
        // FIXME: don't?
        Box::new(exc).raise()
    }

    stack.pop().unwrap()
}
