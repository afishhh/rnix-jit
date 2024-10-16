use std::{
    collections::{BTreeMap, HashMap, HashSet},
    env::VarError,
    fmt::Write,
    path::{Path, PathBuf},
    rc::Rc,
};

use regex::Regex;

use crate::{
    catch_nix_unwind,
    perfstats::measure_parsing_time,
    utils::{
        self,
        json::{JsonPrinter, JsonValue},
    },
    value, LazyValue, ValueUnpackable,
};
use crate::{throw, IRCompiler, NixException, Scope, UnpackedValue, Value, ValueKind, ValueMap};

fn human_valuekind(kind: ValueKind) -> &'static str {
    match kind {
        ValueKind::Attrset => "an attribute set",
        ValueKind::Function => "a function",
        ValueKind::Integer => "an integer",
        ValueKind::Double => "a floating point number",
        ValueKind::String => "a string",
        ValueKind::Path => "a path",
        ValueKind::List => "a list",
        ValueKind::Bool => "a boolean",
        ValueKind::Lazy => unreachable!(),
        ValueKind::Null => "null",
    }
}

const fn english_ordinal_number_suffix(n: u64) -> &'static str {
    match n {
        0 => "th",
        1 => "st",
        2 => "nd",
        3.. => "th",
    }
}

macro_rules! extract_typed {
    (Attrset($($tt: tt)*); $($fmt: tt)*) => {
        unsafe { &*extract_typed!(@impl Attrset($($tt)*); $($fmt)*).get() }
    };
    (List($($tt: tt)*); $($fmt: tt)*) => {
        unsafe { &*extract_typed!(@impl List($($tt)*); $($fmt)*).get() }
    };
    (PathOrAbsoluteString($how: tt $value: expr); $fmtstr: expr $(, $($fmtrest: tt)*)?) => {{
        let value = $value.evaluate();
        match value.kind() {
            ValueKind::Path => value.unpack_typed_ref_or_throw::<PathBuf>().as_path(),
            ValueKind::String => {
                let string = value.unpack_typed_ref_or_throw::<String>();
                let path = Path::new(string);
                if !path.is_absolute() {
                    throw!("string {string:?} doesn't represent an absolute path");
                }
                path
            },
            other => throw!(
                $fmtstr,
                human_valuekind(other),
                "a path or a string representing an absolute path"
                $(, $($fmtrest)*)?
            )
        }
    }};
    ($what: ident($($tt: tt)*); $($fmt: tt)*) => {
        extract_typed!(@impl $what($($tt)*); $($fmt)*)
    };
    (@impl UnpackedValue($($value: tt)*); $fmtstr: expr $(, $($fmtrest: tt)*)?) => {
        extract_typed!(@evaluate_value $($value)*).unpack()
    };
    (@impl $what: ident($($value: tt)*); $fmtstr: expr $(, $($fmtrest: tt)*)?) => {
        match extract_typed!(@evaluate_value $($value)*).unpack() {
            UnpackedValue::$what(ptr) => ptr,
            other => throw!(
                $fmtstr,
                human_valuekind(other.kind()),
                human_valuekind(ValueKind::$what)
                $(, $($fmtrest)*)?
            ),
        }
    };
    (@evaluate_value move $value: expr) => {
        $value.into_evaluated()
    };
    (@evaluate_value mut $value: expr) => {
        $value.evaluate_mut()
    };
    (@evaluate_value ref $value: expr) => {
        $value.evaluate()
    };
}

macro_rules! make_builtin {
    (
        fn $name: ident($($aname: ident: $ty: ident),+) $body: block
    ) => {
        make_builtin!(@impl $name; ($($aname: $ty),*); $body)
    };
    (@impl $name: expr; ($($aname: ident: $ty: ident),+) $body: block) => {{
        make_builtin!(@rec []; $body; $name; { $($aname: $ty)* }; $($aname: $ty)*)
    }};
    (@rec $movethese: tt; $body: expr; $fnname: expr; { $($args: tt)* };) => {{
        make_builtin!(@cast_args $fnname; 1; $($args)*);
        Into::into($body)
    }};
    (@cast_args $fnname: expr; $n: expr;) => { };
    (@cast_arg $fnname: expr; $n: expr; $name: ident: Value; $how: ident) => {};
    (@cast_arg $fnname: expr; $n: expr; $name: ident: $ty: ident; $how: ident) => {
        let $name = extract_typed!(
            $ty($how $name);
            concat!(
                "expected {2}{3} argument to builtin \"", $fnname, "\" to be {1} but it evaluated to {0}",
            ),
            const { $n },
            english_ordinal_number_suffix($n)
        );
    };
    (@cast_args $fnname: expr; $n: expr; $name: ident: $ty: ident) => {
        make_builtin!(@cast_arg $fnname; $n; $name: $ty; move)
    };
    (@cast_args $fnname: expr; $n: expr; $name: ident: $ty: ident $($rest: tt)*) => {
        make_builtin!(@cast_arg $fnname; $n; $name: $ty; ref);
        make_builtin!(@cast_args $fnname; $n + 1; $($rest)*);
    };
    (@moveprevious $next: ident $($move: ident)+) => {
        let $next = $next.clone();
        make_builtin!(@moveprevious $($move)*);
    };
    (@moveprevious $last: ident) => { /* ignore, doesn't have to be cloned yet */ };
    (@moveprevious) => { };
    (@rec [$($move: ident),*]; $body: expr; $fnname: expr; $args: tt; $name: ident: $ty: ident $($rest: tt)*) => {{
        make_builtin!(@moveprevious $($move)*);
        UnpackedValue::new_function(move |$name|
            make_builtin!(@rec [ $($move,)* $name ]; $body; $fnname; $args; $($rest)*)
        ).pack()
    }};
}

pub fn create_root_scope() -> *mut Scope {
    // top-level scope
    let mut values = ValueMap::new();
    // top-level builtins attrset
    let mut builtins = ValueMap::new();

    macro_rules! call_value {
        ($builtin_name: literal, $value: expr $(, $args: expr)+) => {{
            let mut current = $value;
            // FIXME: This error message is terrible (which argument?)
            $(
                let fun = extract_typed!(
                    Function(move current);
                    concat!("builtin \"", $builtin_name, "\" expected {} argument but got {} instead")
                );
                current = fun.run($args);
            )+
            current
        }};
    }

    macro_rules! lazy_call_value {
        ($builtin_name: literal, $value: expr $(, $args: expr)+) => {{
            let value = $value;
            LazyValue::from_closure(move || {
                call_value!($builtin_name, value, $($args),+)
            }).into()
        }};
    }

    macro_rules! make_builtins {
        (
            $(
                $(#[$($attr: tt)*])*
                fn $name: tt($($params: tt)*) $block: block
            )*
        ) => {
            $(make_builtins!(@impl $(#[$($attr)*])*; make_builtins!(@findname $name; $(#[$($attr)*])*); $block; $($params)*);)*
        };
        (@impl $(#[$($attr: tt)*])*; $name: expr; $block: block; $($params: tt)*) => {
            let builtin = make_builtin!(@impl $name; ($($params)*) $block);
            $(make_builtins!(@applyattr $($attr)*; $name; builtin);)*
            builtins.insert($name.to_string(), builtin);
        };
        (@applyattr toplevel_alias; $name: expr; $value: ident) => {
            values.insert($name.to_string(), $value.clone());
        };
        (@applyattr rename = $newname: literal; $name: expr; $value: ident) => {};
        (@findname $fnname: ident; #[rename = $name: literal] $(#[$($rest: tt)*])*) => {
            $name
        };
        (@findname $fnname: ident; #[$attr: meta] $(#[$($rest: tt)*])*) => {
            make_builtins!(@findname $fnname; $(#[$($rest)*])*)
        };
        (@findname $fnname: ident;) => {
            stringify!($fnname)
        };
    }

    make_builtins!(
        #[toplevel_alias]
        fn import(path: Path) {
            import(Rc::unwrap_or_clone(path))
        }

        // TODO: Why is this also in the top-level scope? Backwards compatibility?
        #[toplevel_alias]
        fn map(mapper: Value, list: List) {
            list.iter()
                .map(|x| -> Value {
                    let mapper = mapper.clone();
                    lazy_call_value!("map", mapper.clone(), x.clone())
                })
                .collect::<Value>()
        }

        #[toplevel_alias]
        fn toString(value: UnpackedValue) {
            match value {
                UnpackedValue::Integer(i) => i.to_string().into(),
                UnpackedValue::Double(d) => d.to_string().into(),
                UnpackedValue::Bool(v) => if v { "1" } else { "0" }.to_string().into(),
                UnpackedValue::List(_) => todo!(),
                UnpackedValue::Attrset(_) => todo!(),
                UnpackedValue::String(_) => value,
                UnpackedValue::Function(_) => todo!(),
                UnpackedValue::Path(p) => p.display().to_string().into(),
                UnpackedValue::Lazy(_) => unreachable!(),
                UnpackedValue::Null => String::new().into(),
            }
        }

        fn trace(message: String, ret: Value) {
            eprintln!("trace: {message}");
            ret
        }

        #[toplevel_alias]
        fn __dbg(message: Value, ret: Value) {
            eprintln!("dbg: {message:?}");
            ret
        }

        #[toplevel_alias]
        fn __dbgVal(message: Value) {
            seq(&message, true);
            eprintln!("dbg: {message:?}");
            message
        }

        #[toplevel_alias]
        fn throw(message: String) {
            #[allow(unreachable_code)] // From<!> is not implemented for Value
            {
                NixException::boxed(Rc::unwrap_or_clone(message)).raise() as Value
            }
        }

        #[toplevel_alias]
        fn abort(message: String) {
            #[allow(unreachable_code)] // From<!> is not implemented for Value
            {
                NixException::boxed(Rc::unwrap_or_clone(message)).raise() as Value
            }
        }

        fn filter(filter: Value, list: List) {
            UnpackedValue::new_list(
                list.iter()
                    .filter(|x| {
                        extract_typed!(
                            Bool(move call_value!("filter", filter.clone(), (*x).clone()));
                            "filter function passed to builtins.filter returned {1} instead of {0}"
                        )
                    })
                    .cloned()
                    .collect(),
            )
        }

        fn seq(seqd: Value, ret: Value) {
            seq(&seqd, false);
            ret
        }

        fn deepSeq(seqd: Value, ret: Value) {
            seq(&seqd, true);
            ret
        }

        fn tryEval(seqd: Value) {
            let success = try_eval(&seqd);
            let mut result = ValueMap::new();
            result.insert("success".to_string(), true.into());
            if success {
                result.insert("value".to_string(), seqd.clone());
            }
            UnpackedValue::new_attrset(result)
        }

        // ---- list-related builtins ----
        fn elem(needle: Value, list: List) {
            for value in list.iter() {
                if value.evaluate().equal(&needle).is_true() {
                    return Value::TRUE;
                }
            }
            Value::FALSE
        }

        fn head(list: List) {
            list[0].clone()
        }

        fn tail(list: List) {
            list.last()
                .unwrap_or_else(|| throw!("\"tail\" called on an empty list"))
                .clone()
        }

        fn elemAt(list: List, n: Integer) {
            n.try_into()
                .ok()
                .and_then(|i: usize| list.get(i))
                .unwrap_or_else(|| throw!("list index {n} is out of bounds"))
                .clone()
        }

        fn all(predicate: Value, list: List) {
            list.iter().all(|element| {
                extract_typed!(
                    Bool(move call_value!("all", predicate.clone(), element.clone()));
                    "predicate passed to builtins.all returned {} while {} was expected"
                )
            })
        }

        fn any(predicate: Value, list: List) {
            list.iter().any(|element| {
                extract_typed!(
                    Bool(move call_value!("any", predicate.clone(), element.clone()));
                    "predicate passed to builtins.any returned {} while {} was expected"
                )
            })
        }

        fn length(list: List) {
            list.len() as i32
        }

        #[rename = "foldl'"]
        fn foldl(op: Value, nul: Value, list: List) {
            let mut acc = nul.clone();
            for value in list.iter() {
                acc = call_value!("foldl'", op.clone(), acc, value.clone());
            }
            acc
        }

        fn partition(predicate: Value, list: List) {
            let (right, wrong) = list.iter().cloned().partition(|value| {
                extract_typed!(
                Bool(move call_value!("partition", predicate.clone(), value.clone()));
                "predicate passed to builtins.foldl' returned {} while {} was expected"
                            )
            });
            UnpackedValue::new_attrset({
                let mut result = ValueMap::new();
                result.insert("right".to_string(), UnpackedValue::new_list(right).pack());
                result.insert("wrong".to_string(), UnpackedValue::new_list(wrong).pack());
                result
            })
        }

        fn sort(_comparator: Value, _list: List) {
            #[allow(unreachable_code)] // From<!> is not implemented for Value
            {
                todo!("builtins.sort") as Value
            }
        }

        fn concatLists(lists: List) {
            lists
                .iter()
                .flat_map(|list| {
                    extract_typed!(
                        List(ref list);
                        "found {} in list passed to builtins.concatLists while {} was expected"
                    )
                    .iter()
                })
                .cloned()
                .collect::<Value>()
        }

        fn concatMap(mapper: Value, list: List) {
            list.iter()
                .cloned()
                .flat_map(|value| {
                    let mapped: Value = lazy_call_value!("concatMap", mapper.clone(), value);
                    extract_typed!(
                        List(move mapped);
                        "found {} in list passed to builtins.concatMap while {} was expected"
                    )
                    .iter()
                })
                .cloned()
                .collect::<Value>()
        }

        fn groupBy(f: Value, list: List) {
            let mut result = BTreeMap::new();
            for value in list.iter() {
                let key = extract_typed!(
                    String(move call_value!("groupBy", f.clone(), value.clone()));
                "function passed to builtins.groupBy returned {} while {} was expected"
                );
                result
                    .entry(Rc::unwrap_or_clone(key))
                    .or_insert_with(Vec::new)
                    .push(value.clone());
            }
            UnpackedValue::new_attrset(
                result
                    .into_iter()
                    .map(|(k, v)| (k, UnpackedValue::new_list(v).pack()))
                    .collect(),
            )
        }

        fn genList(generator: Value, n: Integer) {
            let mut result = vec![];
            for i in 0..n {
                result.push(lazy_call_value!("genList", generator.clone(), i.into()))
            }
            UnpackedValue::new_list(result)
        }

        // ---- string-related builtins ----
        fn stringLength(string: String) {
            string.len() as i32
        }

        fn substring(start: Integer, len: Integer, string: String) {
            let start: usize = start
                .try_into()
                .unwrap_or_else(|_| throw!("start position in builtins.substring out of range"));
            UnpackedValue::new_string(if start >= string.len() {
                String::new()
            } else if len == -1 {
                string[start..].to_string()
            } else {
                string[start..std::cmp::min(start + len as usize, string.len())].to_string()
            })
        }

        fn replaceStrings(from: List, to: List, s: String) {
            let mut result = Rc::unwrap_or_clone(s);

            for (i, pattern) in from.iter().enumerate() {
                let pattern = extract_typed!(
                    String(ref pattern);
                    "element of \"from\" list in builtins.replaceStrings is {} but {} was expected"
                );
                // FIXME: make this actually efficient
                if result.contains(pattern.as_str()) {
                    result = result.replace(
                        pattern.as_str(),
                        extract_typed!(
                            String(ref to[i]);
                            "element of \"to\" list in builtins.replaceStrings is {} but {} was expected"
                        ).as_str(),
                    );
                }
            }

            result
        }

        // FIXME: Theoretically one can compile the regex just after receiving the first argument
        //        but that would make the potential type error appear too early (not lazy enough).
        #[rename = "match"]
        fn match_(regex: String, haystack: String) {
            if let Some(captures) = Regex::new(&regex)
                .unwrap_or_else(|e| throw!("failed to compile regex passed to builtins.match: {e}"))
                .captures(haystack.as_str())
            {
                captures
                    .iter()
                    .skip(1)
                    .map(|c| {
                        c.map(|m| m.as_str().to_string().into())
                            .unwrap_or(Value::NULL)
                    })
                    .collect::<Value>()
            } else {
                Value::NULL
            }
        }

        fn split(regex: String, haystack: String) {
            let mut result = vec![];
            let mut last = 0;
            for captures in Regex::new(&regex)
                .unwrap_or_else(|e| throw!("failed to compile regex passed to builtins.match: {e}"))
                .captures_iter(&haystack)
            {
                let whole = captures.get(0).unwrap();
                result.push(haystack[last..whole.start()].to_string().into());
                result.push(
                    captures
                        .iter()
                        .skip(1)
                        .flatten()
                        .map(|x| {
                            UnpackedValue::new_string(haystack[x.start()..x.end()].to_string())
                                .pack()
                        })
                        .collect::<Value>(),
                );
                last = whole.end();
            }
            if last != haystack.len() {
                result.push(haystack[last..].to_string().into())
            };
            UnpackedValue::new_list(result)
        }

        // TODO: Once iter_intersperse is stable (never) the LOC of this function can be halved
        fn concatStringsSep(sep: String, strings: List) {
            let mut result = String::new();
            let mut it = strings.iter();
            if let Some(first) = it.next() {
                result += extract_typed!(
                    String(ref first);
                    "list passed to builtins.concatStringsSep contained {} but {} was expected"
                )
                .as_str();
            }
            for string in it {
                result += sep.as_str();
                result += extract_typed!(
                    String(ref string);
                    "list passed to builtins.concatStringsSep contained {} but {} was expected"
                )
                .as_str();
            }
            result
        }

        fn hasContext(_string: String) {
            Value::FALSE
        }

        fn unsafeDiscardStringContext(value: Value) {
            // TODO: String contexts
            value
        }

        fn addErrorContext(_context: String, value: Value) {
            eprintln!("WARNING: addErrorContext is currently a no-op");
            value
        }

        fn getEnv(name: String) {
            UnpackedValue::new_string(match std::env::var(name.as_str()) {
                Ok(value) => value,
                Err(VarError::NotPresent) => String::new(),
                Err(VarError::NotUnicode(_)) => {
                    throw!("getEnv: variable did not contain valid unicode")
                }
            })
        }

        // ---- path-related builtins ----
        fn pathExists(path: PathOrAbsoluteString) {
            path.try_exists().unwrap_or_else(|e| throw!("{e}"))
        }

        fn readFile(path: PathOrAbsoluteString) {
            std::fs::read_to_string(path).unwrap_or_else(|e| throw!("{e}"))
        }

        fn readFileType(path: PathOrAbsoluteString) {
            let file_type = path
                .metadata()
                .unwrap_or_else(|e| throw!("{e}"))
                .file_type();
            if file_type.is_dir() {
                "directory"
            } else if file_type.is_file() {
                "regular"
            } else if file_type.is_symlink() {
                "symlink"
            } else {
                "unknown"
            }
            .to_string()
        }

        // ---- attrset-related builtins ----
        fn mapAttrs(mapper: Value, attrs: Attrset) {
            let mut result = ValueMap::new();
            for (key, value) in attrs {
                let mapper = mapper.clone();
                result.insert(
                    key.to_string(),
                    lazy_call_value!(
                        "mapAttrs",
                        mapper.clone(),
                        UnpackedValue::new_string(key.clone()).pack(),
                        value.clone()
                    ),
                );
            }
            UnpackedValue::new_attrset(result)
        }

        fn listToAttrs(list: List) {
            list.iter()
                                .map(|item| {
                                    let attrset = extract_typed!(Attrset(ref item);
                                "element of list passed to builtins.listToAttrs is {} while {} was expected");
                                    let name = extract_typed!(
                                        String(ref attrset.get("name").unwrap());
                        "element of list passed to builtins.listToAttrs \
                        has name attribute which is {} but {} was expected"
                                    );
                                    let value = attrset.get("value").unwrap();
                                    (Rc::unwrap_or_clone(name), value.clone())
                                })
                                .collect::<Value>()
        }

        fn getAttr(name: String, attrset: Attrset) {
            attrset
                .get(name.as_str())
                .unwrap_or_else(|| throw!("attribute {name:?} missing"))
                .clone()
        }

        fn hasAttr(name: String, attrset: Attrset) {
            attrset.contains_key(name.as_str())
        }

        fn intersectAttrs(e1: Attrset, e2: Attrset) {
            UnpackedValue::new_attrset(if e2.len() > e1.len() {
                e1.keys()
                    .filter_map(|key| e1.get(key).map(|value| (key.clone(), value.clone())))
                    .collect()
            } else {
                e2.iter()
                    .filter(|&(key, _)| e1.contains_key(key))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            })
        }

        #[toplevel_alias]
        fn removeAttrs(attrs: Attrset, names: List) {
            let names = names
                                .iter()
                                .map(|value| extract_typed!(String(ref value); "element of list passed to builtins.removeAttrs is {} but {} was expected"))
                                .collect::<HashSet<_>>();
            UnpackedValue::new_attrset(
                attrs
                    .iter()
                    .filter(|(key, _)| names.contains(*key))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            )
        }

        fn zipAttrsWith(mapper: Value, sets: List) {
            sets
                .iter()
                .fold(HashMap::<String, Vec<Value>>::new(), |mut acc, value| {
                    for (key, value) in
                        extract_typed!(Attrset(ref value); "element of list passed to builtins.zipAttrsWith is {} but {} was expected")
                    {
                        acc.entry(key.to_string())
                            .or_default()
                            .push(value.clone())
                    }
                    acc
                })
                .into_iter()
                .map(|(key, values)| {
                    (
                        key.clone(),
                        call_value!(
                            "zipAttrsWith",
                            mapper.clone(),
                            UnpackedValue::new_string(key).pack(),
                            UnpackedValue::new_list(values).pack()
                        ),
                    )
                })
                .collect::<Value>()
        }

        fn attrNames(attrset: Attrset) {
            attrset
                .keys()
                .cloned()
                .map(UnpackedValue::new_string)
                .map(UnpackedValue::pack)
                .collect::<Value>()
        }

        fn attrValues(attrset: Attrset) {
            attrset.values().cloned().collect::<Value>()
        }

        fn catAttrs(name: String, list: List) {
            let mut result = vec![];
            for value in list.iter() {
                let attrset = extract_typed!(
                    Attrset(ref value);
                    "element of list passed to builtins.catAttrs is {} but {} was expected"
                );
                if let Some(value) = attrset.get(name.as_str()) {
                    result.push(value.clone());
                }
            }
            UnpackedValue::new_list(result)
        }

        fn genericClosure(_args: Attrset) {
            #[allow(unreachable_code)] // From<!> is not implemented for Value
            {
                throw!("builtins.genericClosure is not yet implemented") as Value
            }
        }

        fn splitVersion(a: String) {
            utils::VersionComponentIterator::new(a.chars())
                .map(|x| UnpackedValue::new_string(x.to_string()).pack())
                .collect::<Value>()
        }

        fn compareVersions(a: String, b: String) {
            match utils::version_compare(&a, &b) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            }
        }

        fn typeOf(value: Value) {
            match value.into_evaluated().kind() {
                ValueKind::Double => "float",
                ValueKind::Attrset => "set",
                ValueKind::Function => "lambda",
                ValueKind::String => "string",
                ValueKind::Path => "path",
                ValueKind::List => "list",
                ValueKind::Integer => "int",
                ValueKind::Bool => "bool",
                ValueKind::Lazy => unreachable!(),
                ValueKind::Null => "null",
            }
        }

        fn fromJSON(x: String) {
            from_json_str(&x)
        }

        fn toJSON(x: Value) {
            to_json(&x)
        }

        fn add(lhs: Value, rhs: Value) {
            Value::add(&lhs, &rhs)
        }

        fn sub(lhs: Value, rhs: Value) {
            Value::sub(&lhs, &rhs)
        }

        fn mul(lhs: Value, rhs: Value) {
            Value::mul(&lhs, &rhs)
        }

        fn div(lhs: Value, rhs: Value) {
            Value::div(&lhs, &rhs)
        }

        fn lessThan(lhs: Value, rhs: Value) {
            Value::less(&lhs, &rhs)
        }

        fn bitAnd(lhs: Value, rhs: Value) {
            Value::bit_and(&lhs, &rhs)
        }

        fn bitOr(lhs: Value, rhs: Value) {
            Value::bit_or(&lhs, &rhs)
        }

        fn bitXor(lhs: Value, rhs: Value) {
            Value::bit_xor(&lhs, &rhs)
        }
    );

    macro_rules! insert_is_builtin {
        ($name: literal, $type: ident) => {
            builtins.insert(
                $name.to_string(),
                UnpackedValue::new_function(|value| {
                    UnpackedValue::Bool(value.into_evaluated().is(ValueKind::$type)).pack()
                })
                .pack(),
            );
        };
    }

    insert_is_builtin!("isPath", Path);
    insert_is_builtin!("isString", String);
    insert_is_builtin!("isInt", Integer);
    insert_is_builtin!("isFloat", Double);
    insert_is_builtin!("isFunction", Function);
    insert_is_builtin!("isBool", Bool);
    insert_is_builtin!("isAttrs", Attrset);
    insert_is_builtin!("isList", List);

    builtins.insert(
        "currentSystem".to_string(),
        UnpackedValue::new_string("x86_64-linux".to_string()).pack(),
    );

    builtins.insert("storeDir".to_string(), "/nix/store".into());
    // The ridiciculous version number is to get around version checks
    builtins.insert("nixVersion".to_string(), "10.0-rnix-jit".into());

    values.insert("true".to_string(), Value::TRUE);
    values.insert("false".to_string(), Value::FALSE);
    values.insert("null".to_string(), UnpackedValue::Null.pack());

    // TODO: merge this with Value::operator methods
    macro_rules! binary_operator {
        ($name: literal $(, int => $op_int: tt)? $(, bool => $op_bool: tt)?) => {
            builtins.insert(
                $name.to_string(),
                UnpackedValue::new_function(|a| {
                    UnpackedValue::new_function(move |b| match (a.unpack(), b.into_unpacked()) {
                        $(
                            (UnpackedValue::Integer(a), UnpackedValue::Integer(b)) => {
                                (a $op_int b).into()
                            }
                        )?
                        $(
                            (UnpackedValue::Bool(a), UnpackedValue::Bool(b)) => {
                                (a $op_bool b).into()
                            }
                        )?
                        (a, b) => throw!(concat!("builtins.", $name, " not supported between values of type {} and {}"), a.kind(), b.kind())
                    }).pack()
                })
                .pack(),
            );
        }
    }

    binary_operator!("add", int => +);
    binary_operator!("sub", int => -);
    binary_operator!("mul", int => *);
    binary_operator!("div", int => /);
    binary_operator!("lessThan", int => <);
    binary_operator!("bitAnd", int => &);
    binary_operator!("bitOr", int => |);
    binary_operator!("bitXor", int => ^);

    values.insert(
        "builtins".to_string(),
        UnpackedValue::new_attrset(builtins).pack(),
    );

    Scope::from_map(values, std::ptr::null_mut())
}

pub fn seq_impl(value: &Value, deep: bool, visited: &mut HashSet<*const ()>) {
    match value.unpack() {
        UnpackedValue::Integer(_) => (),
        UnpackedValue::Double(_) => (),
        UnpackedValue::Bool(_) => (),
        UnpackedValue::List(value) => {
            let list = unsafe { &mut *value.get() };
            if !visited.insert(list as *const _ as *const ()) {
                return;
            }
            for value in list.iter_mut() {
                let value = value.evaluate_mut();
                if deep {
                    seq_impl(value, deep, visited);
                }
            }
        }
        UnpackedValue::Attrset(value) => {
            let map = unsafe { &mut *value.get() };
            if !visited.insert(map as *const _ as *const ()) {
                return;
            }
            for (_, value) in map.iter_mut() {
                let value = value.evaluate_mut();
                if deep {
                    seq_impl(value, deep, visited);
                }
            }
        }
        UnpackedValue::String(_) => (),
        UnpackedValue::Path(_) => (),
        UnpackedValue::Function(_) => (),
        UnpackedValue::Lazy(value) => seq_impl(value.evaluate(), deep, visited),
        UnpackedValue::Null => (),
    }
}

pub fn seq(value: &Value, deep: bool) {
    let mut visited = HashSet::new();
    seq_impl(value, deep, &mut visited)
}

// TODO: this catches wayyyyy too many errors
//       add more metadata to NixException
fn try_eval(value: &Value) -> bool {
    catch_nix_unwind(move || {
        seq(value, false);
    })
    .is_ok()
}

fn from_json(text: JsonValue) -> Value {
    match text {
        JsonValue::Object(x) => x.into_iter().map(|(k, v)| (k, from_json(v))).collect(),
        JsonValue::List(x) => x.into_iter().map(from_json).collect(),
        JsonValue::String(x) => UnpackedValue::new_string(x).pack(),
        JsonValue::Integer(i) => UnpackedValue::Integer(
            i.try_into()
                .unwrap_or_else(|_| throw!("json integer out of range")),
        )
        .pack(),
        JsonValue::Float(f) => UnpackedValue::Double(f).pack(),
    }
}

fn from_json_str(text: &str) -> Value {
    from_json(crate::utils::json::parse(text, false))
}

fn to_json_impl(value: &Value, output: &mut JsonPrinter) {
    match value.evaluate().unpack() {
        UnpackedValue::Integer(i) => output.integer(i.into()),
        UnpackedValue::Double(f) => output.float(f),
        UnpackedValue::Bool(b) => output.bool(b),
        UnpackedValue::List(l) => {
            output.start_list();
            for value in unsafe { &*l.get() }.iter() {
                to_json_impl(value, output);
            }
            output.end_list()
        }
        UnpackedValue::Attrset(s) => {
            output.start_object();
            for (key, value) in unsafe { &*s.get() }.iter() {
                output.key(key);
                to_json_impl(value, output);
            }
            output.end_object()
        }
        UnpackedValue::String(s) => output.string(&s),
        UnpackedValue::Function(_) => throw!("cannot convert a function to json"),
        UnpackedValue::Path(_) => throw!("TODO: cannot convert a path to json"),
        UnpackedValue::Lazy(_) => unreachable!(),
        UnpackedValue::Null => output.null(),
    };
}

fn to_json(value: &Value) -> String {
    let mut printer = JsonPrinter::new();

    to_json_impl(value, &mut printer);

    printer.finish()
}

pub(crate) fn eval_throwing(root: PathBuf, filename: String, code: String) -> Value {
    let parse = {
        let _tc = measure_parsing_time();
        rnix::Root::parse(&code)
    };
    if !parse.errors().is_empty() {
        let mut msg = "parse errors encountered:\n".to_string();
        for (i, error) in parse.errors().iter().enumerate() {
            writeln!(msg, "{}. {}", i + 1, error).unwrap();
        }
        NixException::boxed(msg).raise()
    }
    let expr = parse.tree().expr().unwrap();
    let program = IRCompiler::compile(root, filename, code, expr);

    let mode = std::env::var("RNIX_JIT_MODE").unwrap_or_default();
    let threshold = match mode.as_str() {
        "INTERPRET" => usize::MAX,
        "HYBRID" => 10,
        h if h.starts_with("HYBRID=") => h.strip_prefix("HYBRID=").unwrap().parse().unwrap_or(10),
        "COMPILE" => 0,
        _ => 10,
    };

    let runnable = crate::interpreter::into_dynamically_compiled(program, None, threshold);
    ROOT_SCOPE.with(move |s| unsafe { runnable.run(*s, Value::NULL) })
}

pub fn import(path: PathBuf) -> Value {
    let (path, expr) = match std::fs::read_to_string(&path) {
        Err(e) if e.to_string().contains("Is a directory") => {
            let default = path.join("default.nix");
            std::fs::read_to_string(&default).map(|s| (default, s))
        }
        other => other.map(|s| (path, s)),
    }
    .unwrap();

    eval_throwing(
        path.parent().unwrap().to_path_buf(),
        path.to_string_lossy().to_string(),
        expr,
    )
}

thread_local! {
    pub static ROOT_SCOPE: *mut Scope = create_root_scope();
}
