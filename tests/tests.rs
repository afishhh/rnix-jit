macro_rules! compare_file_eval {
    ($name: ident, $path: literal, $result: tt) => {
        #[test]
        fn $name() {
            let value = rnix_jit::catch_nix_unwind(|| {
                let value = rnix_jit::import(std::path::PathBuf::from("./tests").join($path));
                rnix_jit::seq(&value, true);
                value
            })
            .unwrap();
            assert_eq!(value, rnix_jit::value!($result));
        }
    };
}

compare_file_eval!(test1, "test1.nix", {
    aa = 20;
    false = false;
    functionPatternParam = 62;
    functionsAreSupported = 80;
    hasAttr = [true, false, true];
    hi = 200;
    hi2 = 100;
    is_twenty_bigger_than_20 = false;
    is_twenty_equal_to_20 = true;
    is_twenty_not_equal_to_20 = false;
    is_twenty_smaller_than_or_equal_to_20 = true;
    lazyScopeInherit = 13;
    mapped = [
        [30, 10],
        [60, 20],
        [90, 30],
        [120, 40],
        [150, 50],
        [180, 60],
        [210, 70],
        [240, 80],
        [270, 90],
        [300, 100],
    ];
    setAttrpath = 1;
    some_inherits = {
        a = 1;
        abc = {
            a = 1;
            b = 3;
            c = 2;
        };
        b = 3;
        c = 2;
        twenty = 20;
    };
    stringsExist = "hi";
    xd = [null, true, false];
});

compare_file_eval!(test2, "test2.nix", {
    a = 1;
    b = 3;
    d = {
        bar = "bar";
        foo = "foo";
        foobar = "foobar";
    };
    e = 10;
    x = {
        y = 1;
        z = 3;
    };
});

compare_file_eval!(test3, "test3.nix", 21);
compare_file_eval!(test4, "test4.nix", {
    bar = "bar";
    foo = "foo";
    foobar = "foobar";
});
compare_file_eval!(test5, "test5.nix", {
    a = 4;
    b = 9;
    c = 7;
});

compare_file_eval!(test8, "test8.nix", "hellohello world onexhello hello");
compare_file_eval!(test10, "test10.nix", {
    hasC = true;
    hasD = true;
    hasE = false;
    hasF = false;
});

compare_file_eval!(test11, "test11.nix", {
    all1 = false;
    all2 = true;
    all3 = false;
    any1 = true;
    any2 = true;
    any3 = false;
    imply1 = false;
    imply2 = true;
    imply3 = true;
    imply4 = true;
    mapAttrs1 = {
        a = "a b";
        b = "b a";
        hi = "hi hello";
    };
});

compare_file_eval!(test12, "test12.nix", {
    genList = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14];
    invert1 = false;
    invert2 = true;
});

// This tests every current availble opcode for map construction
compare_file_eval!(
    attribute_set_construction,
    "attribute_set_construction.nix",
    {
        hello = {
            a = {
                b = {
                    c = 1;
                    d = 50;
                    f = 30;
                };
                d = {
                    f = 20;
                };
            };
        };
        ["hello 20"] = {
            aaa = {
                ["x86_64-linux"] = {
                    aa = {
                        value = 20;
                    };
                };
            };
        };
        ["hello 40"] = 2;
        ["inherit"] = {
            a = {
                c = 100;
            };
            c = 100;
        };
        one = {
            two = [1, 2];
        };
        rectest = {
            x = 1;
            y = 2;
            z = 3;
        };
        rectest2 = {
            x = 1;
            y = 2;
            z = 3;
        };
    }
);

compare_file_eval!(from_json, "fromJSON.nix", {
    strict = {
        a = 2;
        c = [1, "2", {}];
    };
});

compare_file_eval!(to_json, "toJSON.nix", {
    emptyObject = "{}";
    manyDiverseThings =
        "{\"a\":{},\"b\":{\"c\":2,\"d\":[\"hello\"]},\"c\":[\"a\",2,{\"value\":\"three\"}]}";
    manyEmptyThings = "{\"a\":[],\"b\":{},\"c\":[{},{},{},[],[],[]]}";
});
