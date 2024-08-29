{
  # TODO: Currently all JSON parsing allows traling commas, change this
  strict = builtins.fromJSON '' { "a": 2, "c": [1, "2", {}] } '';
  # sloppy = builtins.__fromJSONSloppy '' { "a": 2, "c": [1, "2", {},], } '';
}
