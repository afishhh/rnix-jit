let
  a = "hello";
  c = "x${a}";
  b = "${a}${a} world one${c}";
in
''${b} ${a}''
