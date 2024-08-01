let
  x = { a = 5; };
  b = 10;
  inherit (x) a;
  some_inherits = let abc = { a = 1; b = 3; c = 2; }; in {
    inherit abc;
    inherit (abc) a b c;
  };
in
x.a + b + a + some_inherits.a

