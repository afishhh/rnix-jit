let
  fix = f: let x = f x; in x;
in
fix (self: { foo = "foo"; bar = "bar"; foobar = self.foo + self.bar; })
