rec {
  a = 1;
  b = a + 2;
  x = { y = 1; z = x.y + 2; };
  fix = f: let x = f x; in x;
  d = fix (self: { foo = "foo"; bar = "bar"; foobar = self.foo + self.bar; });
  e = (a: b: b) e 10;
}
