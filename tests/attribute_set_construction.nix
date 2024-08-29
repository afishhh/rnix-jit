{
  hello = {
    a.b.c = 1;
    a.b.d = 50;
    a.d.f = 20;
  };

  hello.a.b.f = 30;

  one.two = [ 1 2 ];

  "inherit" = rec {
    a = { c = 100; };
    inherit (a) c;
  };

  "hello ${toString 20}".aaa.${builtins.currentSystem}.aa = rec {
    value = 20;
  };

  "hello ${toString 40}" = { s = 2; }.s;

  rectest = rec {
    x = 1;
  };
  rectest.y = 2;
  rectest = {
    z = x + 2;
  };

  rectest2 = {
    x = 1;
  };
  rectest2.y = 2;
  rectest2 = {
    z = 3;
  };
}
