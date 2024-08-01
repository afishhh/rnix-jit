let a = {
  b = {
    c = 10;
  };
  d = 5;
}; in
{
  hasC = a?b.c;
  hasD = a?d;
  hasE = a?d.e;
  hasF = a?f;
}
