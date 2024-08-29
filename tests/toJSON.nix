{
  emptyObject = builtins.toJSON {};
  manyEmptyThings = builtins.toJSON {
	a = [];
	b = {};
	c = [ {} {} {} [] [] [] ];
  };
  manyDiverseThings = builtins.toJSON {
	a = {};
	b = {
	  c = 2;
	  d = [ "hello" ];
	};
	c = [ "a" 2 { value = "three"; } ];
  };
}
