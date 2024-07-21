{
	hi = 100;
	functionAreSupported = (hello: hello + 30) 50;
	stringsExist = "hi";
	xd = builtins.trace "There are builtins too" [null true false];
	false = true && false;
	mapped = builtins.map (x: [(x * 30) (x * 10)]) [ 1 2 3 4 5 6 7 8 9 10 11 ];
}
