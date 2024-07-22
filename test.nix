let twenty = 20; in {
	# nixpkgsLib = import ./lib;
	hi = if twenty > 20 then 100 else 200;
	hi2 = if twenty == 20 then 100 else 200;
	is_twenty_bigger_than_20 = twenty > 20;
	is_twenty_smaller_than_or_equal_to_20 = twenty <= 20;
	is_twenty_equal_to_20 = twenty == 20;
	is_twenty_not_equal_to_20 = twenty != 20;
	aa = let inherit twenty; in twenty;
	functionsAreSupported = (hello: hello + 30) 50;
	functionPatternParam = ({ hello ? 4, next ? 2 }: hello + next) { hello = 60; };
	hasAttr = [
		({ h = 1; }?h)
		({ l = 1; }?h)
		({ l = 1; h = 2; }?h)
	];
	stringsExist = "hi";
	xd = builtins.trace "There are builtins too" [null true false];
	false = true && false;
	mapped = builtins.map (x: [(x * 30) (x * 10)]) [
		1 2 3 4 5 6 7 8 9 10
	];
	some_inherits = let abc = { a = 1; b = 3; c = 2; }; in {
		inherit twenty abc;
		inherit (abc) a b c;
	};
	lazyScopeInherit = let
		x = { a = z; b = 3; };
		z = 5;
		inherit (x) a c d;
	in a + x.b + z;
}
