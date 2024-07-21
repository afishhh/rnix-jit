let twenty = 20; in {
	hi = if twenty > 20 then 100 else 200;
	hi2 = if twenty == 20 then 100 else 200;
	is_twenty_bigger_than_20 = twenty > 20;
	is_twenty_smaller_than_or_equal_to_20 = twenty <= 20;
	is_twenty_equal_to_20 = twenty == 20;
	is_twenty_not_equal_to_20 = twenty != 20;
	aa = let inherit twenty; in twenty;
	functionsAreSupported = (hello: hello + 30) 50;
	stringsExist = "hi";
	xd = builtins.trace "There are builtins too" [null true false];
	false = true && false;
	mapped = builtins.map (x: [(x * 30) (x * 10)]) [
		1 2 3 4 5 6 7 8 9 10
	];
}
