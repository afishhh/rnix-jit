{
	all1 = builtins.all (x: x > 5) [
		2
		6
	];
	all2 = builtins.all (x: x > 5) [
		9
		6
	];
	all3 = builtins.all (x: x > 5) [
		1
		3
	];
	any1 = builtins.any (x: x > 5) [
		2
		6
	];
	any2 = builtins.any (x: x > 5) [
		9
		6
	];
	any3 = builtins.any (x: x > 5) [
		1
		3
	];
	imply1 = true -> false;
	imply2 = false -> false;
	imply3 = false -> true;
	imply4 = true -> true;
	mapAttrs1 = builtins.mapAttrs (a: b: "${a} ${b}") {
		a = "b";
		b = "a";
		hi = "hello";
	};
}
