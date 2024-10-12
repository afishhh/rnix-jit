work in progress<br>
currently slower than nix<br>
lacks:
- string contexts
- 64 bit integers (NaN packing is in force)
- support for anything other than x86_64-linux (support for 32-bit architectures currently unplanned)
- memory management (most memory is leaked forever, nothing is freed during unwinding)
    - this is mostly due to the fact `Scope`s are not reference counted due to severe laziness
- some builtins
- speed (see line 2 of README.md)
- skill

will I finish this? great question

ideas:
- keep track which identifiers may refer to builtins and also jit builtins
- refactor intermediate representation:
    - add a locals table (nix variables cannot have dynamic names)
        - reuse locals for `parents` in map construction
    - linearize all operations by compiling into a byte buffer
        - store constant operands in-line
        - add jump instruction to actually make this possible
