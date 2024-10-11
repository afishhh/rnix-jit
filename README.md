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
