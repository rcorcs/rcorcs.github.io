
# Static vs Inline 
## and why inline does not actually inline?


>---
> 
>### TL;DR
> 
>#### Quick Practical Tips:
>* Use a header file with declarations and have these functions defined only once in an  implementation file.
>* If distinct functions **must** have the same name in different implementation files, use the `static` keyword in each local definition.
>* Do not use the `inline` keyword.
>* Use link-time optimization (`-flto` flag).
> 
>---


  
Compilers normally operate on a single translation unit at
a time. Each translation unit, which includes a single source
file and its expanded headers, is compiled into a single native
object file, and ultimately the linker combines multiple object
files into a resulting binary or library. Optimizations are
optionally applied within each translation unit (or module)
during the compilation, at both the function scope and the
module scope. The latter is referred to as Inter-Procedural
Optimization (IPO). Function inlining [7][6] is a key IPO.
However, normally a callee can only be inlined into its caller
if they are defined in the same module.






talk about inlining, optimization per compilation unit and how the inline keyword addresses this problem.

talk about hand-optimizing programs.

specialist programmer vs average programmer.

https://www.quora.com/Are-compilers-really-better-than-human-at-code-optimization




then talk about link-time optimization.



practi

File: `user1.c`

```C
#include <stdio.h>

inline void foo() {
  printf("This is the foo function.\n");
}

int fooUser1() {
  foo();
  return 1;
}

```

File: `user2.c`
```C
#include <stdio.h>

inline void foo() {
  printf("This is the foo function.\n");
}

int fooUser2() {
  foo();
  foo();
  return 2;
}

```

File: `main.c`
```C
#include <stdio.h>

void foo() {
  printf("This is the foo function.\n");
}

int fooUser1();
int fooUser2();

int main() {
  fooUser1();
  fooUser2();
  return 0;
}

```
>#### Opinion:
>* The `static` keyword should have been called `internal`.
>* The `inline` keyword was a poorly designed feature with a bad name, though it was driven by practical limitations at the time.


## LLVM Internals


### Clang in C mode

`inline`

`inlinehint` and `available_externally` linkage.

`inline static`

`inlinehint` and `internal` linkage.

`static`

only `internal` linkage.


### Clang in C++ mode

`inline`

`inlinehint` and `linkonce_odr` linkage.

`static`

`inlinehint` and `internal` linkage.
