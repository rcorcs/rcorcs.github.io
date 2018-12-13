
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


talk about hand-optimizing programs.

specialist programmer vs average programmer.

https://www.quora.com/Are-compilers-really-better-than-human-at-code-optimization


talk about inlining, optimization per compilation unit and how the inline keyword addresses this problem.

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
