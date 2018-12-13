
# Static vs Inline 
## and why inline does not actually inline?


### TL;DR
#### Conceptually:
The inline keyword is a bad design choice, victim of technical limitations at the time.

#### Practice for C++:
* If a function is defined in a header file, use the "inline" keyword.
* If distinct functions must have the same name in different implementation files,
use the "static" keyword.

#### Practice for C:
* Don't use the ```inline``` keyword (or read the full blog post).
* If distinct functions must have the same name in different implementation files,
use the ```static``` keyword.
