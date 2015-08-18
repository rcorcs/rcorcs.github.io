CFLAGS = -fopenmp -O3

all: fib sections tasks

fib: fib.c
	$(CXX) fib.c -o fib $(CFLAGS)

sections: sections.c
	$(CXX) sections.c -o sections $(CFLAGS)

tasks: tasks.c
	$(CXX) tasks.c -o tasks $(CFLAGS)

clean:
	rm -f fib pfib0 pfib1 pfib2 sections tasks
