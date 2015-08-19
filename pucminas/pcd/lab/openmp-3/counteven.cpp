#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

int main(int argc, char **argv){
    int n = atoi(argv[1]);
    int a[n];
    int even[n];
    int count = 0;

    srand(time(NULL));
    
    for(int i = 0; i<n; i++){
        a[i] = rand()%n;
    }
    
    for(int i = 0; i<n; i++){
        even[i] = (a[i]%2==0)?1:0;
    }

    for(int i = 0; i<n; i++){
        count += even[i];
    }

    printf("even numbers: %d\n", count);
    return 0;
}
