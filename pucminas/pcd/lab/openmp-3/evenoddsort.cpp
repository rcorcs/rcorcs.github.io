#include<algorithm>
#include<time.h>
#include<stdlib.h>
#include<stdio.h>

#include<omp.h>

using std::swap;

void evenOddSort(int A[], int begin, int end){
   for(int i = begin; i<end; i++){
      int first = i%2;
      
      #pragma omp parallel for
      for(int j = begin+first; j<end-1; j += 2){
         if(A[j]>A[j+1]){
            swap(A[j],A[j+1]);
         }
      }
   }
}

/* Function to print an array */
void printArray(int A[], int size){
    for(int i=0; i < size; i++)
        printf("%d ", A[i]);
    printf("\n");
}

/* Driver program to test sorting functions */
int main(int argc, char **argv){
    if(argc!=2){
       printf("required array size as argument.\n");
       return 1;
    }
    int arr_size = atoi(argv[1]);

    int arr[arr_size];
    
    srand(time(0));
    for(int i = 0; i<arr_size; i++){
       arr[i] = rand()%arr_size;
    }

    printf("Given array is \n");
    printArray(arr, arr_size);
    
    evenOddSort(arr,0,arr_size);

    printf("\nSorted array is \n");
    printArray(arr, arr_size);

    return 0;
}
