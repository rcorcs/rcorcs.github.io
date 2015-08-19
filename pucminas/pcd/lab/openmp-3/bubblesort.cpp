#include<algorithm>
#include<time.h>
#include<stdlib.h>
#include<stdio.h>

#include<omp.h>

using std::swap;

void bubbleSort(int A[], int begin, int end){
   for(int i = begin; i<end; i++){
      for(int j = begin; j<end-1; j++){
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
    if(argc!=3){
       printf("required array size as argument.");
    }
    int arr_size = atoi(argv[1]);
    int nThreads = atoi(argv[2]);

    int arr[arr_size];
    srand(time(0));
    for(int i = 0; i<arr_size; i++){
       arr[i] = rand()%arr_size;
    }

    printf("Given array is \n");
    printArray(arr, arr_size);
    
    bubbleSort(arr,0,arr_size);

    printf("\nSorted array is \n");
    printArray(arr, arr_size);

    return 0;
}
