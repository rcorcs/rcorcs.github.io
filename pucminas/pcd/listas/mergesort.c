/* Recursive C program for merge sort */
#include<time.h>
#include<stdlib.h>
#include<stdio.h>

/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
void merge(int arr[], int left, int m, int right);

/* l is for left index and r is right index of the sub-array
  of arr to be sorted */
void mergeSort(int arr[], int left, int right){
   if(left < right){
      int mid = left+(right-left)/2; //Same as (l+r)/2 but avoids overflow for large l & r
      mergeSort(arr, left, mid);
      mergeSort(arr, mid+1, right);
      merge(arr, left, mid, right);
   }
}

/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
void merge(int arr[], int left, int mid, int right){
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 =  right - mid;

    /* create temp arrays */
    int L[n1], R[n2];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1+ j];

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2){
        if (L[i] <= R[j]){
            arr[k] = L[i];
            i++;
        }else{
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there are any */
    while(i < n1){
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there are any */
    while(j < n2){
        arr[k] = R[j];
        j++;
        k++;
    }
}

/* Function to print an array */
void printArray(int A[], int size){
    for (int i=0; i < size; i++)
        printf("%d ", A[i]);
    printf("\n");
}

/* Driver program to test above functions */
int main(int argc, char **argv){
    if(argc!=2){
       printf("Required array size as argument.\n");
       printf("Usage:\n");
       printf("\t%s <array-size>\n", argv[0]);
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

    mergeSort(arr, 0, arr_size - 1);

    printf("\nSorted array is \n");
    printArray(arr, arr_size);

    return 0;
}
