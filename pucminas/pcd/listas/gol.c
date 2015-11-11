#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void print(int *mat, int width, int height){
	for(int i = 0; i< height; i++){
		for(int j = 0; j< width; j++){
			printf( "%c",(mat[i*width+j] == 0)? '.':'+' );
		}
		printf("\n");
	}
	printf("\n");
}

void randomFill(int *mat, int width, int height){
	srand(time(NULL));

	for(int i = 0; i< height; i++){
		for(int j = 0; j< width; j++){
			mat[i*width+j] = rand()%2;
		}
	}
}

int countNeighboors(int *mat, int width, int height, int row, int col){
	int neighbors = 0;
	for(int i = row-1; i<=row+1; i++){
		for(int j = col-1; j<=col+1; j++){
         if(i>=0 && i<height && j>=0 && j<width)
				neighbors += mat[i*width+j];
		}
	}

	return neighbors;
}

void update(int *in, int *out, int width, int height){
	for(int i = 0; i<height; i += 1){
		for(int j = 0; j<width; j += 1){
			int neighbors = countNeighboors(in, width, height, i, j);
         out[i*width+j] = ((neighbors==3 || (in[i*width+j]==1 && neighbors==2))?1:0);
		}
	}	
}

int main(int argc, char** argv){
	if(argc != 3){
		printf("gol <width> <height>\n");
		return 0;
	}

	int width = atoi(argv[1]);
	int height = atoi(argv[2]);

	int *in, *out;
	in = (int*)malloc(sizeof(int)*width*height);
	out = (int*)malloc(sizeof(int)*width*height);

	randomFill(in,width,height);

   print(in, width, height);

	update(in,out,width,height);

   print(out, width, height);

	free(in);
	free(out);

	return 0;
}















