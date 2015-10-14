#ifndef __IMG_H__
#define __IMG_H__

struct Image {
	int width;
	int height;
	int channel;
	int *data;
};

#define GetPixel(_IMG_,_H_,_W_) (_IMG_).data[(_H_)*(_IMG_).width+(_W_)]
#define GetRGB(_IMG_,_H_,_W_,_C_) (((_IMG_).data[(_H_)*(_IMG_).width+(_W_)]>>(8*((_IMG_).channel-(_C_))))&0xFF)

Image randomImage(int width, int height, int channels){
	Image img;
	img.width = width;
	img.height = height;
	img.channel = channels;
	img.data = (int*)malloc(width*height*sizeof(int));
	
	for(int i = 0; i<height; i++){
		for(int j = 0; j<width; j++){
			int pixel = 0;
			for(int k = 0; k<channels; k++)
				pixel = (pixel<<8) | rand()*0xFF;
			GetPixel(img,i,j) = pixel;
		}
	}
	return img;
}

#endif
