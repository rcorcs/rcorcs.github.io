#ifndef ARRAY_H
#define ARRAY_H

template <typename T>
class array1d {
private:
	T *arr;
	size_t n;

public:
	array1d(size_t n){
		this->n = n;
		this->arr = new T[n];
	}

	inline ~array1d(){
		delete this->arr;
	}

	inline T &operator()(size_t i){
		return this->arr[i];
	}

	inline size_t size(){
		return this->n;
	}
};

template <typename T>
class array2d {
private:
	T *arr;
	size_t w;
	size_t h;

public:
	array2d(size_t width, size_t height){
		this->w = width;
		this->h = height;
		this->arr = new T[width*height];
	}

	inline ~array2d(){
		delete this->arr;
	}

	inline T &operator()(size_t h, size_t w){
		return this->arr[h*this->w + w];
	}

	inline size_t width(){
		return this->w;
	}

	inline size_t height(){
		return this->h;
	}
};

#endif
