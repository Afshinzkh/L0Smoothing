// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2014/2015, March 2 - April 3
// ###
// ###
// ### Thomas Moellenhoff, Robert Maier, Mohamed Souiai, Alfonso Ros Dos Santos, Caner Hazirbas
// ###
// ###


//TODO: to use planMany instead of plan1D
//TODO: not to use complex imgIn (to use R2C as in convolution example)





#include "aux.h"
#include <iostream>
#include <complex>
#include <cufft.h>

using namespace std;


typedef float2 Complex;

// Complex addition
__device__ __host__ Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex multiplication
__device__ __host__ Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}


// Complex scale
__device__ __host__ Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex division
__device__ __host__ Complex ComplexDiv(Complex a, Complex b)
{
		Complex c;
		c.x = (a.x * b.x + a.y * b.y)/(b.x*b.x + b.y*b.y);
		c.y = ( a.y * b.x - a.x * b.y )/(b.x*b.x + b.y*b.y); 
		return c;
}

// Complex Conjugate
__device__ __host__ Complex Conj(Complex a)
{
		Complex c;
		c.x = a.x;
		c.y = -a.y;
		return c;
}


//Kernel to Compute fft(dx) and fft(y)
__global__ void Calculate_fximgfyimg (float *fximg, float *fyimg, int w, int h, int nc){
	int x = threadIdx.x + blockDim.x * blockIdx.x;
  	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x<w && y<h){
		for(int c=0; c<nc; c++){
			int ind = x + w*y + w*h*c;		// take care of this indexing!!!!

			if( ind == (0 + (0)*w + w*h*c) )	fximg[ind] = 1.0f;			
			if( ind == ((w-1) + (0)*w + w*h*c) )	fximg[ind] = -1.0f;
			if( ind == (0 + (0)*w + w*h*c) )	fyimg[ind] = 1.0f;
			if( ind == (0 + (h-1)*w + w*h*c) )	fyimg[ind] = -1.0f;
			
		}		
	}	
}


// just to check the results seperately for each step
__global__ void Calculate_temp (cufftComplex *fximg, float *temp, int w, int h, int nc){
	int x = threadIdx.x + blockDim.x * blockIdx.x;
  	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x<w && y<h){
		for(int c=0; c<nc; c++){
		int ind = x + w*y + w*h*c;
		temp[ind] = sqrt(pow(fximg[ind].x ,2) + pow(fximg[ind].y ,2));
	}}
}



// Kernel to Compute h and v
__global__ void Calculate_hv(float *S, float *h, float *v, int width, int height, int nc, float lambda, float beta)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
  	int y = threadIdx.y + blockDim.y * blockIdx.y;
	float S_x, S_y;

	if (x+1<width && y+1<height){
		for (int c=0; c<nc; c++){		
			float right_pixel, up_pixel, center_pixel;
			int ind = x + width*y + width*height*c;
			if (x<width-1)	right_pixel = S[ind + 1];
			else		right_pixel = S[ind - width + 1];
			if (y<height-1)	up_pixel = S[ind + width];
			else 		up_pixel = S[ind -height + width];
			center_pixel = S[ind];
			
			S_x = center_pixel - right_pixel;
			S_y = center_pixel - up_pixel;
	
			float threshhold = S_x * S_x + S_y * S_y;
			if (threshhold <= lambda/beta){
				h[ind] = 0.0f;
				v[ind] = 0.0f;
			}
			else {
				h[ind] = S_x;
				v[ind] = S_y; 
			}
		}
	}
}




// kernel to Compute Denominator of EQ.8 --> fft(dx)* fft(dx) + fft(dy)* fft(dy)
__global__ void Calculate_MFT(cufftComplex *MFT, cufftComplex *fftfx_img, cufftComplex *fftfy_img, int width, int height, int nc, float beta)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
  	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x<width && y<height){
		for(int c=0; c<nc; c++){
			int ind = x + width*y+ c*height*width;	

			float2 result;
			result = ComplexAdd( ComplexMul(Conj(fftfx_img[ind]),fftfx_img[ind]), ComplexMul(Conj(fftfy_img[ind]),fftfy_img[ind]));

			MFT[ind].x = result.x;
			MFT[ind].y = 0.0f; 
		}
	}	
}

// Kernel to Compute the upstairs of EQ.8 --> fft(dx) * fft(h) + fft(dy) * fft(v)
__global__ void Calculate_FS(cufftComplex *FS, cufftComplex *fftfx_img, cufftComplex *fftfy_img,  cufftComplex *fft_h, cufftComplex *fft_v,
			     int width, int height, int nc)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
  	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x<width && y<height){
		for(int c=0; c<nc; c++){
			int ind = x + width*y + c*height*width;		// take care of this indexing!!!!
			
			float2 result; 
			result = ComplexAdd( ComplexMul(Conj(fftfx_img[ind]),fft_h[ind]), ComplexMul(Conj(fftfy_img[ind]),fft_v[ind]) );


			FS[ind].x = result.x;
			FS[ind].y = result.y;
			 		
		}
	}	

}

__global__ void Calculate_MFT_FS(cufftComplex *MFT, cufftComplex *FS, cufftComplex *fftfximg, cufftComplex *fftfyimg, 
													cufftComplex *ffth, cufftComplex *fftv, int width, int height, int nc, float beta){
	int x = threadIdx.x + blockDim.x * blockIdx.x;
  	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x<width && y<height){
		for(int c=0; c<nc; c++){
			int ind = x + width*y+ c*height*width;	

			float2 resultMFT;
			resultMFT = ComplexAdd( ComplexMul(Conj(fftfximg[ind]),fftfximg[ind]), ComplexMul(Conj(fftfyimg[ind]),fftfyimg[ind]));

			MFT[ind].x = resultMFT.x;
			MFT[ind].y = 0.0f; 

			float2 resultFS; 
			resultFS = ComplexAdd( ComplexMul(Conj(fftfximg[ind]),ffth[ind]), ComplexMul(Conj(fftfyimg[ind]),fftv[ind]) );


			FS[ind].x = resultFS.x;
			FS[ind].y = resultFS.y;
	
		}
	}
}



// kernel to Calculate the whole EQ.8 without the inverse
__global__ void Calculate_BIG_F(cufftComplex *BIG_F, cufftComplex *FS, cufftComplex *MFT, cufftComplex *fft_imgIn,
				int width, int height, int nc, float beta)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
  	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x<width && y<height){
		for(int c=0; c<nc; c++){
			int ind = x + width*y + c*height*width;		

			float2 num = ComplexAdd(fft_imgIn[ind], ComplexScale(FS[ind], beta));
			float2 denom;
			denom.x = 1.0f + beta * MFT[ind].x;
			denom.y = 0.0f;
			float2 result = ComplexDiv(num, denom);
			 BIG_F[ind].x = result.x/((float)height * (float)width);
	     		 BIG_F[ind].y = result.y/((float)height * (float)width);
			

		}		

	}	

}




// uncomment to use the camera
//#define CAMERA

int main(int argc, char **argv)
{
	// Before the GPU can process your kernels, a so called "CUDA context" must be initialized
	// This happens on the very first call to a CUDA function, and takes some time (around half a second)
	// We will do it right here, so that the run time measurements are accurate
	cudaDeviceSynchronize();  CUDA_CHECK;




	// Reading command line parameters:
	// getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
	// If "-param" is not specified, the value of "var" remains unchanged
	//
	// return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
	// input image
	string image = "";
	bool ret = getParam("i", image, argc, argv);
	if (!ret) cerr << "ERROR: no image specified" << endl;
	if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
	cout<< "image: "<< image << endl;
#endif
    
	// number of computation repetitions to get a better run time measurement
	int repeats = 1;
	getParam("repeats", repeats, argc, argv);
	cout << "repeats: " << repeats << endl;
    
	// load the input image as grayscale if "-gray" is specifed
	bool gray = false;
	getParam("gray", gray, argc, argv);
	cout << "gray: " << gray << endl;

	// Initialize "kappa" , deafult: 2.0 . at least: 1.05.
	// the variable helps to keep edges in a better way. but increase the iteration number
	float kappa = 2.0f;
	getParam("kappa", kappa, argc, argv);
	cout << "kappa: " << kappa << endl;

	// Initialize lambda.
	// define in what way lambda helps
	float lambda = 0.02f;
	getParam("lambda", lambda, argc, argv);
	cout << "lambda: " << lambda << endl;

	// Initialize Sigma for nois
	// define in what way lambda helps
	float sigma = 0.05f;
	getParam("sigma", sigma, argc, argv);
	cout << "Sigma: " << sigma << endl;



	// Init camera / Load input image
#ifdef CAMERA

	// Init camera
	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
	int camW = 640;
	int camH = 480;
	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
	// read in first frame to get the dimensions
	cv::Mat mIn;
	camera >> mIn;
    
#else
    
	// Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
	cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
	// check
	if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

	// convert to float representation (opencv loads image values as single bytes by default)
	mIn.convertTo(mIn,CV_32F);
	// convert range of each channel to [0,1] (opencv default is [0,255])
	mIn /= 255.f;
	// get image dimensions
	int w = mIn.cols;         // width
	int h = mIn.rows;         // height
	int nc = mIn.channels();  // number of channels
	cout << "image: " << w << " x " << h << endl;
	addNoise(mIn, sigma);



	// Set the output image format
	// ###
	// ###
	// ### TODO: Change the output image format as needed
	// ###
	// ###
	cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
	//cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
	//cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
	// ### Define your own output images here as needed

	// allocate raw input image array
	float *imgIn  = new float[(size_t)w*h*nc];

	// allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t)w*h*mOut.channels()];




	// Allocate arrays
	// input/output image width: w
	// input/output image height: h
	// input image number of channels: nc
	// output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

	//image size
	int img_size = w*h*nc;



	/***************************** INITIALIZE CUDA KERNEL AND VARIABLES **********************************/

	// initialize cuda kernel
	dim3 block = dim3(32,16,1);
	dim3 grid = dim3((w+block.x-1)/block.x, (h+block.y-1)/block.y, 1);

	// initialize Sizes of input and output
	size_t nbytesIn = w*h*nc*sizeof(float);
	size_t nbytesOut = w*h*mOut.channels()*sizeof(float);
	
	/************** Device variables ***************/
	

	float *d_h = NULL; 		// h 
	cufftComplex *d_ffth;		// fft(h)
	float *d_v = NULL;		// and v
	cufftComplex *d_fftv;		// and fft(v)
	cufftComplex *d_fftImgIn;	//device fft of imgIn	
	float *d_fximg;			// this is actually dx
	cufftComplex *d_fftfximg;	// and this is fft(dx)
	float *d_fyimg;			// this is actually dy
	cufftComplex *d_fftfyimg;	// and this is fft(dy)
	cufftComplex *d_MFT; 		// MFT is denominator of EQ.8 without fft(1)
	cufftComplex *d_FS;		// FS is upstairs of EQ.8 without fft(I)
	cufftComplex *d_BIG_F;		// is the whole EQ.8 without inversion
	

	// Allocate Memory to Device Variables

	cudaMalloc(&d_h, nbytesIn);					CUDA_CHECK;
	cudaMalloc(&d_v, nbytesIn); 					CUDA_CHECK;
	cudaMalloc(&d_fftImgIn, img_size*sizeof(cufftComplex)); 	CUDA_CHECK;
	cudaMalloc(&d_fximg, img_size*sizeof(float));	 		CUDA_CHECK;
	cudaMalloc(&d_fftfximg, img_size*sizeof(cufftComplex)); 	CUDA_CHECK;
	cudaMalloc(&d_fyimg, img_size*sizeof(float)); 			CUDA_CHECK;
	cudaMalloc(&d_fftfyimg, img_size*sizeof(cufftComplex)); 	CUDA_CHECK;
	cudaMalloc(&d_MFT, img_size*sizeof(cufftComplex));		CUDA_CHECK;
	cudaMalloc(&d_FS, img_size*sizeof(cufftComplex)); 		CUDA_CHECK;
	cudaMalloc(&d_BIG_F, img_size*sizeof(cufftComplex)); 		CUDA_CHECK;
	cudaMalloc(&d_ffth, img_size*sizeof(cufftComplex)); 		CUDA_CHECK;
	cudaMalloc(&d_fftv, img_size*sizeof(cufftComplex)); 		CUDA_CHECK;

	// Set values to each Device Variable
	
	cudaMemset(d_h, 0, nbytesIn); 					CUDA_CHECK;
	cudaMemset(d_v, 0, nbytesIn); 					CUDA_CHECK;
	cudaMemset(d_ffth, 0 , img_size*sizeof(cufftComplex)); 		CUDA_CHECK;
	cudaMemset(d_fftv, 0 , img_size*sizeof(cufftComplex)); 		CUDA_CHECK;
	cudaMemset(d_fftImgIn, 0, img_size*sizeof(cufftComplex)); 	CUDA_CHECK;	
	cudaMemset(d_fximg, 0 , img_size*sizeof(float)); 		CUDA_CHECK; 
	cudaMemset(d_fyimg, 0 , img_size*sizeof(float)); 		CUDA_CHECK;
	cudaMemset(d_fftfximg, 0 , img_size*sizeof(cufftComplex));	CUDA_CHECK;
	cudaMemset(d_fftfyimg, 0 , img_size*sizeof(cufftComplex));	CUDA_CHECK;
	cudaMemset(d_MFT, 0 , img_size*sizeof(cufftComplex)); 		CUDA_CHECK;	
	cudaMemset(d_FS, 0 , img_size*sizeof(cufftComplex)); 		CUDA_CHECK;	
	cudaMemset(d_BIG_F, 0 , img_size*sizeof(cufftComplex)); 	CUDA_CHECK;

	

	/***************************** Initial Cudafft Variables **********************************/
	cufftHandle fftPlan, ifftPlan, rcfftPlan;
        int sizes[] = { h, w , nc};
        int embed[] = { h, w , nc};

        cufftPlanMany(&fftPlan, 2, sizes, embed, 1, h * w, embed, 1, h * w, CUFFT_C2C, nc);
        cufftPlanMany(&ifftPlan, 2, sizes,embed, 1, h * w, embed, 1, h * w, CUFFT_C2R, nc);
	cufftPlanMany(&rcfftPlan, 2, sizes,embed, 1, h * w, embed, 1, h * w, CUFFT_R2C, nc);

	

	/***************************** Initial Calculations **********************************/
	int iteration = 0;	// to calculate number of iterations
	float beta_max = 100000.0f;
	float beta = 2.0f * lambda;
	


	/***************************** Calculate MFT *****************************************/
	/********** MFT is the denominator of equation 8 without F(1) ************************/

	Calculate_fximgfyimg <<<grid,block>>> (d_fximg, d_fyimg, w, h, nc);		CUDA_CHECK;
	cudaDeviceSynchronize();

	cufftExecR2C(rcfftPlan, d_fximg, d_fftfximg);
	cufftExecR2C(rcfftPlan, d_fyimg, d_fftfyimg);

	






	// For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
	// Read a camera image frame every 30 milliseconds:
	// cv::waitKey(30) waits 30 milliseconds for a keyboard input,
	// returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
	while (cv::waitKey(30) < 0)
	{
	// Get camera image
	camera >> mIn;
	// convert to float representation (opencv loads image values as single bytes by default)
	mIn.convertTo(mIn,CV_32F);
	// convert range of each channel to [0,1] (opencv default is [0,255])
	mIn /= 255.f;
#endif

	// Init raw input image array
	// opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
	// But for CUDA it's better to work with layered images: rrr... ggg... bbb...
	// So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
	convert_mat_to_layered (imgIn, mIn);
	float *d_imgIn = NULL; 		// input image
	float *d_S = NULL;		// output S
	cudaMalloc(&d_imgIn, nbytesIn);					CUDA_CHECK;
	cudaMalloc(&d_S, nbytesIn); 					CUDA_CHECK;
	cudaMemcpy(d_imgIn, imgIn, nbytesIn, cudaMemcpyHostToDevice);	CUDA_CHECK;	
	cudaMemcpy(d_S,     imgIn, nbytesIn, cudaMemcpyHostToDevice);	CUDA_CHECK;
	
	/********************** Calculate FFT for input image F(I) **************************/	
	cufftExecR2C(rcfftPlan, d_imgIn, d_fftImgIn); 					CUDA_CHECK;


	/***************************** Main Loop **********************************/
	

	

	while (beta < beta_max){
		
		Timer timer; timer.start();

		//Calculate h and v, Fourier transform of h and v 
		Calculate_hv <<<grid, block>>> (d_imgIn, d_h, d_v, w, h, nc, lambda, beta);				CUDA_CHECK;	
		cudaDeviceSynchronize();	
	
		//timer.end();  
		//float t = timer.get();  // elapsed time in seconds
		//cout << "time hv: " << t*1000 << " ms" << endl;


		cufftExecR2C(rcfftPlan, d_h, d_ffth); 
		cufftExecR2C(rcfftPlan, d_v, d_fftv);



		//Calculate_MFT <<<grid,block>>> (d_MFT, d_fftfximg, d_fftfyimg, w, h, nc, beta);	CUDA_CHECK;
		//cudaDeviceSynchronize();

		//timer.start();
		//Calculate FS i.e. fft(dx)*fft(h) + fft(dy)*fft(v)	
		//Calculate_FS <<<grid,block>>> (d_FS, d_fftfximg, d_fftfyimg, d_ffth, d_fftv, w, h, nc); 		CUDA_CHECK;
		//cudaDeviceSynchronize();



		Calculate_MFT_FS <<<grid, block>>> (d_MFT, d_FS, d_fftfximg, d_fftfyimg, d_ffth, d_fftv, w, h, nc, beta); CUDA_CHECK;
		cudaDeviceSynchronize();
		
		//timer.end();  t = timer.get();  // elapsed time in seconds
		//cout << "time FS: " << t*1000 << " ms" << endl;

		//timer.start();
		//Calculate BIG_F, i.e (F(I) + beta * FS)/ (F(1) + beta * MFT)
		Calculate_BIG_F <<<grid,block>>> (d_BIG_F, d_FS, d_MFT, d_fftImgIn, w, h, nc, beta);		CUDA_CHECK;
		cudaDeviceSynchronize();

		//timer.end();  t = timer.get();  // elapsed time in seconds
		//cout << "time BIG F: " << t*1000 << " ms" << endl;

		//Calculate S, i.e. inverse FFT(BIG_F) 
		cufftExecC2R(ifftPlan, d_BIG_F, d_S);	
		timer.end();  float t = timer.get();  // elapsed time in seconds
		cout << "time of 1 iteration: " << t*1000 << " ms" << endl;

		
	
		// update
		beta*=kappa;
		iteration+=1;
		cudaMemcpy(d_imgIn, d_S, nbytesIn, cudaMemcpyDeviceToDevice);		CUDA_CHECK;

	}


	

	cudaMemcpy(imgOut, d_S, nbytesOut, cudaMemcpyDeviceToHost);			CUDA_CHECK;






	// show input image
	showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

	// show output image: first convert to interleaved opencv format from the layered raw array
	convert_layered_to_mat(mOut, imgOut);
	showImage("Output", mOut, 100+w+40, 100);

	// ### Display your own output images here as needed





	//cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
	//cv::imwrite("image_result.png",mOut*255.f);

	// save input and result
	std::ostringstream nameInput;
	nameInput << "input_" << image << "_noise" << sigma << ".png";
	cv::imwrite(nameInput.str() ,mIn*255.f);	

	std::ostringstream nameOutput;
	nameOutput << "output_" << image << "_lambda_" << lambda << "_kappa_"<< kappa <<".png";
	cv::imwrite(nameOutput.str(), mOut*255.f);
	
	



	// free cuda memories
	cudaFree(d_imgIn);	CUDA_CHECK;
	cudaFree(d_S);  	CUDA_CHECK;	
	
#ifdef CAMERA
	// end of camera loop
	}
#else
	// wait for key inputs
	cv::waitKey(0);
#endif
	cudaFree(d_h);  	CUDA_CHECK;	
	cudaFree(d_v);  	CUDA_CHECK;	
	cudaFree(d_MFT); 	CUDA_CHECK;	
	cudaFree(d_FS);  	CUDA_CHECK;	
	cudaFree(d_BIG_F);  	CUDA_CHECK;
	cudaFree(d_ffth); 	CUDA_CHECK;
	cudaFree(d_fftv); 	CUDA_CHECK;
	cudaFree(d_fftImgIn); 	CUDA_CHECK;
	cudaFree(d_fximg);  	CUDA_CHECK;	
	cudaFree(d_fyimg);  	CUDA_CHECK;	
	cudaFree(d_fftfximg);  	CUDA_CHECK;	
	cudaFree(d_fftfyimg);  	CUDA_CHECK;		

	cufftDestroy(fftPlan);
	cufftDestroy(ifftPlan);
	cufftDestroy(rcfftPlan);
	// free allocated arrays
	delete[] imgIn;
	delete[] imgOut;

	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}



