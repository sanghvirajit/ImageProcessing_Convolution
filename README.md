
# Kernel (image processing)

In image processing, a kernel, convolution matrix, or mask is a small matrix. It is used for blurring, sharpening, embossing, edge detection, and more. This is accomplished by doing a convolution between a kernel and an image.

The general form for matrix convolution is

![](images/convolution_picture.png)

The 3D visualization of the concolution operation can be seen as follow,

![convolution](https://user-images.githubusercontent.com/42026685/106669526-1b326580-65ac-11eb-9304-3d99ecb8b0c0.png)

Hence, we run the kernel/ filter over the original Image and we get the  output image.

The kernel matrix depends upon the features that we want to extract from the original image.

![unnamed](https://user-images.githubusercontent.com/42026685/106669849-9136cc80-65ac-11eb-9d3c-4d1c48681b05.png)
 
There are many such kind of filter such as Idendity, Edge detection, Sharpen, Box Blur, Gaussian Blur(3x3), Gaussian Blur(5x5), sobel edge filter, laplacian, laplacian of gaussian etc.

This convolution operation is the backborn of the Convolution Neural Networks (CNNs).
