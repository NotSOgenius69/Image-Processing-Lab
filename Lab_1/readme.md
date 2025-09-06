<h1>Assignment 01</h1>
<h2>Gaussian Image Convolution</h2>

In this assignment , a smoothing and sharpening kernel is generated from gaussian smoothing and sharpening function and then convolved with
a RGB and HSV image to see the results. Applied the effect to each R,G,B channel separately and then merged them to see the final output.

<h2>Outputs</h2>

<h3>Original Image</h3>
<img width="514" height="542" alt="image" src="https://github.com/user-attachments/assets/0202af47-6202-4fe3-98e2-ce005a982d0d" />

<h3>Smoothing Operation on Separate Channels</h3>
<img width="1360" height="543" alt="image" src="https://github.com/user-attachments/assets/7f01cdcd-d4aa-4c93-aad7-5fb41b43d2c2" />
<h3>Sharpening Operation on Separate Channels</h3>
<img width="1356" height="542" alt="image" src="https://github.com/user-attachments/assets/9dfca971-46f6-43d3-b9e1-cf18169a3978" />
<h3>Smoothing and Sharpening Final Results on RGB Image</h3>
<img width="1360" height="543" alt="image" src="https://github.com/user-attachments/assets/546b1e53-a4e2-45b8-827b-f2a48cd8361a" />
<h3>Smoothing and Sharpening Final Results on HSV Image</h3>
<img width="1357" height="542" alt="image" src="https://github.com/user-attachments/assets/6664c40a-5594-4a4a-a7df-56f38b80ae0e" />

<h1>Assignment 02</h1>
<h2>Convolution with SVD based Kernel Approximation</h2>

In this assignment , an arbitrary 2d kernel is taken and convolution is done on a RGB image.Later the 2d kernel is decomposed into two 1D filters using SVD technique and the rank-1 approximation of the kernel is taken from the largest sigma value from the sigma vector. The convolution with the actual kernel and the rank-1 approximation kernel is compared. The computational cost of both the operation is also compared and shown that the computational cost reduces by a great extent by decomposing the kernel into its 1D vectors through SVD technique.

<h2>Outputs</h2>
<img width="900" height="332" alt="image" src="https://github.com/user-attachments/assets/bd513cce-6b38-48ad-85b9-0b7c66b3b887" />
<h3>Comparison of the no. of operations</h3>
<img width="604" height="307" alt="image" src="https://github.com/user-attachments/assets/0d1c50d2-e432-48bd-9f6c-17b3e4fee727" />

