<h1>Assignment 01</h1>
<h2>Histogram Equalization on Color and HSV Images</h2>

In this assignment, a color image is taken and histogram equalization is applied to enhance its contrast. The process is performed in two ways:

<ul>
  <li><b>RGB Channels:</b> The image is split into Blue, Green, and Red channels. Histogram equalization is applied separately to each channel using the cumulative distribution function (CDF). The equalized channels are then merged to form the output image.</li>
  <li><b>HSV Color Space:</b> The image is converted to HSV color space. Histogram equalization is applied only to the Value (V) channel, and the result is merged back with the original Hue (H) and Saturation (S) channels to produce the final equalized HSV image.</li>
</ul>

The assignment also visualizes the original and equalized images, their histograms, and CDFs for each channel using matplotlib.

<h2>Outputs</h2>

<h3>Outputs of the RGB image</h3>
<img width="1171" height="692" alt="image" src="https://github.com/user-attachments/assets/42a7f05e-b375-47d1-8fbe-c453c0f34d54" />
<h3>Outputs of the HSV image</h3>
<img width="1140" height="699" alt="image" src="https://github.com/user-attachments/assets/2ce409f6-e423-4dba-8c65-9d41c5ceea7c" />

<h2>Conclusion</h2>
We mainly compared the equalized image from two approach and show that the equalized image from hsv image is much more natural than the rgb equalized image.
<hr>
<hr>
<h1>Assignment 02</h1>
<h2>Histogram Matching</h2>

In this assignment, histogram matching is performed to adjust the pixel intensity distribution of an input image to match that of a target histogram. The process involves the following steps:

1. Generate a double Gaussian target histogram.
2. Compute the probability density function (PDF) and cumulative distribution function (CDF) for both the input image and the target histogram.
3. Create a mapping from the input CDF to the target CDF.
4. Apply the mapping to the input image to obtain the output image with the desired histogram.

The assignment also visualizes and compares the original and matched images, their histograms, and CDFs using matplotlib.

<h2>Outputs</h2>
<img width="1120" height="693" alt="image" src="https://github.com/user-attachments/assets/644b87d6-d138-494e-8fbc-718f2b5fbb6a" />

<h2>Conclusion</h2>
Histogram matching effectively transforms the input image to have a pixel intensity distribution that closely resembles the target histogram. The visualizations confirm that the matched image's histogram aligns well with the target histogram, demonstrating the success of the histogram matching process.
