function stretched_img = stretch_hist(img, x1, y1, x2, y2)
% This function stretches the histogram of an input image using the
% specified input points (x1, y1) and (x2, y2).

% Compute the histogram of the input image
hist = imhist(img);

% Create a lookup table (LUT) using the input points (x1, y1) and (x2, y2)
LUT = create_LUT(x1, y1, x2, y2);

% Apply the LUT to the input image to stretch the histogram
stretched_img = intlut(img, LUT);

% Plot the original and stretched image histograms
figure;
subplot(2,1,1);
imhist(img);
title('Original Image Histogram');

subplot(2,1,2);
imhist(stretched_img);
title('Stretched Image Histogram');

end