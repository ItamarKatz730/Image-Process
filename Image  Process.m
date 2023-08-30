% Q 1.1

clear all; close all;

tire = imread('tire.tif');
figure,imhist(tire), title('tire');
figure, imshow(tire) , title('tire hist');

new_tire= uint8(0.4*double(tire)) ;
figure, imhist(new_tire), title('new tire');
figure, imshow(new_tire), title('new tire hist');

%% 
% Q 1.1.1
% 
% its not allways possible to restore the original image if we substurct the
% 
% constant because the max value is 255 and if the adding of the const gets
% 
% a value higher than 255 the substruct of the const will not restore the
% 
% original value. same goes for substructing under the min value of 0;
% 
% 
% 
% Q 1.1.2
% 
% if we duplicate the image with a number in range of 0<n<1 we can restore
% 
% otherwise we may not be able to restore;
%% Q 1.3

clear all; close all;
L = 1000;

% Const seed for const randoms
rng(5);

% x vector
vec = randi([0 255], 1, L, 'uint8');

% a
% Transformation
A = 0.4;
B = 50;


% Applying Transformation on x vector
tic;
res_1 = hist_strech(vec,A,B);
t_1 = toc;
% b
% LUT Transformation
% LUT creation - column of values by index
lut = uint8(linspace(0, 255, 256)*A + B);

% Applying Transformation on x vector - vector operation
tic;
res_2 = lut(double(vec)+1);
t_2 = toc ;

% LUT Transformation
% LUT creation - column of values by index
lut = uint8(linspace(0, 255, 256)*A + B);

% Applying Transformation on x vector 
tic;
res_2 = lut(double(vec)+1);
t_2 = toc ;

% c
if t_1 < t_2
    disp("Transformation Function is faster then LUT");
else
    disp("LUT is faster then Transformation Function");
end

% d
% Inv Transformation

transformation_inv = @(x) (x - A)/B;

tic;
% Applying Transformation on x vector - vector operation
res_1_inv = transformation_inv(res_1);
t_1_inv = toc;
e_trn = sum((res_1_inv - vec).^2);
snr_trn = 10*log10(sum(vec.^2)/e_trn);


% Inverse LUT
lut_inv = uint8((linspace(0, 255, 256) -  A) / B);

tic;
res_2_inv = lut_inv(double(res_2)+1);
t_2_inv = toc;
e_lut = sum((res_2_inv - vec).^2);
snr_lut = 10*log10(sum(vec.^2)/e_lut);

% x vector
vec_double = rand(1, L, 'double');

% a
% Transformation
A = 0.4;
B = 50/256;
transformation_double = @(x) (A*x+B);

% Applying Transformation on x vector
tic;
res_1_double = transformation_double(vec_double);
t_1 = toc;

% LUT Transformation
% LUT creation - column of values by index
lut_double = linspace(0, 1, 256)*A + B;

% Applying Transformation on x vector - vector operation
tic;
res_2_double = lut_double(uint8((vec_double)*256 + 1));
t_2 = toc ;
%% 2.1

clear all; close all;

pout = imread('pout.tif');

pout2 = imadjust(pout, [0.1 0.6],[0.1 1], 1);
figure('Position', [0, 0, 1200, 1000]);
subplot(2,2,1), imshow(pout), title('pout');
subplot(2,2,2), imshow(pout2) ,title('pout 2');
subplot(2,2,3), imhist(pout), title('pout hist');
subplot(2,2,4), imhist(pout2), title('pout 2 hist');

pout3 = imadjust(pout, [0.1 0.6],[1 0.1], 1);
figure('Position', [0, 0, 1200, 1000]);
subplot(2,2,1), imshow(pout), title('pout');
subplot(2,2,2), imshow(pout3) ,title('pout 3');
subplot(2,2,3), imhist(pout), title('pout hist');
subplot(2,2,4), imhist(pout3), title('pout 3 hist');

pout4 = imadjust(pout, [0.1 0.6],[0.1 1], 1.8);
figure('Position', [0, 0, 1200, 1000]);
subplot(2,2,1), imshow(pout), title('pout');
subplot(2,2,2), imshow(pout4) ,title('pout 4 gamma>1');
subplot(2,2,3), imhist(pout), title('pout hist');
subplot(2,2,4), imhist(pout4), title('pout 4 hist');


pout5 = imadjust(pout, [0.1 0.6],[0.1 1], 0.5);
figure('Position', [0, 0, 1200, 1000]);
subplot(2,2,1), imshow(pout), title('pout');
subplot(2,2,2), imshow(pout5) ,title('pout 5 gamma<1');
subplot(2,2,3), imhist(pout), title('pout hist');
subplot(2,2,4), imhist(pout5), title('pout 5 hist');
%% 3.1 / 2.1.1

clear all; close all;

tire = imread('tire.tif');
imhist(tire), title('tire'); 
figure, imshow(tire) , title('tire hist');
lut = uint8(linspace(0, 255, 256));

new_tire= uint8(0.4*double(tire)) ;
figure, imhist(new_tire), title('new tire'); 
figure, imshow(new_tire), title('new tire hist');

% negativ
negative_tire = uint8(255 - double(tire));
figure, imhist(negative_tire), title('negative tire'); 
figure, imshow(negative_tire), title('negative tire hist');

T_neg=255:-1:0;
figure, plot(T_neg);

T_bri=0:+50:255;
figure, plot(T_bri);

pout = imread("pout.tif");
imhist(pout), title('pout'); 
figure, imshow(pout) , title('pout hist');


dark_pout= uint8(hist_strech(double(pout),0.4,20));
lut_dark=uint8(hist_strech(double(lut),0.4,20));
figure, imhist(dark_pout), title('dark pout'); 
figure, imshow(dark_pout), title('dark pout hist');


figure, plot(lut_dark),title('pout vs dark pout');

bright_pout = uint8(double(pout) + 100);
figure, imhist(bright_pout), title('bright pout'); 
figure, imshow(bright_pout), title('bright pout');
%% 
% Gamma correction is a technique used to adjust the brightness and contrast 
% of an image by manipulating
% 
% the gamma value.
% 
% Gamma value is a parameter that affects the brightness of an image by changing 
% the relationship between
% 
% the pixel values and the displayed brightness levels.
% 
% 
% 
% Gamma transformation is the process of applying a gamma function to an image 
% to change the brightness
% 
% and contrast of an image.
% 
% The gamma function is a non-linear function that maps the input image values 
% to output values based
% 
% on the gamma value.
% 
% Gamma transformation is used to enhance the image contrast and improve the 
% visibility of details
% 
% 
% 
% imadjust applies a linear transformation, but it can be used to apply gamma 
% transformation by setting
% 
% the gamma parameter to a non-default value
% 
% Therefore the function is deppend on the user to decide which of the
% 
% above to use.
%% Q 3.2 / 2.1.2

clear all; close all;

P = (imread('pout.tif'));

hist_P = histogram(P);

imhist_P(:,1) = 0:255;
imhist_P(:,2) = imhist(P);
imhist_P(:,3) = imhist_P(:,2)/sum(imhist_P(:,2));
imhist_P(:,4) = cumsum(imhist_P(:,3));

figure('Position', [100, 100, 1900, 1000]);

subplot(2,3,1);
bar(hist_P(:,2));
title('Histogram without imhist:')

subplot(2,3,2);
bar(hist_P(:,3));
title('Histogram-NORM without imhist:')

subplot(2,3,3);
plot(hist_P(:,4));
title('Accumulate Histogram-NORM without imhist:')

subplot(2,3,4);
imhist(P);
title('Histogram with imhist:')
ylim([0,4000]);

subplot(2,3,5);
bar(imhist_P(:,3));
title('Histogram with imhist:')

subplot(2,3,6);
plot(imhist_P(:,4));
title('Accumulate Histogram-NORM with imhist:')
%% Q 2.3

clear all; close all;
p=imread('pout.tif');
hist = imhist(p);
My_LUT = zeros(numel(hist),1);
x1=input('Enter X1:');
y1=input('Enter Y1:');

x2=input('Enter X2:');
y2=input('Enter Y2:');

a=y1/x1;

for i=1:x1
    My_LUT(i,1)=i*a;
end

b=(y2-y1)/(x2-x1);
for i=x1+1:x2
    My_LUT(i,1)=(i-x1)*b+(y1);
end

c=(256-y2)/(256-x2);
for i=x2+1:256
    My_LUT(i,1)=(i-x2)*c+(y2);
end

figure; plot(My_LUT);
%% 
%% 2.3.1 , 3.4

clear all; close all;
img = imread('pout.tif');
stretched_img = stretch_hist(img, 50, 50, 200, 200);
figure;
imshow(img);
figure;
imshow(stretched_img);
%% 3.5

clear all; close all;

figure('Position', [0, 0, 1200, 1000]);
pout = imread('pout.tif');
hist_pout = histogram(pout); %% PDF and CDF
pout_eq = uint8(histogeq(pout));
pout_eq_hist = histogram(pout_eq);
CDF = cumsum(hist_pout(:,2));
[rows,columns] = size(pout);
L = double(max(max(pout)));
const = (rows.*columns) ./ L;
h = linspace(1 , 256 , 256);
T = @(x)CDF(x)/const - 1;

subplot(5,4,1), imshow(pout), title(' Pout');
subplot(5,4,2), bar(hist_pout(:,3)), title('Hist norm pout'); %PDF
subplot(5,4,3), plot(hist_pout(:,4)), title('CDF norm pout'); %CDF
subplot(5,4,[4,8,12]), plot(T(h)), title('T[pout]'); 
subplot(5,4,5), imshow(pout_eq), title('pout eq');
subplot(5,4,6), bar(pout_eq_hist(:,2)), title('Hist norm ');
subplot(5,4,7), plot(pout_eq_hist(:,4)), title('CDF norm ');
subplot(5,4,9), imshow(histeq(pout)), title('func histeq');
subplot(5,4,10), imhist(histeq(pout)), title('func histeq');

pout_histeq = histeq(pout);
diff = pout - pout_histeq;

subplot(5,4,[13,17]), imshow(pout), title('Pout');
subplot(5,4,[14,18]), imshow(pout_histeq), title('histeq pout');
subplot(5,4,[15,19]), imshow(diff), title('Difference');
subplot(5,4,[16,20]), imshow(diff , []), title('Difference');
%% Q 2.4.1 , 2.4.2 / 3.6

clear all; close all;

T= imread('tire.tif');
T_hist= imhist(T);
T_CDF = cumsum(T_hist)/numel(T);

A= imread('cameraman.tif');
C_hist= imhist(A);
C_CDF = cumsum(C_hist)/numel(A);

% tire to cameraman
M1=zeros(1,256,'uint8');
for i=1:256
    Diff = abs(T_CDF(i)-C_CDF);
    [~,ind] = min(Diff);
    M1(i)=ind-1;
end
Match1 = M1(double(T)+1);

% cameraman to tire  
M2=zeros(1,256,'uint8');
for i=1:256
    Diff = abs(C_CDF(i)-T_CDF);
    [~,ind] = min(Diff);
    M2(i)=ind-1;
end
Match2 = M2(double(A)+1);

figure('Position', [0, 0, 1200, 1000]);
subplot(3,2,1); imshow(T); title('Source Image');
subplot(3,2,2); imshow(A); title('Ref Image');
subplot(3,2,3); imhist(T); title('Source Image hist');
subplot(3,2,4); imhist(A); title('Ref Image hist');
subplot(3,2,5); plot(T_CDF); title('accumulative hist Tire ');
subplot(3,2,6); plot(C_CDF); title('accumulative hist Cameraman');


figure('Position', [0, 0, 1200, 1000]);
subplot(3,2,1); plot(M1); title('Transform G');
subplot(3,2,2); plot(M2); title('Transform G1');
subplot(3,2,3); imshow(Match1); title('New P1= G(P1)');
subplot(3,2,4); imshow(Match2); title('New P2= G1(P2)');
subplot(3,2,5); imhist(Match1); title('New P1 hist');
subplot(3,2,6); imhist(Match2); title('New P2 hist');
%% 3.6 using histeq

clear all; close all;

T= imread('tire.tif');
T_hist= imhist(T);

C= imread('cameraman.tif');
C_hist= imhist(C);

M1=histeq(T,C_hist);
M2=histeq(C,T_hist);

figure('Position', [0, 0, 1200, 1000]);
subplot(4,2,1),imshow(T),title('tire');
subplot(4,2,2),imshow(C),title('cameraman');
subplot(4,2,3),imhist(T),title('tire hist');
subplot(4,2,4),imhist(C),title('cameraman hist');
subplot(4,2,5),imshow(M1),title('tire with cameraman hist');
subplot(4,2,6),imshow(M2),title('cameraman with tire hist');
subplot(4,2,7),imhist(M1);
subplot(4,2,8),imhist(M2);
%% 
% As we can see there is no difference between the two methods