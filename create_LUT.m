function LUT = create_LUT(x1, y1, x2, y2)
% This function creates a lookup table (LUT) for stretching the histogram
% of an image using the input points (x1, y1) and (x2, y2).

% Compute the slope of the line between (x1, y1) and (x2, y2)
a = y1 / x1;

% Create the LUT for the first section of the curve
LUT1 = a * (1:x1);

% Compute the slope of the line between (x1, y1) and (x2, y2)
b = (y2 - y1) / (x2 - x1);

% Create the LUT for the second section of the curve
LUT2 = b * (1:x2-x1) + y1;

% Compute the slope of the line between (x2, y2) and (255, 255)
c = (255 - y2) / (255 - x2);

% Create the LUT for the third section of the curve
LUT3 = c * (x2+1:256) + y2;

% Combine the three sections of the LUT
LUT = [LUT1, LUT2, LUT3];

% Normalize the LUT to have a maximum value of 255
LUT = LUT / max(LUT) * 255;

% Convert the LUT to an integer array for indexing
LUT = uint8(LUT);

end