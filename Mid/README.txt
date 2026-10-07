CV LAB EXAM CHEAT SHEET - CODE FILES

37 separate Python/OpenCV files. Change placeholder filenames (image.jpg, road.jpg,
template.jpg, etc.) to your own files.

CORE RULES:
3 point correspondences -> affine
4 point correspondences -> homography
adaptive threshold -> uneven lighting
Otsu -> automatic global threshold
opening -> remove small noise
closing -> fill small gaps/holes
watershed -> separate touching objects
Canny -> edges
HoughLinesP -> lines
HoughCircles -> circles
SIFT + ratio test -> feature matching
RANSAC -> reject outliers
warpPerspective -> apply homography
bitwise_and + mask -> apply ROI
