from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2


def scan_image(image):
	# image = cv2.imread(img)
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	print "STEP 1: Edge Detection"
	# cv2.imshow("Image", image)
	# cv2.imshow("Edged", edged)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	_,cnts,hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		print(len(approx))

 		if len(approx) == 4:
			screenCnt = approx
			break

		else:
			return


	print "STEP 2: Find contours of paper"
	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	# cv2.imshow("Outline", image)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)


	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	warped = threshold_adaptive(warped, 251, offset = 10)
	warped = warped.astype("uint8") * 255


	print "STEP 3: Apply perspective transform"
	# cv2.imshow("Original", imutils.resize(orig, height = 650))
	cv2.imshow("Scanned", imutils.resize(warped, height = 650))

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# scan_image(cv2.imread("page.jpg"))

cam = cv2.VideoCapture(0)
cam.read()
cv2.namedWindow("test")
img_counter = 0

while True:
	ret, frame = cam.read()
	cv2.imshow("test", frame)
	if not ret:
		break
	k = cv2.waitKey(1)
	if k%256 == 27:
		print("Escape hit, closing...")
		break
	elif k%256 == 32:
		img_name = "opencv_frame_{}.png".format(img_counter)
		# cv2.imwrite(img_name, frame)
		scan_image(frame)

		# print("{} written!".format(img_name))
		img_counter += 1

cam.release()
