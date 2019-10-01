from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2
import os,glob,cv2
import sys
import tensorflow as tf


image_size=128
num_channels=3
images = []

def scan_image(image, img_counter):
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
	# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
	img_name = "opencv_frame_{}.png".format(img_counter)
	cv2.imwrite(img_name, imutils.resize(warped, height = 650))



######################################################################################################
	image = cv2.imread(img_name)
	# image = cv2.imread(frame)
	# Resizing the image to our desired size and preprocessing will be done exactly as done during training
	image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
	# images.append(image)
	images = np.array([image], dtype=np.uint8)
	# print(image)
	images = images.astype('float32')
	images = np.multiply(images, 1.0/255.0)
	#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
	x_batch = images.reshape(1, image_size,image_size,num_channels)

	## Let us restore the saved model
	sess = tf.Session()
	# Step-1: Recreate the network graph. At this step only graph is created.
	saver = tf.train.import_meta_graph('left_right-model.meta')
	# Step-2: Now let's load the weights saved using the restore method.
	saver.restore(sess, tf.train.latest_checkpoint('./'))

	# Accessing the default graph which we have restored
	graph = tf.get_default_graph()

	# Now, let's get hold of the op that we can be processed to get the output.
	# In the original network y_pred is the tensor that is the prediction of the network
	y_pred = graph.get_tensor_by_name("y_pred:0")

	## Let's feed the images to the input placeholders
	x= graph.get_tensor_by_name("x:0")
	y_true = graph.get_tensor_by_name("y_true:0")
	y_test_images = np.zeros((1, 4))


	### Creating the feed_dict that is required to be fed to calculate y_pred
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	result=sess.run(y_pred, feed_dict=feed_dict_testing)
	# result is of this format [probabiliy_of_rose probability_of_sunflower]
	print(result[0])

	if result[0][0] > 0.8:
		print("GO")

	elif result[0][1] > 0.8:
		print("LEFT")

	elif result[0][2] > 0.8:
		print("RIGHT")

	else:
		print("STOP")


	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

# scan_image(cv2.imread("page.jpg"))

cam = cv2.VideoCapture(1)
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
		scan_image(frame, img_counter)

		# print("{} written!".format(img_name))
		img_counter += 1

cam.release()
