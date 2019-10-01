import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath("/home/mahendra/python_prog/image-classifier"))
image_path=sys.argv[0]
filename = dir_path +'/' +image_path

image_size=128
num_channels=3
images = []
# Reading the image using OpenCV

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
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        # cv2.imwrite(img_name, frame)
        # print("{} written!".format(img_name))
        img_counter += 1


        image = frame
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

cam.release()
cv2.destroyAllWindows()
