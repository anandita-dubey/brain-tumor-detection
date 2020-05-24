import tensorflow as tf
import os
import numpy as np
from PIL import Image
from skimage import transform
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
import json
import imageio
import scipy.ndimage


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
data_dict = {}
UPLOAD_FOLDER = "/opt/python/current/app/static/uploads/Results/"

# Loading trained and saved model
graph = tf.Graph()
with graph.as_default():
       session = tf.Session()
       with session.as_default():
            model = tf.keras.models.load_model("/opt/python/current/app/saved_model/Detection_from_cropped_images.h5",compile=False)
graph_var = graph
session_var = session
graph1 = tf.Graph()
with graph1.as_default():
       session1 = tf.Session()
       with session1.as_default():
            model1 = tf.keras.models.load_model("/opt/python/current/app/saved_model/Class_Detection_from_cropped_images.h5",compile=False)
graph_var1 = graph1
session_var1 = session1


# Function for loading and pre processing an image for predciton
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype("float32") / 255
    np_image = transform.resize(np_image, (150, 150, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


# Function for classifying the tumor if tumor is found
def classification(image, im1):
    with graph_var1.as_default():
        with session_var1.as_default():
            prediction1 = model1.predict(image)
            numpy_prediction = np.array(prediction1)
            max_probability = np.amax(numpy_prediction, axis=1)
            for i in range(len(numpy_prediction)):
                ind = list(
                    np.unravel_index(
                        np.argmax(numpy_prediction[i], axis=None), numpy_prediction[i].shape
                    )
                )
                if ind[0] == 0:
                    print("Class 1:meningioma tumor", max_probability)
                    data_dict["Class"] = "Class 1:meningioma tumor"
                    ShowImage1("Watershed segmented image", im1, "gray")
                elif ind[0] == 1:
                    print("Class 2:glioma tumor", max_probability)
                    data_dict["Class"] = "Class 2:glioma tumor"
                    ShowImage1("Watershed segmented image", im1, "gray")
                elif ind[0] == 2:
                    print("Class 3:pituitary tumor", max_probability)
                    data_dict["Class"] = "Class 3:pituitary tumor"
                    ShowImage1("Watershed segmented image", im1, "gray")
            data_dict["c1_probability"] = round(numpy_prediction[0][0] * 100, 2)
            data_dict["c2_probability"] = round(numpy_prediction[0][1] * 100, 2)
            data_dict["c3_probability"] = round(numpy_prediction[0][2] * 100, 2)


def ShowImage1(title, img, ctype):
    plt.figure(figsize=(4, 4))
    if ctype == "gray":
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(UPLOAD_FOLDER, "Result.png"))
        # plt.show()


def predictresult(inputimage):
    img = cv2.imread(inputimage)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    grayscale = imageio.imread(inputimage, as_gray=True)
    grayscale = 255 - grayscale
    median_filtered = scipy.ndimage.median_filter(grayscale, size=3)
    plt.figure(figsize=(4, 4))
    plt.imshow(median_filtered, cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(UPLOAD_FOLDER, "Result_watershed.png"))

    # Giving an image for pre processing
    image = load(inputimage)

    # Time for prediction
    with graph_var.as_default():
        with session_var.as_default():
            prediction = model.predict(image)
            # Checking the class with max probability
            numpy_prediction = np.array(prediction)
            max_probability = np.amax(numpy_prediction, axis=1)
            data_dict["No_probability"] = round(numpy_prediction[0][0] * 100, 2)
            data_dict["Yes_probability"] = round(numpy_prediction[0][1] * 100, 2)
            for i in range(len(numpy_prediction)):
                ind = list(
                    np.unravel_index(
                        np.argmax(numpy_prediction[i], axis=None), numpy_prediction[i].shape
                    )
                )
                if ind[0] == 0:
                    print("\nTumor ? : NO", max_probability)
                    data_dict["Tumor"] = "No"
                elif ind[0] == 1:
                    print("\nTumor ? : YES", max_probability)
                    data_dict["Tumor"] = "Yes"
                    print("\nTumor Class:")
                    classification(image, im1)
            print(data_dict)
            return json.dumps(data_dict)
