import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import json

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
person=truck=both=False


# Number of classes the object detector can identify
NUM_CLASSES = 33
threshold=25
Person_only=[]
Truck_only=[]
Both=[]
def csv_to_pandas(filepath):
    csv=pd.read_csv(filepath)
    global person,truck,both
    for index, rows in csv.iterrows():
        person=truck=both=False
        print(rows['Image_name'])
        return_value=detect(rows['Image_name'])
        if return_value==0:
            Both.append(0)
            Person_only.append(8)
            Truck_only.append(8)
            print('None')
        if return_value==1:
            Both.append(0)
            Person_only.append(1)
            Truck_only.append(8)
            print('Person')
        if return_value==2:
            Both.append(0)
            Person_only.append(8)
            Truck_only.append(1)
            print('Truck')
        if return_value==3:
            Both.append(1)
            Person_only.append(1)
            Truck_only.append(1)
            print('Both')
    csv['Person_only']=Person_only
    csv['Truck_only']=Truck_only
    csv['Both']=Both
    csv.to_csv('result_csv.csv')
    '''
    global person, truck, both
    person=truck=both=False
    return_value=detect(csv.loc[13,'Image_name'])
    print(return_value)
    '''

def detect(image):    
    
    # Path to image
    PATH_TO_IMAGE=os.path.join(CWD_PATH,"JPEGImages")
    PATH_TO_IMAGE = os.path.join(PATH_TO_IMAGE,image)
    print(PATH_TO_IMAGE)
    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    width=300
    height=300

    dim=(width,height)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    #image=cv2.resize(image,dim)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    #json_info=json.loads(category_index)
    print(classes)
    print(scores)
    global person, truck, both

    for i in range(scores.shape[1]):
        if classes[0][i]==6 and (scores[0][i]*100)>threshold:
            print(scores[0][i]*100)
            person=True
        if classes[0][i]==13 and (scores[0][i]*100)>threshold:
            print(scores[0][i]*100)
            truck=True
       
    
    if person == True and truck == False:
        return 1
    
    elif person == False and truck == True:
        return 2
    
    elif person == True and truck == True:
        return 3
    else:
        return 0


def main(filename):
    csv_to_pandas(filename)


if __name__=='__main__':
    main('Test_set.csv')

