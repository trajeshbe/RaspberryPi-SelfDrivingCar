import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import sys
import io
#from grabscreen import grab_screen
#from getkeys import key_check
import pygame
import os

## Import Modules for Socket Connections
from tkinter import *
from socket import *      # Import necessary modules
import datetime
from time import ctime
#from client_App import forward_fun
#from client_App import right_fun
#import client_App


from collections import defaultdict
from io import StringIO
import matplotlib
#matplotlib.use('GTK3Cairo')
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from PIL import Image

# import the necessary packages
import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

#Resize for Training

WIDTH = 160
HEIGHT = 120


## Pi Car Host Info
ctrl_cmd = ['forward', 'backward', 'left', 'right', 'stop', 'read cpu_temp', 'home', 'distance', 'x+', 'x-', 'y+', 'y-', 'xy_home']


#top = Tk()   # Create a top window
#top.title('Raspberry Pi Self Driving Car')
#top.mainloop()
HOST = '169.254.0.11'    # Server(Raspberry Pi) IP address
PORT = 21567
BUFSIZ = 1024             # buffer size
ADDR = (HOST, PORT)

tcpCliSock = socket(AF_INET, SOCK_STREAM)   # Create a socket
tcpCliSock.connect(ADDR)

## Set Speed 
tmp = 'speed'
global spd
#spd = speed.get()
#print('Current Speed',spd)
spd=35
data = tmp + str(spd)  # Change the integers into strings and combine them with the string 'speed'. 
print ('sendData = %s' % data)
tcpCliSock.send(data.encode())  # Send the speed data to the server(Rasp













 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1280, 960) #40*480
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(1280, 960))
stream = io.BytesIO()
#import numpy as np
import cv2
#cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output15.avi',fourcc, 90.0, (1280, 960))

cap=rawCapture
'''
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

print(tf.__version__)

'''
#os = import_system_module("os")

#import matplotlib
#matplotlib.use('GTK3Cairo')
#import matplotlib.pyplot as plt

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
sys.path.append("/home/pi/tensorflow/models")
sys.path.append("/home/pi/tensorflow/models-master/models-master/research")
sys.path.append("/home/pi/tensorflow/models-master/models-master/research/slim")
sys.path.append("/home/pi/tensorflow/models-master/models-master/research/object_detection/utils")
sys.path.insert(0,"/home/pi/tensorflow/models-master/models-master/research/object_detection")
print(sys.path)

# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:


import imp
label_map_util = imp.load_source('module.name', '/home/pi/tensorflow/models-master/models-master/research/object_detection/utils/label_map_util.py')

#from utils import label_map_util

#from utils import visualization_utils as vis_util


import imp
visualization_utils= imp.load_source('module.name', '/home/pi/tensorflow/models-master/models-master/research/object_detection/utils/visualization_utils.py')
vis_util=visualization_utils


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/pi/tensorflow/models-master/models-master/research/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[5]:

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)

tar_file = tarfile.open('/home/pi/tensorflow/ssd_mobilenet_v1_coco_11_06_2017.tar.gz.1')

for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  print(file_name)

  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    print('Tensorflow Graph imported')



# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  print('Image Size', image.size)       
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images,
#just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/home/pi/tensorflow/images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image1.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)




#Grab Driving Keys

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [A,W,D] boolean values.
    '''
    output = [0,0,0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'W' in keys:
        output[1] = 1
    elif 'E' in keys:
        output[3] = 1
    elif 'S' in keys:
        output[4] = 1
   
    return output


file_name = 'training_data_keys_v15.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []



paused = False
pygame.init()
size = (640,480)
pygame.display.set_caption("OpenCV camera stream on Pygame")
screen=pygame.display.set_mode((640,480))

# In[10]:
dirname='/home/pi/car/Pi3/Sunfounder_Smart_Video_Car_Kit_for_RaspberryPi/client/selfdriving/driving_images'
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    count = 6000
    #while True:
    #with picamera.PiCamera() as camera:
    try:
      for frame in camera.capture_continuous(rawCapture, format="bgr"): 
        if not paused:  
          #ret, image_np = cap.read()
          image_np = np.array(frame.array)
          cv2.imwrite(os.path.join(dirname, "frame%d.jpg" %count), image_np)
          image_np_without_object = image_np

          #cv2.imshow('Show Image',image_np_resize)
          #cv2.waitkey(0)
          keys='Q'
          #time.sleep(2)

   

          ## Tesnforflow Object Detection Start
          

          #frame = cv2.flip(frame,0)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

          #print(np.squeeze(classes).astype(np.int32))
          #print(category_index)
          #print(image_np)
          #print(image_tensor)
          #print(category_index)
          #print(image_np)

          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)

          #cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
          #cv2.imshow('object detection',image_np)
          cv2.imwrite("frame%d.jpg" %count, image_np)     # save frame as JPEG file


          screen.fill([0,0,0])
          frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

          frame = np.rot90(frame)
          frame = pygame.surfarray.make_surface(frame)
          frame = pygame.transform.flip(frame, True, False)

          screen.blit(frame, (0,0))
          pygame.display.update()
   

  ##        ## Tesnforflow Object Detection End

          
          #Create Training Data with Keys
          #keys = key_check()
          
          while True:
             print('Waiting for Keypress...')
             key_flag=1
             for event in pygame.event.get():
              #while True:
                
                print('Key Flag',key_flag)
                #print('event.type',event.type)
                if event.type == pygame.KEYDOWN and key_flag==1:
                  #print('Waiting for Keypress...')
                  #pygame.event.clear()
                  #event = pygame.event.wait()
                  if event.key == pygame.K_w:
                    print('Forward')
                    tcpCliSock.send('home'.encode())
                    time.sleep(1)
                    tcpCliSock.send('forward'.encode())
                    time.sleep(1)
                    tcpCliSock.send('stop'.encode())
                    keys='W'
                    key_flag=0
                  elif event.key == pygame.K_a:
                    print('Left')
                    tcpCliSock.send('left'.encode())
                    time.sleep(0.5)
                    tcpCliSock.send('forward'.encode())
                    time.sleep(0.5)
                    tcpCliSock.send('stop'.encode())
                    keys='A'
                    key_flag=0
                  elif event.key == pygame.K_s:
                    print('Reverse')
                    tcpCliSock.send('home'.encode())
                    time.sleep(1)
                    tcpCliSock.send('backward'.encode())
                    time.sleep(1)
                    tcpCliSock.send('stop'.encode())
                    keys='S'
                    reverse = True
                    key_flag=0
                  elif event.key == pygame.K_e:
                    print('Home')
                    tcpCliSock.send('home'.encode())
                    time.sleep(1)
                    tcpCliSock.send('xy_home'.encode())
                    time.sleep(1)
                    
                    #tcpCliSock.send('x-'.encode())
                    #time.sleep(1)
                    #tcpCliSock.send('y+'.encode())
                    #time.sleep(1)
                    keys='E'
                    home = True
                    key_flag=0
                  elif event.key == pygame.K_d:
                    print('Right')
                    
                    tcpCliSock.send('x+'.encode())
                    time.sleep(0.5)
                    tcpCliSock.send('y-'.encode())
                    time.sleep(0.5)
                    #tcpCliSock.send('right'.encode())
                    #time.sleep(1)
                    tcpCliSock.send('x+'.encode())
                    time.sleep(0.5)
                    tcpCliSock.send('y-'.encode())
                    time.sleep(0.5)
                    tcpCliSock.send('right'.encode())
                    time.sleep(0.5)
 
                    cap = cv2.VideoCapture(0)
                    while(cap.isOpened()):
                          ret, image_web_np = cap.read()
                          if ret==True:
                            image_web_np = cv2.flip(image_web_np,0)
                            #cv2.imshow('frame',image_web_np)
                            print('Writing front cam image..')
                            
                            cv2.imwrite(os.path.join(dirname, "front_cam%d.jpg" %count),image_web_np)

                            cap.release()
                            break
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                              break
                          else:
                            break

                    
                    tcpCliSock.send('forward'.encode())
                    time.sleep(0.5)
                    tcpCliSock.send('xy_home'.encode())
                    time.sleep(0.5)
                    tcpCliSock.send('stop'.encode())
                    keys='D'
                    right = True
                    key_flag=0
                elif event.type == pygame.KEYUP:
                  if event.key == pygame.K_w:
                    forward = False
                    key_flag=0
                  elif event.key == pygame.K_a:
                    left = False
                    key_flag=0
                  elif event.key == pygame.K_s:
                    reverse = False
                    key_flag=0
                  elif event.key == pygame.K_d:
                    right = False
                    key_flag=0
                  elif event.key == pygame.K_e:
                    home = False
                    key_flag=0
                  
                    
             time.sleep(1)
                  
             if(keys!='Q'):
               break
             #
          print('Keypress',keys)
          output = keys_to_output(keys)
          print('Keys',output)
          time.sleep(1)
          image_np_without_object=cv2.resize(cv2.imread(os.path.join(dirname, "frame%d.jpg" %count),cv2.IMREAD_GRAYSCALE),(WIDTH,HEIGHT))
          if (keys=='D'):
            image_np_front_without_object=cv2.resize(cv2.imread(os.path.join(dirname, "front_cam%d.jpg" %count),cv2.IMREAD_GRAYSCALE),(WIDTH,HEIGHT))
            #training_data.append([np.array(image_np_front_without_object),np.array(output)])

          image_np_without_object_cropped=image_np_without_object[int(HEIGHT/2):HEIGHT , 0:int(WIDTH)]
        #image_np_without_object_cropped = [image_np_without_object.shape[0]/2:image_np_without_object.shape[0]]
        #image_np_without_object_cropped = cv2.resize(image_np_without_object ,None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)

          training_data.append([np.array(image_np_without_object),np.array(output)])
          print('Training Image shape',image_np_without_object.shape)
          print('Training Image shape -Cropped',image_np_without_object_cropped.shape)
          cv2.imwrite(os.path.join(dirname, "cropped_frame%d.jpg" %count), image_np_without_object_cropped)
          print('Lenght of Traning Data',len(training_data))
          np.save(file_name,training_data)

          #if len(training_data) % 1 == 0:
           #       print('Lenght of Traning Data',len(training_data))
           #       np.save(file_name,training_data)

          keys = pygame.key.get_pressed()
##          if 'T' in keys:
##              if paused:
##                  paused = False
##                  print('unpaused!')
##                  time.sleep(1)
##              else:
##                  print('Pausing!')
##                  paused = True
##                  time.sleep(1)

          count += 1
          rawCapture.truncate()
          rawCapture.seek(0)
          out.write(image_np)

          time.sleep(1)
          #if process(rawCapture):
           #break 

          if cv2.waitKey(25) & 0xFF == ord('q'):
             cv2.destroyAllWindows()
             break
      
    finally:
      camera.close()
      cap.close()
      tcpCliSock.close()
          
        

    
  
