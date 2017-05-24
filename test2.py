import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from random import shuffle
from tqdm import tqdm
from datetime import timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pickle
import prettytensor as pt
from cifarModel import create_network, global_step, x, y_true, img_size, num_channels, class_names

TEST_DIR = './testCifar'
LOGDIR = './logs/cifar_cnn/tensorboard/'
if not os.path.exists(LOGDIR):
  os.makedirs(LOGDIR)
MODEL_NAME = "./saved_model/cifar_cnn/cnn_model.ckpt"
MODEL_DIR = "./saved_model/cifar_cnn"
if not os.path.exists(MODEL_DIR):
  os.makedirs(MODEL_DIR)

IMAGE = "./testCifar/dog_cat_2.jpg"
DIRNAME = './testImg/'
IMG_SIZE=100
thres = 76
global sess
sess = tf.InteractiveSession()

def loadModel():
  # _, loss = create_network(training=True)
  y_pred, _ = create_network(training=False)
  y_pred_cls = tf.argmax(y_pred, 1)

  saver = tf.train.Saver()
  try:
      print("============================================>111 Trying to restore last checkpoint ...")

      last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=MODEL_DIR)

      # Try and load the data in the checkpoint.
      saver.restore(sess, save_path=last_chk_path)

      # If we get to this point, the checkpoint was successfully loaded.
      print("============================================> Restored checkpoint from:", MODEL_DIR)
  except:
      # If the above failed for some reason, simply
      # initialize all the variables for the TensorFlow graph.
      print("============================================> Failed to restore checkpoint. Initializing variables instead.")
      tf.global_variables_initializer().run() 

  tf.get_variable_scope().reuse_variables()  

def extractCoodinary(array):
  x = array[0][0]
  y = array[1][0]
  w = array[0][1] - x
  h = array[1][1] - y
  return x, y, w, h

def predict(img):
  # sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
  y_pred, _ = create_network(training=False)
  y_pred_cls = tf.argmax(y_pred, 1)
  feed_dict = {x:img}

  sample_pred_cls, sample_pred = sess.run([y_pred_cls, y_pred], feed_dict=feed_dict)
  print('sample_pred = ', sample_pred)
  print('sample_pred_cls = ', sample_pred_cls , 'true label: ', class_names[sample_pred_cls[0]])
  return class_names[sample_pred_cls[0]]

class classifyObj():
  """docstring for ClassName"""
  pos = {}
  img_blur_gray = []
  img_scaled = []

  def __init__(self, img):
    super(classifyObj, self).__init__()
    self.img = img

    max_dimension = max(img.shape)
    scale = 700/max_dimension
    self.img_scaled = cv2.resize(img, None, fx=scale, fy=scale)
    img_blur = cv2.GaussianBlur(self.img_scaled, (7,7), 0)
    self.img_blur_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    

  def thres_callback(self, img):
    thres = cv2.getTrackbarPos('Threshold', 'image')
    # gaus = cv2.adaptiveThreshold(self.img_blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    res, img_thres = cv2.threshold(self.img_blur_gray, thres, 255,cv2.THRESH_BINARY)
    edges = cv2.Canny(img_thres,10,250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    _, contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_copy = self.img_scaled.copy()
    total = 0
    self.pos = {}
    idx = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w>IMG_SIZE and h>IMG_SIZE:
            idx+=1
            self.pos[idx] = np.array([(x,x+w), (y,y+h)])
            cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0,0,0), 2)

    cv2.imshow('image', img_copy)
    print('index ========= ', idx)
    print('len(pos) ========', int(len(self.pos)))

  def predict_callback(self, ori_img):
    pass

  def __del__(self):
    super(classifyObj, self).__del__()
    classifyObj = self.__class__.__name__
    print('destroy all instances')

loadModel()
ori_img = cv2.imread(IMAGE)
classifyObj = classifyObj(ori_img)

cv2.namedWindow('image')
cv2.createTrackbar('Threshold','image',thres,255,classifyObj.thres_callback)
classifyObj.thres_callback(ori_img)

cv2.createTrackbar('switch', 'image', 0, 1, classifyObj.predict_callback)
classifyObj.predict_callback(ori_img)


while 1:
  test_img = classifyObj.img_scaled.copy()
  sw = cv2.getTrackbarPos('switch', 'image')
  if sw == 0:
    pass
  elif sw == 1:
    cv2.setTrackbarPos('switch', 'image', 0)
    print(int(len(classifyObj.pos)))
    for i in range(int(len(classifyObj.pos))):
      pos_x, pos_y, w, h = extractCoodinary(classifyObj.pos[i+1])
      img = test_img[pos_y:pos_y+h, pos_x:pos_x+w]
      img = cv2.resize(img, (img_size,img_size))
      img = np.reshape(img, [-1, img_size, img_size, num_channels])
      label = predict(img)
      cv2.rectangle(test_img, (pos_x,pos_y), (pos_x+w, pos_y+h), (0,0,0), 2)
      cv2.putText(test_img,str(label),(pos_x,pos_y+h), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)

    cv2.imshow('image', test_img)
    print('====================')

  test_img = []
  k = cv2.waitKey(5) & 0xFF
  if k == 27:
    break 

cv2.destroyAllWindows()


