import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import numpy as np
import cv2
import os
import time
import sys
from lib.cfg import root_path
from lib.running import run_tf_pb

model_flag = 'ssd_mobilenet_v2_coco_2018_03_29'

warm_loop = 0
loop = 1

if(len(sys.argv)>1):
    assert(len(sys.argv)==3)
    warm_loop = int(sys.argv[1])
    loop = int(sys.argv[2])

results_save_path = os.path.join(root_path,'results/'+model_flag+'/time_wholepb_warm'+str(warm_loop)+'_loop'+str(loop)+'.txt')

PATH_TO_FROZEN_GRAPH = os.path.join(root_path,'model_whole/'+model_flag+'/frozen_inference_graph.pb')

itensor_names = ['image_tensor:0']
otensor_names = ['num_detections:0','detection_classes:0','detection_boxes:0','detection_scores:0']

img = cv2.imread(os.path.join(root_path,"data/aa.JPEG"))

img = np.expand_dims(img,0)

rets,t = run_tf_pb(PATH_TO_FROZEN_GRAPH,itensor_names,[img],otensor_names,warm_loop,loop)
        
print('num_detections: ', rets[0])
print('detection_classes: ', rets[1])        
print('detection_boxes: ', rets[2])        
print('detection_scores: ', rets[3])

print('time: ', t)
with open(results_save_path,'w+') as rsp:
    rsp.write(str(t)+'\n')
    