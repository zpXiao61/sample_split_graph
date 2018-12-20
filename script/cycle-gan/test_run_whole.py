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

model_flag = 'cycle-gan'

warm_loop = 0
loop = 1

if(len(sys.argv)>1):
    assert(len(sys.argv)==3)
    warm_loop = int(sys.argv[1])
    loop = int(sys.argv[2])

results_save_path = os.path.join(root_path,'results/'+model_flag+'/time_wholepb_warm'+str(warm_loop)+'_loop'+str(loop)+'.txt')
output_image_save_path = os.path.join(root_path,'results/'+model_flag+'/orange_wholepb.jpeg')

PATH_TO_FROZEN_GRAPH = os.path.join(root_path,'model_whole/'+model_flag+'/apple2orange.pb')

itensor_names = ['input_image:0']
otensor_names = ['output_image:0']

img = cv2.imread(os.path.join(root_path,"data/apple.png"))
img = cv2.resize(img,(256,256))
img = img[...,::-1]

rets,t = run_tf_pb(PATH_TO_FROZEN_GRAPH,itensor_names,[img],otensor_names,warm_loop,loop)

with open(output_image_save_path,'w+') as save_img:
    save_img.write(rets[0])

print('time: ', t)
with open(results_save_path,'w+') as rsp:
    rsp.write(str(t)+'\n')
    