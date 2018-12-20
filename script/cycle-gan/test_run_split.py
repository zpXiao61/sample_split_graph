import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import numpy as np
import cv2
import os
import time
import sys
from lib.running import run_tf_pb
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from lib.cfg import root_path

model_flag = 'cycle-gan'

warm_loop1 = 0
loop1 = 1
warm_loop2 = 0
loop2 = 1

if (len(sys.argv)>1):
    if len(sys.argv)==5:
        warm_loop1 = int(sys.argv[1])
        loop1 = int(sys.argv[2])
        warm_loop2 = int(sys.argv[3])
        loop2 = int(sys.argv[4])
    elif len(sys.argv)==3:
        warm_loop1 = int(sys.argv[1])
        loop1 = int(sys.argv[2])
        warm_loop2 = int(sys.argv[1])
        loop2 = int(sys.argv[2])
    else:
        print(os.path.abspath(__file__)+' wrong input args.')
        sys.exit(-1)
        
results_save_path = os.path.join(root_path,'results/'+model_flag+'/time_split_without_bm_warm_'+str(warm_loop1)+'_'+str(warm_loop2)+'_loop_'+str(loop1)+'_'+str(loop2)+'.txt')
output_image_save_path = os.path.join(root_path,'results/'+model_flag+'/orange_split_without_bm.jpeg')

PB1 = os.path.join(root_path,'model_parts/'+model_flag+'/part_1.pb')
PB2 = os.path.join(root_path,'model_parts/'+model_flag+'/part_2.pb')

img = cv2.imread(os.path.join(root_path,"data/apple.png"))
img = cv2.resize(img,(256,256))
img = img[...,::-1]

g1_input_tensors = ['input_image:0']
g1_output_tensors = ['G_7/output/Tanh:0']
g2_input_tensors = ['G_7/output/Tanh:0']
g2_output_tensors = ['output_image:0']

g1_input_values = [img]

g1_output_values,t1 = run_tf_pb(PB1,g1_input_tensors,g1_input_values,g1_output_tensors,warm_loop1,loop1)

g2_input_values = g1_output_values

g2_output_values,t2 = run_tf_pb(PB2,g2_input_tensors,g2_input_values,g2_output_tensors,warm_loop2,loop2)
        
with open(output_image_save_path,'w+') as save_img:
    save_img.write(g2_output_values[0])

print('time1: ', t1)
print('time2: ', t2)

with open(results_save_path,'w+') as rsp:
    rsp.write('time1: ' + str(t1)+'\n')
    rsp.write('time2: ' + str(t2)+'\n')


