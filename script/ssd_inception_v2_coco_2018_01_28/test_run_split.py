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

model_flag = 'ssd_inception_v2_coco_2018_01_28'

warm_loop1 = 0
loop1 = 1
warm_loop2 = 0
loop2 = 1
warm_loop3 = 0
loop3 = 1

if (len(sys.argv)>1):
    if len(sys.argv)==7:
        warm_loop1 = int(sys.argv[1])
        loop1 = int(sys.argv[2])
        warm_loop2 = int(sys.argv[3])
        loop2 = int(sys.argv[4])
        warm_loop3 = int(sys.argv[5])
        loop3 = int(sys.argv[6])
    elif len(sys.argv)==3:
        warm_loop1 = int(sys.argv[1])
        loop1 = int(sys.argv[2])
        warm_loop2 = int(sys.argv[1])
        loop2 = int(sys.argv[2])
        warm_loop3 = int(sys.argv[1])
        loop3 = int(sys.argv[2])
    else:
        print(os.path.abspath(__file__)+' wrong args.')
        sys.exit(-1)

results_save_path = os.path.join(root_path,'results/'+model_flag+'/time_split_without_bm_warm_'+str(warm_loop1)+'_'+str(warm_loop2)+'_'+str(warm_loop3)+'_loop_'+str(loop1)+'_'+str(loop2)+'_'+str(loop3)+'.txt')

PB1 = os.path.join(root_path,'model_parts/'+model_flag+'/part_1.pb')
PB2 = os.path.join(root_path,'model_parts/'+model_flag+'/part_2.pb')
PB3 = os.path.join(root_path,'model_parts/'+model_flag+'/part_3.pb')

img = cv2.imread(os.path.join(root_path,"data/aa.JPEG"))
img = np.expand_dims(img,0)

g1_input_tensors = ['image_tensor:0']
g1_output_tensors = ['Preprocessor/sub:0','Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3:0']
g2_input_tensors = ['Preprocessor/sub:0']
g2_output_tensors = ['Squeeze:0','concat_1:0']
g3_input_tensors = ['Preprocessor/sub:0','Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3:0','Squeeze:0','concat_1:0']
g3_output_tensors = ['num_detections:0','detection_classes:0','detection_boxes:0','detection_scores:0']

g1_input_values = [img]

g1_output_values,t1 = run_tf_pb(PB1,g1_input_tensors,g1_input_values,g1_output_tensors,warm_loop1,loop1)

g2_input_values = [g1_output_values[0]]
g3_input_values = g1_output_values

g2_output_values,t2 = run_tf_pb(PB2,g2_input_tensors,g2_input_values,g2_output_tensors,warm_loop2,loop2)
        
for value in g2_output_values:
    g3_input_values.append(value)

g3_output_values,t3 = run_tf_pb(PB3,g3_input_tensors,g3_input_values,g3_output_tensors,warm_loop3,loop3)
        
print('num_detections: ', g3_output_values[0])
print('detection_classes: ', g3_output_values[1])        
print('detection_boxes: ', g3_output_values[2])        
print('detection_scores: ', g3_output_values[3])

print('time1: ', t1)
print('time2: ', t2)
print('time3: ', t3)

with open(results_save_path,'w+') as rsp:
    rsp.write('time1: ' + str(t1)+'\n')
    rsp.write('time2: ' + str(t2)+'\n')
    rsp.write('time3: ' + str(t3)+'\n')


