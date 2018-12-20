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

model_flag = 'faster_rcnn_resnet50_coco_2018_01_28'

warm_loop1 = 0
loop1 = 1
warm_loop2 = 0
loop2 = 1
warm_loop3 = 0
loop3 = 1
warm_loop4 = 0
loop4 = 1
warm_loop5 = 0
loop5 = 1

if (len(sys.argv)>1):
    if len(sys.argv)==11:
        warm_loop1 = int(sys.argv[1])
        loop1 = int(sys.argv[2])
        warm_loop2 = int(sys.argv[3])
        loop2 = int(sys.argv[4])
        warm_loop3 = int(sys.argv[5])
        loop3 = int(sys.argv[6])
        warm_loop4 = int(sys.argv[7])
        loop4 = int(sys.argv[8])
        warm_loop5 = int(sys.argv[9])
        loop5 = int(sys.argv[10])
    elif len(sys.argv)==3:
        warm_loop1 = int(sys.argv[1])
        loop1 = int(sys.argv[2])
        warm_loop2 = int(sys.argv[1])
        loop2 = int(sys.argv[2])
        warm_loop3 = int(sys.argv[1])
        loop3 = int(sys.argv[2])
        warm_loop4 = int(sys.argv[1])
        loop4 = int(sys.argv[2])
        warm_loop5 = int(sys.argv[1])
        loop5 = int(sys.argv[2])
    else:
        print(os.path.abspath(__file__)+' wrong args.')
        sys.exit(-1)

results_save_path = os.path.join(root_path,'results/'+model_flag+'/time_split_without_bm_warm_'+str(warm_loop1)+'_'+str(warm_loop2)+'_'+str(warm_loop3)+'_loop_'+str(loop1)+'_'+str(loop2)+'_'+str(loop3)+'.txt')

PB1 = os.path.join(root_path,'model_parts/'+model_flag+'/part_1.pb')
PB2 = os.path.join(root_path,'model_parts/'+model_flag+'/part_2.pb')
PB3 = os.path.join(root_path,'model_parts/'+model_flag+'/part_3.pb')
PB4 = os.path.join(root_path,'model_parts/'+model_flag+'/part_4.pb')
PB5 = os.path.join(root_path,'model_parts/'+model_flag+'/part_5.pb')

img = cv2.imread(os.path.join(root_path,"data/aa.JPEG"))
img = np.expand_dims(img,0)

g1_input_tensors = ['image_tensor:0']
g1_output_tensors = ['Preprocessor/sub:0',
                     'Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3:0']
g2_input_tensors = ['Preprocessor/sub:0']
g2_output_tensors = ['Squeeze:0',
                    'FirstStageBoxPredictor/concat_1:0',
                    'FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0']
g3_input_tensors = ['Preprocessor/sub:0',
                    'Squeeze:0',
                    'FirstStageBoxPredictor/concat_1:0',
                    'FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0']
g3_output_tensors = ['BatchMultiClassNonMaxSuppression/map/TensorArrayStack_4/TensorArrayGatherV3:0',
                     'map_1/TensorArrayStack/TensorArrayGatherV3:0',
                     'MaxPool2D/MaxPool:0']

g4_input_tensors = ['MaxPool2D/MaxPool:0']
g4_output_tensors = ['Squeeze_2:0','Squeeze_3:0']

g5_input_tensors = ['Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3:0',
                     'BatchMultiClassNonMaxSuppression/map/TensorArrayStack_4/TensorArrayGatherV3:0',
                     'map_1/TensorArrayStack/TensorArrayGatherV3:0','Squeeze_2:0','Squeeze_3:0']

g5_output_tensors = ['num_detections:0','detection_classes:0','detection_boxes:0','detection_scores:0']

g1_input_values = [img]

g1_output_values,t1 = run_tf_pb(PB1,g1_input_tensors,g1_input_values,g1_output_tensors,warm_loop1,loop1)

g2_input_values = [g1_output_values[0]]
g3_input_values = [g1_output_values[0]]
g5_input_values = [g1_output_values[1]]

g2_output_values,t2 = run_tf_pb(PB2,g2_input_tensors,g2_input_values,g2_output_tensors,warm_loop2,loop2)

for v in g2_output_values:
    g3_input_values.append(v)

g3_output_values,t3 = run_tf_pb(PB3,g3_input_tensors,g3_input_values,g3_output_tensors,warm_loop3,loop3)

g4_input_values = [g3_output_values[2]]

g4_output_values,t4 = run_tf_pb(PB4,g4_input_tensors,g4_input_values,g4_output_tensors,warm_loop4,loop4)

g5_input_values.append(g3_output_values[0])
g5_input_values.append(g3_output_values[1])
g5_input_values.append(g4_output_values[0])
g5_input_values.append(g4_output_values[1])

g5_output_values,t5 = run_tf_pb(PB5,g5_input_tensors,g5_input_values,g5_output_tensors,warm_loop5,loop5)
        
print('num_detections: ', g5_output_values[0])
print('detection_classes: ', g5_output_values[1])        
print('detection_boxes: ', g5_output_values[2])        
print('detection_scores: ', g5_output_values[3])

print('time1: ', t1)
print('time2: ', t2)
print('time3: ', t3)
print('time3: ', t4)
print('time3: ', t5)

with open(results_save_path,'w+') as rsp:
    rsp.write('time1: ' + str(t1)+'\n')
    rsp.write('time2: ' + str(t2)+'\n')
    rsp.write('time3: ' + str(t3)+'\n')
    rsp.write('time4: ' + str(t4)+'\n')
    rsp.write('time5: ' + str(t5)+'\n')

