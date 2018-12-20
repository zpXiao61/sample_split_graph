from lib.split_graph import split_graph
from lib.cfg import root_path
import os

model_flag = 'ssd_mobilenet_v2_coco_2018_03_29'

path_input_ssd_pb = os.path.join(root_path,'model_whole/'+model_flag+'/frozen_inference_graph.pb')
outpath = os.path.join(root_path,'model_parts/'+model_flag+'/')
name_tmp = 'tmp.pb'
name_pb1 = 'part_1.pb'
name_pb2 = 'part_2.pb'
name_pb3 = 'part_3.pb'

placeholder_save_path = os.path.join(root_path,'results/'+model_flag+'/placeholders.txt')

# scopes of ssd_2
scopes2 = ['FeatureExtractor','BoxPredictor']
# scopes of ssd_3
scopes3 = ['Postprocessor','MultipleGridAnchorGenerator']

ps1 = split_graph(path_input_ssd_pb,outpath,name_tmp,name_pb3,scopes_pb2=scopes3,optypes_pb2=None,opnames_pb2=None,savepbtxt=False)

ps2 = split_graph(outpath+name_tmp,outpath,name_pb1,name_pb2,scopes_pb2=scopes2)

with open(placeholder_save_path,'w+') as psp:
    psp.write('p1:\n')
    print('p1:')
    for p in ps1:
        psp.write(p+'\n')
        print(p)
    psp.write('p2:\n')
    print('p2:')
    for p in ps2:
        psp.write(p+'\n')
        print(p)

import os 
os.remove(outpath+name_tmp)
print('tmp.pb removed.')

'''
ssd res50 fpn:
ssd_1:
  input: 'image_tensor:0'
  output: 'Preprocessor/sub:0','Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3:0'
ssd_2:
  input: 'image_tensor:0'
  output: 'concat:0','concat_1:0'
ssd_3:
  input: 'Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3:0','concat:0','concat_1:0'
  output: 'num_detections:0','detection_classes:0','detection_boxes:0','detection_scores:0'
'''