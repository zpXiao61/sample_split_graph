from lib.split_graph import split_graph
from lib.cfg import root_path
import os

model_flag = 'faster_rcnn_resnet50_coco_2018_01_28'

placeholder_save_path = os.path.join(root_path,'results/'+model_flag+'/placeholders.txt')

path_input_ssd_pb = os.path.join(root_path,'model_whole/'+model_flag+'/frozen_inference_graph.pb')
outpath = os.path.join(root_path,'model_parts/'+model_flag+'/')
name_pb1 = 'tmp_1.pb'
name_pb2 = 'tmp_2.pb'

scopes2 = ['SecondStage']
ps1 = split_graph(path_input_ssd_pb,outpath,name_pb1,name_pb2,scopes_pb2=scopes2,optypes_pb2=None,opnames_pb2=None,savepbtxt=False)

###########################
path_input_ssd_pb = os.path.join(root_path,'model_parts/'+model_flag+'/tmp_1.pb')
outpath = os.path.join(root_path,'model_parts/'+model_flag+'/')
name_tmp = 'tmp.pb'
name_pb1 = 'part_1.pb'
name_pb2 = 'part_2.pb'
name_pb3 = 'part_3.pb'

scopes2 = ['FirstStageFeatureExtractor','BatchMultiClassNonMaxSuppression','map','Decode']
names2 = ['Shape','Shape_1','Shape_4','Shape_5','Reshape_3','CropAndResize','Squeeze','Conv/weights/read']
ps2 = split_graph(path_input_ssd_pb,outpath,name_pb1,name_tmp,scopes_pb2=scopes2,optypes_pb2=None,opnames_pb2=names2,savepbtxt=False)
scopes2 = ['BatchMultiClassNonMaxSuppression','map','Decode']
names2 = ['Shape','Shape_1','Shape_4','Shape_5','Reshape_3','CropAndResize','ExpandDims_1']
ps3 = split_graph(outpath+name_tmp,outpath,name_pb2,name_pb3,scopes_pb2=scopes2,optypes_pb2=None,opnames_pb2=names2,savepbtxt=False)

os.remove(outpath+name_tmp)
os.remove(path_input_ssd_pb)

###########################
path_input_ssd_pb = os.path.join(root_path,'model_parts/'+model_flag+'/tmp_2.pb')
outpath = os.path.join(root_path,'model_parts/'+model_flag+'/')
name_pb1 = 'part_4.pb'
name_pb2 = 'part_5.pb'
scopes2 = ['SecondStagePostprocessor']

ps4 = split_graph(path_input_ssd_pb,outpath,name_pb1,name_pb2,scopes_pb2=scopes2,optypes_pb2=None,opnames_pb2=None,savepbtxt=False)

os.remove(path_input_ssd_pb)
##############################

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
    psp.write('p3:\n')
    print('p3:')
    for p in ps3:
        psp.write(p+'\n')
        print(p)
    psp.write('p4:\n')
    print('p4:')
    for p in ps4:
        psp.write(p+'\n')
        print(p)

