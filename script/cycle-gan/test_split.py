from lib.split_graph import split_graph
from lib.cfg import root_path
import os

model_flag = 'cycle-gan'

path_input_ssd_pb = os.path.join(root_path,'model_whole/'+model_flag+'/apple2orange.pb')
outpath = os.path.join(root_path,'model_parts/'+model_flag+'/')

name_pb1 = 'part_1.pb'
name_pb2 = 'part_2.pb'

placeholder_save_path = os.path.join(root_path,'results/'+model_flag+'/placeholders.txt')

scopes2 = ['map_4']

ps1 = split_graph(path_input_ssd_pb,outpath,name_pb1,name_pb2,scopes_pb2=scopes2,optypes_pb2=None,opnames_pb2=None,savepbtxt=False)

with open(placeholder_save_path,'w+') as psp:
    psp.write('p1:\n')
    print('p1:')
    for p in ps1:
        psp.write(p+'\n')
        print(p)

