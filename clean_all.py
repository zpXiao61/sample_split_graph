import os
from lib.cfg import root_path

model_flags = []
model_list_file = os.path.join(root_path,'model_list')

with open(model_list_file,'r') as mll:
    for line in mll:
        line = line.strip()
        line = line.split(' ')
        if line[1] == '1':
            model_flags.append(line[0])

for model_flag in model_flags:
    cmd = 'sh script/'+model_flag+'/clean.sh'
    os.system(cmd)