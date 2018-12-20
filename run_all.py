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
    cmd_split = 'python script/'+model_flag+'/test_split.py'
    cmd_run_whole = 'python script/'+model_flag+'/test_run_whole.py 0 1'
    cmd_run_split = 'python script/'+model_flag+'/test_run_split.py 0 1'

    
    print('#########################\n'+cmd_split)
    os.system(cmd_split)
    
    print('#########################\n'+cmd_run_whole)
    os.system(cmd_run_whole)
    
    print('#########################\n'+cmd_run_split)
    os.system(cmd_run_split)
    
    
    