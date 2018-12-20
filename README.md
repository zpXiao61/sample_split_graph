### This project show several samples of tensorflow model splitting.

Dependencies:
* python2/python3
* opencv
* numpy
* tensorflow

## Usage

step1: <br>
> 
if models already exist, skip. <br>
run "sh download_models.sh" <br>
to download sample models and place the pbfiles into model_whole/*/ <br>
cycle-gan: https://github.com/vanhuyz/CycleGAN-TensorFlow/releases/download/v0.1-alpha/apple2orange.pb <br>
ssd and fasterrcnn: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md <br>

step2: <br>
>
run "source env.sh" <br>
to add lib/ to PYTHONPATH <br>

step3: <br> 
>
run "python run_all.py" <br>

## supplementary
all scripts are under the "script/" folder. <br>
for each case: <br>
>
test_split.py will split the model in "model/xxx/yyy.pb" and place the splitted models into "model_parts/xxx/" <br>
test_run_whole.py will do inference using the original pbmodel "model_whole/xxx/yyy.pb" <br>
test_run_split.py will do inference using the splitted pbmodels under "model_parts/xxx/" <br>
