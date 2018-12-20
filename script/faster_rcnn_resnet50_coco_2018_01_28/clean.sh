model_flag=faster_rcnn_resnet50_coco_2018_01_28

cd model_parts/${model_flag}/
rm *.pb

cd ../../results/${model_flag}/
rm *