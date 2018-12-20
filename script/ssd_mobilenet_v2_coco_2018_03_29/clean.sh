model_flag=ssd_mobilenet_v2_coco_2018_03_29

cd model_parts/${model_flag}/
rm *.pb

cd ../../results/${model_flag}/
rm *