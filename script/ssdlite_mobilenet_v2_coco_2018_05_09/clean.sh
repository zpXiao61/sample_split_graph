model_flag=ssdlite_mobilenet_v2_coco_2018_05_09

cd model_parts/${model_flag}/
rm *.pb

cd ../../results/${model_flag}/
rm *