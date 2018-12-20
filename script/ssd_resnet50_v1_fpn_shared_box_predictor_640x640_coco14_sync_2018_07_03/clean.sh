model_flag=ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03

cd model_parts/${model_flag}/
rm *.pb

cd ../../results/${model_flag}/
rm *