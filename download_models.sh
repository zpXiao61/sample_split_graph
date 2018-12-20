current_folder=`pwd`

cd ${current_folder}/model_whole/cycle-gan
wget https://github.com/vanhuyz/CycleGAN-TensorFlow/releases/download/v0.1-alpha/apple2orange.pb

download(){
  cd $1/model_whole/$2
  wget http://download.tensorflow.org/models/object_detection/$2.tar.gz
  tar -zxvf ${2}.tar.gz
  rm ${2}.tar.gz
  mv ${2}/frozen_inference_graph.pb ./
  rm -rf $2
}

download $current_folder faster_rcnn_resnet50_coco_2018_01_28
download $current_folder ssd_inception_v2_coco_2018_01_28
download $current_folder ssd_mobilenet_v2_coco_2018_03_29
download $current_folder ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
download $current_folder ssdlite_mobilenet_v2_coco_2018_05_09
