# these are my training notes

|resource            |                 location                                             |
|--------------------|----------------------------------------------------------------------|
|models zoo          | models/research/object_detection/g3doc/detection_model_zoo.md        |
|initial used model  | faster_rcnn_inception_v2_coco_2017_11_08                             |
|initial config file | models/research/object_detection/samples/configs/faster_rcnn_inception_v2_coco.config |
|initial model tar   | http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2017_11_08.tar.gz |

- create 'training' folder
- copy the config file into the 'training' folder
- create 'training/data' folder
- copy generated 'train.record' in 'training/data' folder
- download and extract faster_rcnn_inception_v2_coco_2017_11_08.tar.gz file somewhare
- create 'training/models' folder
- move the 3 extracted ckpt files in 'training/models'
- create 'training/models/train'
