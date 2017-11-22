# these are my training notes

|resource            |                 location                                             |
|--------------------|----------------------------------------------------------------------|
|instalation.md      | models/research/object_detection/g3doc/                              |
|models zoo          | models/research/object_detection/g3doc/detection_model_zoo.md        |
|initial used model  | faster_rcnn_inception_v2_coco_2017_11_08                             |
|initial config file | models/research/object_detection/samples/configs/faster_rcnn_inception_v2_coco.config |
|initial model tar   | http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2017_11_08.tar.gz |
|initial train.py    | models/research/object_detection/train.py                            |
|tensorboard         | an utility that gets installed when you install tensorflow           |
|export_inference_graph.py| models/research/object_detection/export_inference_graph.py      |
|labelImg            | https://github.com/tzutalin/labelImg.git                             |



- create 'training' folder
- copy the config file into the 'training' folder
- create 'training/data' folder
  - copy generated 'train.record' and testing.record in 'training/data' folder
  - copy bosh_label_map.pbtxt in 'training/data' folder
- edit the config file so that it points to the new training and testing record and lable_map
- download and extract faster_rcnn_inception_v2_coco_2017_11_08.tar.gz file somewhare
- create 'training/models' folder
- move the 3 extracted ckpt files in 'training/models'
- create 'training/models/train'
- copy 'train.py' to 'training'

- start training
```
python train.py --logtostderr \
  --train_dir=./models/train \
  --pipeline_config_path=faster_rcnn_inception_v2_bosh.config
```

- use tensorboard to analyze the process from inside the training folder
```
tensorboard --logdir=models/train/
```
- copy export_inference_graph.py to 'training'
- export the inference graph specific to a checkpoint
```
python export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path ./faster_rcnn_inception_v2_bosh.config \
  --trained_checkpoint_prefix ./models/train/model.ckpt-19478 \
  --output_directory ./inference_graps
```

- copy export_inference_graph.py to 'training' folder

- install training data generator utils. from /models/research folder
```
python setup.py install
```
