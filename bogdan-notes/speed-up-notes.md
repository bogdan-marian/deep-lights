#### what is this

This is just a bunch of startup notes that i kept keeping arownd while working
on creating the deep-light model for Udacity. I ended up like doing hundreds of
steps and this sheet turned out to be a real time server.

---
mount the windows drive and then move there
```
cd /media/bogdan/Programare/Bogdan

```

start nvidia-docker container
```
nvidia-docker run -it --rm -p 8888:8888 -v `pwd`:/src tensorflow/tensorflow:latest-gpu bash
```

update tensorflow in the container and then save the container
```
docker ps
docker commit 5e0dfd671ae6  tensorflow/tensorflow:bogdan_v1
docker images
```
do some more work
```

docker commit 5e0dfd671ae6 tensorflow/tensorflow:bogdan-v2
```

current functional image
```
nvidia-docker run -it --rm -p 8888:8888 -v `pwd`:/src tensorflow/tensorflow:bogdan-v2 bash
```

retag image
```
docker tag tensorflow/tensorflow:bogdan-v2 bogdanoloeriu/tensorflow:bogdan-v2

$ docker push bogdanoloeriu/tensorflow:bogdan-v2

nvidia-docker run -it --rm -p 8888:8888 -v `pwd`:/src bogdanoloeriu/tensorflow:bogdan-v2 bash

```
model zoo location
```
/models/research/object_detection/g3doc/detection_model_zoo.md
```
connect new shell to running docker instance
```
sudo docker exec -i -t 665b4a1e17b6 /bin/bash #by ID
```
- install training data generator utils. from /models/research folder
```
python setup.py install
```
lounch jupyter as root
```
jupyter notebook --allow-root
```

tensorflow initialization
From tensorflow/models/research/
```
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
```
