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
jupyter notebook --allow-root
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
