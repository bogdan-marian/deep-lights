port forword example for jupyter
```
ssh -N -f -L localhost:8888:localhost:8888 bogdan_oloeriu@35.205.55.124
```

copy from gcloud buckets
```
gsutil cp gs://[BUCKET_NAME]/[OBJECT_NAME] [OBJECT_DESTINATION]
gsutil cp gs://deep-learning-oloeriu/*.tar.gz ./
```
