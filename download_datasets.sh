#!/bin/bash
if [ ! -f ./dataset/intel-image-classification.zip ]; then
    mkdir dataset
    curl -L -o ./dataset/intel-image-classification.zip\
     https://www.kaggle.com/api/v1/datasets/download/puneet6060/intel-image-classification
fi

cd dataset
unzip ./intel-image-classification.zip
mv ./intel-image-classification/* ./

if [ ! -f ./dataset_1024/intel-image-classification.zip ]; then
    mkdir dataset_1024
    #!/bin/bash
    curl -L -o ./dataset_1024/celebahq.zip\
      https://www.kaggle.com/api/v1/datasets/download/lamsimon/celebahq
fi

cd dataset_1024
unzip ./celebahq.zip
mv ./celebahq/train/male/* ./
mv ./celebahq/train/female/* ./
mv ./celebahq/val/male/* ./
mv ./celebahq/val/female/* ./
