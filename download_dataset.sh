#!/bin/bash
if [ ! -f ./dataset/intel-image-classification.zip ]; then
    mkdir dataset
    curl -L -o ./dataset/intel-image-classification.zip\
     https://www.kaggle.com/api/v1/datasets/download/puneet6060/intel-image-classification
fi

cd dataset
unzip ./intel-image-classification.zip
mv ./intel-image-classification/* ./
