#!/bin/bash
if [ ! -f ./dataset_150x150/intel-image-classification.zip ]; then
    mkdir dataset_150x150
    curl -L -o ./dataset_150x150/intel-image-classification.zip\
     https://www.kaggle.com/api/v1/datasets/download/puneet6060/intel-image-classification
fi

cd dataset_150x150
unzip ./intel-image-classification.zip
mv ./seg_pred/seg_pred/* ./
mv ./seg_test/seg_test/buildings/* ./
mv ./seg_test/seg_test/forest/* ./
mv ./seg_test/seg_test/glacier/* ./
mv ./seg_test/seg_test/mountain/* ./
mv ./seg_test/seg_test/sea/* ./
mv ./seg_test/seg_test/street/* ./

mv ./seg_train/seg_train/buildings/* ./
mv ./seg_train/seg_train/forest/* ./
mv ./seg_train/seg_train/glacier/* ./
mv ./seg_train/seg_train/mountain/* ./
mv ./seg_train/seg_train/sea/* ./
mv ./seg_train/seg_train/street/* ./

rm -r ./seg_pred/
rm -r ./seg_test/
rm -r ./seg_train/

cd ..
if [ ! -f ./dataset_1024x1024/celebahq.zip ]; then
    mkdir dataset_1024x1024
    #!/bin/bash
    curl -L -o ./dataset_1024x1024/celebahq.zip\
      https://www.kaggle.com/api/v1/datasets/download/lamsimon/celebahq
fi

cd dataset_1024x1024
unzip ./celeba_hq.zip
mv ./celeba_hq/train/male/* ./
mv ./celeba_hq/train/female/* ./
mv ./celeba_hq/val/male/* ./
mv ./celeba_hq/val/female/* ./

rm -r celeba_hq