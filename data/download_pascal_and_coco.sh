
##### Download Pascal VOC 2012 dataset and add trainaug split.
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip

tar -xf VOCtrainval_11-May-2012.tar
unzip SegmentationClassAug.zip -d VOCdevkit/VOC2012

mv trainaug.txt VOCdevkit/VOC2012/ImageSets/Segmentation
mv VOCdevkit/VOC2012/SegmentationClassAug/* VOCdevkit/VOC2012/SegmentationClass/

rm -r VOCdevkit/VOC2012/__MACOSX
rm SegmentationClassAug.zip
rm VOCtrainval_11-May-2012.tar

###### Download COCO 2017 dataset.
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

mkdir -p COCO/annotations
mkdir -p COCO/images
mv annotations COCO/
mv train2017 COCO/images
mv val2017 COCO/images

rm -r annotations_trainval2017.zip
rm train2017.zip
rm val2017.zip



