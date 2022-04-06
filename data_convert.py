import os
import labelme2coco

try:
    os.mkdir('data')
    os.mkdir('data/coco')
except:
    pass

train_dir = 'dataset/train'
val_dir = 'dataset/val'

train_annotations = 'data/coco/annotations/instances_train2017.json'
val_annotations = 'data/coco/annotations/instances_val2017.json'

labelme2coco.convert(train_dir, train_annotations)
labelme2coco.convert(val_dir, val_annotations)

try:
    os.mkdir('data/coco/train2017')
    os.mkdir('data/coco/train2017/dataset')
except:
    pass
try:
    os.mkdir('data/coco/val2017')
    os.mkdir('data/coco/val2017/dataset')
except:
    pass

os.system('cp -r dataset/train data/coco/train2017/dataset')
os.system('cp -r dataset/val data/coco/val2017/dataset')