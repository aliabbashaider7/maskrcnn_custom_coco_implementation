# MaskRCNN Custom Training:
MaskRCNN implementation on custom dataset labelled by labelme

# Pre-requisits:
The code is tested on ubuntu 20.04 with python3.6 (recommended)

sudo apt-get install python3.6-dev

pip3 install -r requirements.txt

Note: UserWarnings during training can be suppressed with a little adjustment in the versions according to requirement.

# Training
After setting up the environment. Make sure you have your labelme dataset in format of dataset directory in the repo.
running data_convert.py will create a data folder having coco format to train maskrcnn. run train.py to begin training.

Note: Be sure to change number of classes in CocoConfid class inside of train.py and download pretrained weights from
[here](https://drive.google.com/file/d/1QAr9cK2ZirhiYXR6bM_jfi_-o-TTk8nv/view?usp=sharing)
before you hit training.
Trained weights will be saved under logs directory.

# Inference:
Run test.py with correct model path, image path and of course a list of class names you trained.
An opencv window pops up showing masks, corresponding bounding boxes and labels as the results.
