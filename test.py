import cv2
import numpy as np
from skimage.measure import find_contours
import mrcnn.model as modellib
from train import CocoConfig

class InferenceConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

def display_instances(image,boxes, masks, class_ids, class_names,
                      scores=None):

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    ori_img = image.copy()

    for i in range(N):
        box = boxes[i]
        x1,y1,x2,y2 = box[0], box[1], box[2], box[3]
        mask = masks[:, :, i]
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        contours_cv = []
        for c in contours:
            for j in c:
                contours_cv.append([int(j[1]), int(j[0])])
        final_contours =  np.array([contours_cv])
        color = np.random.randint(0, 255, size=(3,))
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.drawContours(ori_img, final_contours, 0, color, 2)
        cv2.rectangle(ori_img, (y1,x1), (y2,x2), color, 1)
        cv2.putText(ori_img, f'{class_names[int(class_ids[i])-1]} {scores[i]}', (y1,x1),cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1)
    return ori_img

MODEL_DIR = 'logs/'
COCO_MODEL_PATH='path/to/h5/model'

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

"""
COCO Class names:
Adjust Accordingly... 
Make sure to keep the List Length equal to
the number of Classes trained
"""
class_names = ["person", "car"]

if __name__ == '__main__':
    path = 'path/to/image'
    image = cv2.imread(path)
    results = model.detect([image], verbose=1)
    r = results[0]
    image_result = display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    cv2.imshow('Frame', image_result)
    cv2.waitKey()