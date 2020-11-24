"""
Written by: Rahmad Sadli
Website : https://machinelearningspace.com

I finally made this program simple and readable
Hopefully, this program will help some beginners like me to understand better object detection.
If you want to redistribute it, just keep the author's name.

In oder to execute this program, you need to install TensorFlow 2.0 and opencv 4.x

For more details about how this program works. I explained well about it, just click the link below:
https://machinelearningspace.com/the-beginners-guide-to-implementing-yolo-v3-in-tensorflow-2-0-part-1/

Credit to:
Ayoosh Kathuria who shared his great work using pytorch, really appreaciated it.
https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
"""

import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import numpy as np
from yolov3 import YOLOv3Net
from imutils import paths
from pathlib import Path
import csv

# TODO: I do not know what this part does but commenting out this part made the code work
# now do this on images in multiple images in euro and ug2 dataset - the way Hog happened
# should classify what the difference in accuracy this implementation has over HOG to
# support why you went with this. Probably might not tell the whole story b/c HOG would have good
# performance on items with occlusion but not on other types of noise but the reverse true for yolo
# should think about metric and producing graphs early


# If you don't have enough GPU hardware device available in your machine, uncomment the following three lines:
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_size = (416, 416, 3)
num_classes = 80
class_name = './data/coco.names'
max_output_size = 40
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.5

cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'
img_filename = "data/images/test.jpg"

clearpath = r'D:\Computer Vision Project\GRAZ DATASET\Graz_01\persons\persons'

noisypath = r'D:\Computer Vision Project\UG Dataset\RTTS\JPEGImages'
europath = r'D:\Computer Vision Project\Eurocity Dataset\ECP_day_img_train\ECP\day\img\train\amsterdam'


# ug2path =

def main():
    model = YOLOv3Net(cfgfile, model_size, num_classes)
    model.load_weights(weightfile)

    class_names = load_class_names(class_name)

    # loop over the image paths
    for imagePath in paths.list_images(europath):
        image = cv2.imread(imagePath)
        # image = cv2.imread(img_filename)
        image = np.array(image)
        image = tf.expand_dims(image, 0)

        resized_frame = resize_image(image, (model_size[0], model_size[1]))
        pred = model.predict(resized_frame)

        boxes, scores, classes, nums = output_boxes(
            pred, model_size,
            max_output_size=max_output_size,
            max_output_size_per_class=max_output_size_per_class,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold)

        image = np.squeeze(image)
        img = draw_outputs(image, boxes, scores, classes, nums, class_names)
        # TODO: need to write this part out into csv file to calculate mAP later for a baseline
        # Reference: https://stackoverflow.com/questions/48343678/store-tensorflow-object-detection-api-image-output-with-boxes-in-csv-format
        # print(boxes)
        # get dimensions of image
        dimensions = image.shape
        height = image.shape[0]
        width = image.shape[1]
        box2 = [0, 0, 0, 0]
        i = 0
        box2es = list()
        for box in np.squeeze(boxes):
            # The coordinates in the boxes array ([ymin, xmin, ymax, xmax]) are normalized
            box2[0] = box[0] * height  # ymin
            box2[1] = box[1] * width  # xmin
            box2[2] = box[2] * height  # ymax
            box2[3] = box[3] * width  # xmax
            # to put them in same order as the actual label box dimensions
            #  fieldnames = ['path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
            print("box2 is ", [imagePath, box2[1], box2[0], box2[3], box2[2], class_names[i]])
            if (str(class_names[i]) == str("person")) and (int(box2[0]) != 0 or int(box2[1]) != 0 or int(box2[2]) != 0 or int(box2[3]) != 0) :
                box2es.append([imagePath, box2[1], box2[0], box2[3], box2[2], class_names[i]])
            i += 1

            yolo_pred_train = Path("D:/Computer Vision Project/Eurocity Dataset/OUTPUT/Train/yolo_pred_train.csv")

        with open(yolo_pred_train, 'a+', newline='') as csvfile:
            fieldnames = ['path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
            writer = csv.writer(csvfile, delimiter=',')
            # writer.writeheader(fieldnames)
            writer.writerows(box2es)



        # np.savetxt(yolo_pred_train, box2, delimiter=',')

        win_name = 'Image detection'
        cv2.imshow(win_name, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    # If you want to save the result, uncommnent the line below:
    # cv2.imwrite('test.jpg', img)


if __name__ == '__main__':
    main()
