# Reference : https://towardsdatascience.com/object-detection-on-aerial-imagery-using-retinanet-626130ba2203
# Modified the above code a lot to work for JSON coco annotations and EUROCITY instead of XML annotation

# *** Python script to create the train/test set ***

# import the necessary packages
# from config import esri_retinanet_config as config
# from bs4 import BeautifulSoup
from imutils import paths
import random
import os
import json
import pprint
from pathlib import Path
import csv

annot_train_path = Path("D:/Computer Vision Project/Eurocity Dataset/ECP_day_labels_train/ECP/day/labels/train")
annot_val_path = Path("D:/Computer Vision Project/Eurocity Dataset/ECP_day_labels_val/ECP/day/labels/val")

train_jsons = Path("D:/Computer Vision Project/Eurocity Dataset/ECP_day_img_train/ECP/day/img/train")
val_jsons = Path("D:/Computer Vision Project/Eurocity Dataset/ECP_day_img_val/ECP/day/img/val")

''' 
# Create easy variable names for all the arguments
annot_train_path = 'D:/Computer Vision Project/Eurocity Dataset/ECP_day_labels_train/ECP/day/labels/train'
train_jsons = 'D:/Computer Vision Project/Eurocity Dataset/ECP_day_img_train/ECP/day/img/train'
# For me this is the validation images because the test data for Eurocity did not have labels
# val_jsons = "D:\Computer Vision Project\Eurocity Dataset\ECP_day_img_val\ECP\day\img\val"
val_jsons = 'D:/Computer Vision Project/Eurocity Dataset/ECP_day_img_val/ECP/day/img/val'
annot_val_path = 'D:/Computer Vision Project/Eurocity Dataset/ECP_day_labels_val/ECP/day/labels/val'
'''

# print(train_jsons)

trainImagePaths = list(paths.list_files(train_jsons))
valImagePaths = list(paths.list_files(val_jsons))

trainImageLabels = list(paths.list_files(annot_train_path))
valImageLabels = list(paths.list_files(annot_val_path))
# for a in trainImageLabels:
#    print(a)

train_csv_path_out = Path("D:/Computer Vision Project/Eurocity Dataset/OUTPUT/Train")
val_csv_path_out = Path("D:/Computer Vision Project/Eurocity Dataset/OUTPUT/Validation")
# create the list of datasets to build
dataset = [("train", trainImagePaths, train_csv_path_out),
           ("test", valImagePaths, val_csv_path_out)]

# initialize the set of classes we have
CLASSES = set()

count = 0
# loop over the datasets
for (dType, imagePaths, outputCSV) in dataset:
    count += 1
    if count > 1:
        # TODO: only one to do for training for now
        break
    # load the contents
    print("[INFO] creating '{}' set...".format(dType))
    print("[INFO] {} total images in '{}' set".format(len(imagePaths), dType))

    # open the output CSV file
    # csv = open(outputCSV, "r")
    # print("length of imagepath", len(imagePaths))

    # loop over the image paths
    k = 0
    outputCSV = Path(str(train_csv_path_out) + "/train.csv")
    objectsInImage = list()
    classes_csv = Path(str(train_csv_path_out) + "/classes.csv")

    for imagePath in imagePaths:

        # open the output CSV file
        # csv = open(outputCSV, "w")

        '''
         Loop over each dataset (train and test) and open the output CSV file to be written.
         For each dataset, we loop over each image path. For each image, extract the filename 
         and build the corresponding annotation path. This works out because, usually, the image 
         and annotation files have the same name but different extensions
        '''

        # build the corresponding annotation path
        fname = imagePath.split(os.path.sep)[-1]
        fname = "{}.json".format(fname[:fname.rfind(".")])
        # annotPath = os.path.sep.join([annot_train_path, fname])
        # print("imagepath: ", imagePath)
        # print(fname)

        # Already can created above :
        trainImageLabels
        valImageLabels

        # extract the image dimensions
        w = 0
        h = 0
        label = list()
        xMin = list()
        yMin = list()
        xMax = list()
        yMax = list()

        # for jsonlabelfile in trainImageLabels:
        # print(jsonlabelfile)
        with open(trainImageLabels[k], 'r') as COCO:
            coco = json.loads(COCO.read())
            imageHeight = int(json.dumps(coco['imageheight']))
            imageWidth = int(json.dumps(coco['imagewidth']))
            # print(imageHeight)
            if imageHeight != h or imageWidth != w:
                h = imageHeight
                w = imageWidth
            ''' For every image, find all the objects and iterate over each one of them. Then, find the bounding box (
            xmin, ymin, xmax, ymax) and the class label (name) for each object in the annotation. Do a cleanup by 
            truncating any bounding box coordinate that falls outside the boundaries of the image. Also, do a sanity 
            check if, by error, any minimum value is larger than the maximum value and vice-versa. If we find such 
            values, we will ignore these objects and continue to the next one. 
            '''

            # loop over all child elements
            # print("line 97 ", len(coco['children']))
            for i in range(len(coco['children'])):
                # pprint.pprint(coco['children'][i]['identity'])

                label_i = coco['children'][i]['identity']
                xMin_i = int(float(coco['children'][i]['y0']))
                yMin_i = int(float(coco['children'][i]['x1']))
                xMax_i = int(float(coco['children'][i]['x1']))
                yMax_i = int(float(coco['children'][i]['y1']))

                # truncate any bounding box coordinates that fall outside
                # the boundaries of the image
                xMin_i = max(0, xMin_i)
                yMin_i = max(0, yMin_i)
                xMax_i = min(w, xMax_i)
                yMax_i = min(h, yMax_i)

                ''' 
                # ignore the bounding boxes where the minimum values are larger
                # than the maximum values and vice-versa due to annotation errors
                if xMin_i >= xMax_i or yMin_i >= yMax_i:
                    continue
                elif xMax_i <= xMin_i or yMax_i <= yMin_i:
                    continue
             
                label.append(label_i)
                xMin.append(xMin_i)
                yMin.append(yMin_i)
                xMax.append(xMax_i)
                yMax.append(yMax_i)
                '''
                # update the set of unique class labels
                CLASSES.add(label_i)

                # File will hold all the annotations for training in the following format:
                # <path/to/image>,<xmin>,<ymin>,<xmax>,<ymax>,<label>
                row = [imagePath, str(xMin_i), str(yMin_i), str(xMax_i),
                       str(yMax_i), str(label_i)]
                objectsInImage.append(row)
                print("row is ", row)

                # csv = open(outputCSV, "w")
            k += 1

        # print("objects ", objectsInImage)
        # print("out: ", outputCSV)

    with open(outputCSV, 'w', newline='') as csvfile:
        fieldnames = ['path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
        writer = csv.writer(csvfile, delimiter=',')

        # writer.writeheader(fieldnames)
        writer.writerows(objectsInImage)


        # break
        # break
        # break
    print("unique classes ", list(CLASSES))
    # write the classes to file
    print("[INFO] writing classes...")
    with open(classes_csv, 'w', newline='') as csvfile:
        # rows = [",".join([c, str(i)]) for (i, c) in enumerate(CLASSES)]
        writer = csv.writer(csvfile)
        for item in list(CLASSES):
            writer.writerow([item])

    print("end")

# close the CSV file
# csv.close()
