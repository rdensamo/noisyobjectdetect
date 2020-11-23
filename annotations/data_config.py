# Reference: https://towardsdatascience.com/object-detection-on-aerial-imagery-using-retinanet-626130ba2203

"""
 write a config file that will hold the paths to images,
 annotations, output CSVs — train, test, and classes, and the test-train split value.
 Having such a config file makes the code versatile for use with different datasets.
"""

# import the necessary packages
import os

# initialize the base path for the logos dataset
BASE_PATH = "dataset"

# build the path to the annotations and input images
ANNOT_PATH = os.path.sep.join([BASE_PATH, 'annotations'])
IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images'])

# degine the training/testing split
TRAIN_TEST_SPLIT = 0.75

#  build the path to the output training and test .csv files
TRAIN_CSV = os.path.sep.join([BASE_PATH, 'train.csv'])
TEST_CSV = os.path.sep.join([BASE_PATH, 'test.csv'])

# build the path to the output classes CSV files
CLASSES_CSV = os.path.sep.join([BASE_PATH, 'classes.csv'])

# build the path to the output predictions dir
OUTPUT_DIR = os.path.sep.join([BASE_PATH, 'predictions'])