import os

BATCHSIZE = 64
SIZE = 448
HEIGHT = SIZE
WIDTH = SIZE
N_CHANNEL = 3
INPUT_SHAPE = (HEIGHT, WIDTH, N_CHANNEL)
MOMENTUM = 0.9
LAMBDA_COORD = 5
LAMBDA_NOBJ = 0.5
S = 7
B = 2
C = 20

PATH_BASE = '/content'
DIR_VOC = 'VOCdevkit'
DIR_YEAR = 'VOC2012'
PATH_VOC2012 = os.path.join(PATH_BASE, DIR_VOC, DIR_YEAR)

DIR_ANNOTATION = 'Annotations'
DIR_IMAGES = 'JPEGImages'
DIR_IMAGESETS = 'ImageSets'
DIR_TASK = 'Main'

#position of class labels sorted ascending by
#the num. of images containing that class
PASCAL_VOC_SORT_IDX = (12, 7, 14, 5, \
                        13, 2, 17, 15, \
                        16, 0, 8, 18, \
                        3, 6, 19, 9, \
                        1, 4, 10, 11)

PASCAL_VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', \
                      'bottle', 'bus', 'car', 'cat', \
                      'chair', 'cow', 'diningtable', 'dog', \
                      'horse', 'motorbike', 'person', 'pottedplant', \
                      'sheep', 'sofa', 'train', 'tvmonitor')


DICT_CLASSES = dict(zip(PASCAL_VOC_CLASSES, range(len(PASCAL_VOC_CLASSES))))

assert C == len(PASCAL_VOC_CLASSES), "Differing number of classes"

