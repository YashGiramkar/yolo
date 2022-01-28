import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tensorflow import keras

from yolov1tiny.globalConst import *

def getAnnotations(xmlFile):
    path_annotation = os.path.join(PATH_VOC2012, DIR_ANNOTATION)
    tree = ET.parse(os.path.join(path_annotation, xmlFile))
    root = tree.getroot()
    filename = root.find('filename').text
  
    size = root.find('size')
    size = [int(size.find('width').text), int(size.find('height').text), int(size.find('depth').text)]
  
    bndboxes = []
    classes = []
    objects = root.findall('object')
    for obj in objects:
        objclass = obj.find("name")
        classes.append(objclass.text)

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bndboxes.append([xmin, ymin, xmax, ymax])

    return filename, size, classes, bndboxes

def getClassObjDict(dir_task=os.path.join(PATH_VOC2012, DIR_IMAGESETS, DIR_TASK)):
    trainDict = {key : [] for key in PASCAL_VOC_CLASSES}
    valDict = {key : [] for key in PASCAL_VOC_CLASSES}
    
    annotationExt = '.xml'
    for fname in os.listdir(dir_task):
        try:
            name, ext = fname.split('.')
            classname, train_test = name.split('_')
            if classname not in PASCAL_VOC_CLASSES:
                print("W: classname ", classname)
                continue
        except:
            print("W: fname ", fname)
            continue
        
        with open(os.path.join(dir_task, fname), 'r') as f:
            for line in f:
                #removes consecutive whitespace, refer https://docs.python.org/3/library/stdtypes.html#str.split
                entry = line.split()
                if entry[1] == '1':
                    if train_test == 'train':
                        trainDict[classname].append(entry[0]+annotationExt)
                    elif train_test == 'val':
                        valDict[classname].append(entry[0]+annotationExt)
                    else:
                        continue

    return trainDict, valDict

def getTrainVal(dir_task=os.path.join(PATH_VOC2012, DIR_IMAGESETS, DIR_TASK)):
    trainFile = 'train.txt'
    valFile = 'val.txt'
    
    trainList = []
    valList = []
    ext = '.xml'
    with open(os.path.join(dir_task, trainFile), 'r') as f:
        for line in f:
            entry = line.split()
            trainList.append(entry[0]+ext)

    with open(os.path.join(dir_task, valFile), 'r') as f:
        for line in f:
            entry = line.split()
            valList.append(entry[0]+ext)
        
    return trainList, valList

class getVOCData(keras.utils.Sequence):
    def __init__(self, filenames, classLabels = PASCAL_VOC_CLASSES, \
                    path_image = os.path.join(PATH_VOC2012, DIR_IMAGES), \
                    path_annotation = os.path.join(PATH_VOC2012, DIR_ANNOTATION), \
                    augment=False, augEach=3, finalSize=SIZE, S=S, B=1, batchSize=BATCHSIZE):
        self.filenames = filenames
        self.classLabels = classLabels
        self.path_image = path_image
        self.path_annotation = path_annotation

        self.finalSize = finalSize
        self.S = S
        self.C = len(classLabels)
        self.B = B

        self.batchSize = batchSize
        self.augment = augment
        self.augEach = 3

    def __len__(self):
        return (1 + self.augment * self.augEach) * int(np.floor(len(self.filenames) / self.batchSize))
    
    def transformCoordinates(self, c, newSize, pad):
        return int((c + pad) * self.finalSize / newSize)

    def transformImageNBoxes(self, image, size, bndboxes):
        #resize image to finalSize x finalSize with zero padding
        maxDim = np.argmax(size[:-1])
        newSize = size[maxDim]
        top = 0
        bottom = 0
        left = 0
        right = 0
        if maxDim == 0:
            top = bottom = (newSize - size[1]) // 2 
        elif maxDim == 1:
            left = right = (newSize - size[0]) // 2
        else:
            print("W: transformImageNBoxes", size)

        image = cv2.copyMakeBorder(image,top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
        image = cv2.resize(image, (self.finalSize, self.finalSize), interpolation=cv2.INTER_AREA)

        #transform bounding box coordinates
        for bbox in bndboxes:
            for i, dim in enumerate(bbox):
                if i % 2 == 0:
                    pad = left
                else:
                    pad = top

                bbox[i] = self.transformCoordinates(dim, newSize, pad)

        return cv2.normalize(image, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F), bndboxes
    
    def createLabel(self, bndboxes, classes):
        image_label = np.zeros((self.S, self.S, self.C + 5*self.B), dtype=np.float32)
        for bbox, objClass in zip(bndboxes, classes):
            if objClass not in self.classLabels:
                continue
            
            #create label with yolo format := [classProbs * C, bbox * B]
            xmid = (bbox[0] + bbox[2]) // 2
            ymid = (bbox[1] + bbox[3]) // 2
            
            #normalize box coordinates to cell size
            xmid_float = xmid / (self.finalSize / self.S)
            ymid_float = ymid / (self.finalSize / self.S)

            xmid_offset, column = np.modf(xmid_float)
            ymid_offset, row = np.modf(ymid_float)

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1] 
            
            #normalize box dimensions to image size
            norm_width = np.log(width / float(self.finalSize))
            norm_height = np.log(height / float(self.finalSize))

            classEncoding = [0] * self.C
            classEncoding[DICT_CLASSES[objClass]] = 1

            #since a grid cell has only one label
            #this ensures the label corresponds to class of highest freq. in the dataset
            prev_classLabel = image_label[int(row), int(column), : self.C]
            if prev_classLabel.any(): 
                idx = np.argmax(prev_classLabel)
                if PASCAL_VOC_SORT_IDX[idx] > PASCAL_VOC_SORT_IDX[DICT_CLASSES[objClass]]:
                    continue

            #add [1] for object confidence, bbox := [xmid, ymid, width, height, objectness]
            image_label[int(row), int(column)] = classEncoding + [xmid_offset, ymid_offset, norm_width, norm_height] + [1]
        return image_label

    def __getitem__(self, idx):
        batch_filenames = self.filenames[idx * self.batchSize : (idx + 1) * self.batchSize]
        images = []
        labels = []
        for xmlPath in batch_filenames:
            filename, size, classes, bndboxes = getAnnotations(xmlPath)
            image = cv2.imread(os.path.join(self.path_image, filename), cv2.IMREAD_COLOR)
    
            image, bndboxes = self.transformImageNBoxes(image, size, bndboxes)
            #add image to batch
            images.append(image)
    
            image_label = self.createLabel(bndboxes, classes)
            #add label to batch
            labels.append(image_label)
    
            #permutation = list(range(len(images)))
            #random.shuffle(permutation)

        return np.array(images), np.array(labels)



#if self.augment:
#                aug_images = []
#                aug_labels = []
#                for a in range(self.augEach):
#                    aug_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#                    aug_image[:, :, 0] = np.clip(aug_image[:, :, 0] * np.random.uniform(0.9, 1.1), 0, 180)
#                    aug_image[:, :, 1] = np.clip(aug_image[:, :, 1] * np.random.uniform(0.75, 1.25), 0, 255)
#                    aug_image[:, :, 2] = np.clip(aug_image[:, :, 2] * np.random.uniform(0.75, 1.25),0 , 255)
#                    aug_image = cv2.cvtColor(aug_image, cv2.COLOR_HSV2BGR)
#                    
#                    aug_image, aug_bndboxes = self.transformImageNBoxes(aug_image, size, bndboxes)
#                    aug_images.append(aug_image)
#
#                    aug_image_label = self.createLabel(aug_bndboxes, classes)
#                    #add label to batch
#                    aug_labels.append(aug_image_label)
#
#                images += aug_images
#                labels += aug_labels
#
#
