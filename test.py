from yolov1tiny import *

import random
import numpy as np

def test_getAnnotations():
    test_XML = '2007_005331.xml'
    filename, size, classes, bndboxes = getAnnotations(test_XML)

    t_filename = '2007_005331.jpg'
    t_size = [318, 480, 3] 
    t_classes = ['horse', 'person', 'car', 'car', 'car'] 
    t_bndboxes = [[71, 142, 291, 448], [101, 84, 245, 304], [84, 87, 161, 140], [34, 92, 82, 138], [1, 91, 40, 140]]

    if (t_filename, t_size, t_classes, t_bndboxes) == (filename, size, classes, bndboxes):
        print("S: getAnnotations")
    else:
        print("F: getAnnotations")
        print(filename, size, classes, bndboxes)

    print(t_filename, t_size, t_classes, t_bndboxes)

def test_getClassObjDict():
    trainDict, valDict = getClassObjDict()
    if len(trainDict.keys()) == C and len(valDict.keys()) == C:
        sum_train = 0
        sum_val = 0
        set_train = set()
        set_val = set()
        
        for key in trainDict.keys():
            sum_train += len(trainDict[key])
            set_train = set_train.union(set(trainDict[key]))

            #print("train", key, len(trainDict[key]))

        print("Train images: ", len(set_train))
        
        for key in valDict.keys():
            sum_val += len(valDict[key])
            set_val = set_val.union(set(valDict[key]))

            #print("val", key, len(trainDict[key]))

        print("Val images: ", len(set_val))

        xmlFile = trainDict[PASCAL_VOC_CLASSES[0]][0]
        _, ext = xmlFile.split('.')
        if ext != 'xml':
            print("F: extension ", ext)
        else:
            print("S: getClassObjDict")
    else:
        print("F: getClassObjDict")

def test_getTrainVal():
    trainList, valList = getTrainVal()
    if len(trainList) != 5717 or len(valList) != 5823:
        print("F: getTrainVal")
    else:
        print("S: getTrainVal")

def test_getVOCData():
    shape_grid = (S, S)
    test_XML = '2008_000008.xml'
    filename, size, classes, bndboxes = getAnnotations(test_XML)
    print(filename, size, classes, bndboxes)
    
    testClasses = list(PASCAL_VOC_CLASSES)
    testClasses.remove('horse')

    dataGen = getVOCData([test_XML], classLabels=testClasses, batchSize=1)
    images, labels = dataGen[0]
    print(np.shape(labels[0]))
    
    img = drawGrid(images[0], shape_grid)
    img = drawLabels(img, labels[0], numClasses=len(testClasses))
    cv2.imshow('no_horse', img)
    cv2.waitKey(0)
    #print(labels[0])

    trainList, valList = getTrainVal()
    trainGen = getVOCData(trainList)
    valGen = getVOCData(valList)
    
    batches_train = len(trainGen)
    batches_val = len(valGen)

    print("Train batches: ",  batches_train, "Val batches: ", batches_val)
    
    images, labels = trainGen[random.randint(0, batches_train)]

    rand_idx = random.randint(0, BATCHSIZE)
    img = drawGrid(images[rand_idx], shape_grid)
    img = drawLabels(img, labels[rand_idx])
    cv2.imshow('test', img)
    cv2.waitKey(0)
    #print(labels[0])

def logWH(box):
    return np.array([box[0], box[1], np.log(box[2]), np.log(box[3])], dtype=np.float32)

def test_calcIOU():
    box = [0.25, 0.25, 0.5, 0.5]
    box1 = [0.25, 0.25, 0.5, 0.5]
    box2 = [0.75, 0.75, 0.5, 0.5]

    box_small = [0, 0, 1/7, 1/7]
    box3 = [1, 1, 0.5/7, 0.5/7]
    box4 = [1, 1, 1.1/7, 1.1/7]

    labels = np.zeros((2,1,1,1,4), dtype=np.float32)
    labels[0, 0, 0, 0] = logWH(box)
    labels[1, 0, 0, 0] = logWH(box_small)

    results = np.zeros((2,1,1,2,4), dtype=np.float32)
    results[0, 0, 0, 0] = logWH(box1)
    results[0, 0, 0, 1] = logWH(box2)
    results[1, 0, 0, 0] = logWH(box3)
    results[1, 0, 0, 1] = logWH(box4)

    IOU = np.array([[[[1., 0.58064514]]], [[[0., 0.0011325 ]]]])
    calc_IOU = calcIOU(labels, results).numpy()

    try:
        np.testing.assert_allclose(calc_IOU, IOU, rtol=1e-4)
        print("success: calcIOU")
    except:
        print("failed: calcIOU")
        print("calcIOU:", calc_IOU)
        print("IOU:", IOU)

def test_yoloLoss():
    truthTensor = np.zeros((1,7,7,25), dtype=np.float32)
    predTensor = np.zeros((1,7,7,30), dtype=np.float32)

    r, c = np.random.randint(0,6,2)
    [x, y, w, h] = np.random.random_sample(4)
    print(x, y, w, h)
    w = np.log(w)
    h = np.log(h)

    randClass = np.random.randint(0,20)
    truthTensor[0, r, c, randClass] = 1
    truthTensor[0, r, c, C : ] = [x, y, w, h] + [1]

    predTensor[0, r, c, randClass] = 1
    predTensor[0, r, c, C : C+5] = [x, y, w, h] + [0]
    predTensor[0, r, c, C+5 : ] = [x, y, np.log(np.exp(w) + 0.2), h] + [1]

    loss = yoloLoss(truthTensor, predTensor, verbose=True)

    tf.print(loss, summarize = -1)

def test_yoloClassLoss():
    truthTensor = np.zeros((1,7,7,25), dtype=np.float32)
    predTensor = -9999 * np.ones((1,7,7,20), dtype=np.float32)

    r, c = np.random.randint(1, 6, 2)

    randClass = np.random.randint(1, 19)

    truthTensor[0, r, c, randClass] = 1
    truthTensor[0, r, c, C + 4 : C + 5] = [1]
    
    predTensor[0, r, c, randClass] = -9999
    #predTensor[0, r, c, randClass + 1] = 0
    #predTensor[0, r+1, c, randClass] = 1

    loss = yoloClassLoss(truthTensor, predTensor)

    tf.print(loss, summarize = -1)

if __name__ == '__main__':
    #test_getAnnotations()
    #test_getClassObjDict()
    #test_getTrainVal()
    #test_getVOCData()
    #test_calcIOU()
    #test_yoloLoss()
    test_yoloClassLoss()
