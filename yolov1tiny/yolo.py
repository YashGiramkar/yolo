import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, ReLU, LeakyReLU, BatchNormalization, Flatten, Dense

from yolov1tiny.globalConst import *

def baseModel():
    model = keras.models.Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    
    model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(32, 3, 1, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(64, 3, 1, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(128, 3, 1, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(256, 3, 1, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(512, 3, 1, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(1024, 3, 1, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(256, 3, 1, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Flatten())
    #model.summary()

    return model

def yoloTinyModel(classModel):
    finalModel = classModel

    finalModel.pop()
    finalModel.pop()
  
    finalModel.add(Dense(S*S*(C+5*B), activation='linear'))
    finalModel.add(tf.keras.layers.Reshape((S, S, C + 5*B), name='detection'))
    
    return finalModel

def yoloClassModel():
    classModel = baseModel()
    
    classModel.add(Dense(S*S*C, activation='linear'))
    classModel.add(tf.keras.layers.Reshape((S, S, C), name='classification'))
    
    return classModel

len_boxData = 5
n_boxProp = 4
n_objProb = 1
n_coord  = 2

def yoloLoss(yTrue, yPred, verbose=False):
    #yVal := (batchSize, S, S, detectionLayer(C + len_boxData*B))
    #allow only on box per cell for true label
    yTrue = K.reshape(yTrue, (-1, S, S, C + len_boxData))
    yPred = K.reshape(yPred, (-1, S, S, C + len_boxData*B))

    #classProb := (..., C(class probabilities))
    classProb = yTrue[..., : C]
    #boxData := (batchSize, S, S, B, len_boxData(x, y, w, h, c))
    #allow only on box per cell for true label
    boxData = K.reshape(yTrue[..., C : ], (-1, S, S, 1, len_boxData))
    #bBox := (..., boxCoord(x, y, w, h))
    bBox = boxData[..., : n_boxProp]
    bBox_wh = tf.exp(bBox[..., 2 :])

    #objProb := (..., objProb)
    objProb = boxData[..., n_boxProp : n_boxProp + n_objProb]

    pred_classProb = yPred[..., : C]
    pred_boxData = K.reshape(yPred[..., C : ], (-1, S, S, B, len_boxData))
    pred_bBox = pred_boxData[..., : n_boxProp]
    pred_bBox_wh = tf.exp(pred_bBox[..., 2 :])

    #pred_bBox = tf.split(pred_bBox, num_or_size_splits=B, axis=-2)
    pred_objProb = pred_boxData[..., n_boxProp : n_boxProp + n_objProb]

    #mask for responsible box
    boxIOUs = calcIOU(bBox, pred_bBox)
    bestIOUs = K.max(boxIOUs, axis=-1, keepdims=True)
    boxMask = K.cast(boxIOUs >= bestIOUs, K.dtype(bestIOUs))

    locLoss = LAMBDA_COORD * objProb[..., 0] * boxMask * K.sum(K.square(bBox[..., : n_coord] - pred_bBox[..., : n_coord]), axis=-1)
    locLoss = K.sum(locLoss, axis=-1)
    whLoss = LAMBDA_COORD * objProb[..., 0] * boxMask * K.sum(K.square(K.sqrt(bBox_wh[..., : ]) - K.sqrt(pred_bBox_wh[..., : ])), axis=-1)
    whLoss = K.sum(whLoss, axis=-1) 

    objLoss = objProb[..., 0] * boxMask * K.sum(K.square(1 - pred_objProb), axis = -1)
    objLoss = K.sum(objLoss, axis=-1)
    nobjLoss = LAMBDA_NOBJ * (1 - objProb[..., 0] * boxMask) * K.sum(K.square(pred_objProb), axis = -1)
    nobjLoss = K.sum(nobjLoss, axis=-1)

    classLoss = objProb[..., 0] * K.sum(K.square(classProb - pred_classProb), axis = -1, keepdims=True)
    classLoss = K.sum(classLoss, axis=-1)

    #classLoss = tf.keras.losses.binary_crossentropy(classProb, pred_classProb, from_logits = True)

    if verbose:
        tf.print("locLoss", K.sum(locLoss, axis=(1,2)), summarize = -1)
        tf.print("whLoss", K.sum(whLoss, axis=(1,2)), summarize = -1)
        tf.print("objLoss", K.sum(objLoss, axis=(1,2)), summarize = -1)
        tf.print("nobjloss", K.sum(nobjLoss, axis=(1,2)), summarize = -1)
        tf.print("classLoss", K.sum(classLoss, axis=(1,2)), summarize = -1)

    totalLoss = locLoss + whLoss + objLoss + nobjLoss + classLoss

    return K.sum(totalLoss, axis=(1,2))

def yoloClassLoss(yTrue, yPred):
    #yVal := (batchSize, S, S, detectionLayer(C + len_boxData*B))
    #allow only on box per cell for true label
    yTrue = K.reshape(yTrue, (-1, S, S, C + len_boxData))
    objProb = yTrue[..., C + 4 : C + 5]

    yPred = K.reshape(yPred, (-1, S, S, C))
    yPred = yPred[..., : C]
    
    classProb = yTrue[..., : C]
    pred_classProb = yPred[..., : C]

    pred_classProb = tf.math.sigmoid(pred_classProb)
    
    #entropyLoss = tf.keras.losses.binary_crossentropy(classProb, pred_classProb, from_logits = True)
    mseLoss = K.sum(objProb * K.square(classProb - pred_classProb), axis=-1) / C
    #entropyLoss = -classProb * K.log(pred_classProb) - (1-classProb) * K.log(1-pred_classProb)

    totalLoss = mseLoss

    return K.sum(totalLoss, axis=(1,2))

def xywh2minmax(bBox):
    #bBox(..., x, y, w ,h)
    #bBox x, y are normalised down to w,h dimensions
    bBox_wh = tf.exp(bBox[..., 2 :])
    xy_min = bBox[..., : 2] / S - bBox_wh[..., :] / 2. 
    xy_max = bBox[..., : 2] / S + bBox_wh[..., :] / 2.

    return xy_min, xy_max

def calcIOU(bb1, bb2):
    bb1_xy_min, bb1_xy_max = xywh2minmax(bb1)
    bb2_xy_min, bb2_xy_max = xywh2minmax(bb2)
    intersect_mins = K.maximum(bb1_xy_min, bb2_xy_min)
    intersect_maxes = K.minimum(bb1_xy_max, bb2_xy_max)

    intersect_wh = K.maximum((intersect_maxes - intersect_mins), tf.zeros(tf.shape(intersect_maxes)))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    bb1_wh = tf.exp(bb1[..., 2 : ])
    bb2_wh = tf.exp(bb2[..., 2 : ])
    bb1_area = bb1_wh[..., 0] * bb1_wh[..., 1]
    bb2_area = bb2_wh[..., 0] * bb2_wh[..., 1]

    union_area = bb1_area + bb2_area - intersect_area
    iou_score = intersect_area / union_area
    
    #round off to one & convert nan to zero
    iou_score = tf.where(tf.math.is_nan(iou_score), tf.zeros_like(iou_score), iou_score)
    iou_score = tf.where(tf.greater(iou_score, 1), tf.ones_like(iou_score), iou_score)
    
    return iou_score
