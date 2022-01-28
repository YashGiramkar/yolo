import cv2
import numpy as np

from yolov1tiny.globalConst import *

def drawGrid(img, shape_grid=(7, 7), color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = shape_grid
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def drawText(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 255)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (int(x + text_w), int(y + text_h)), text_color_bg, -1)
    cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)

    return img

def drawLabels(image, label, imSize=SIZE, numClasses=C, imSegments=S, thresh=0.2):
    for ridx, row in enumerate(label):
        for cidx, cell in enumerate(row):
            idx_objClass = np.argmax(cell[:numClasses])
            if type(cell) == np.ndarray:
                classProb = cell[idx_objClass]
            else:
                classProb = cell[idx_objClass].numpy()
            
            boxes = np.reshape(cell[numClasses:], (-1, 5))
            for box in boxes:
                if box[4] >= thresh:
                    print(PASCAL_VOC_CLASSES[idx_objClass], classProb, box[4])
                    #print([(c, v) for c, v in zip(PASCAL_VOC_CLASSES, cell[:C])])

                    #transform from cell coordinates to image coordinates
                    midX = int((cidx + box[0]) * (imSize/imSegments))
                    midY = int((ridx + box[1]) * (imSize/imSegments))
                    width = int(np.exp(box[2]) * imSize)
                    height = int(np.exp(box[3]) * imSize)
                    objProb = box[4]

                    image = cv2.circle(image, (midX, midY), radius=0, color=(0,0,255), thickness=2)
                    image = cv2.rectangle(image, (midX - width//2, midY - height//2), (midX + width//2, midY + height//2), color=(0,0,255))
                    image = drawText(image, PASCAL_VOC_CLASSES[idx_objClass] + ': ' + f"{classProb:.2f}" + ', ' + f"{objProb:.2f}", pos=(midX - width//2, midY - height//2))
    return image

