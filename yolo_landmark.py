from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet_landmark as darknet



def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img ,ratio_w , ratio_h ):
    print(len(detections))
    for detection in detections:
        # print(detection)
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        b = np.array(detection[3]) / np.array([ratio_w,ratio_h,ratio_w,ratio_h,ratio_w,ratio_h,
                                               ratio_w,ratio_h,ratio_w,ratio_h])
        b = b.astype(np.int)
        xmin, ymin, xmax, ymax = convertBack(
            float(x) / ratio_w , float(y) / ratio_h , float(w) / ratio_w, float(h) / ratio_h)

        cv2.circle(img, (b[0], b[1]), 1, (0, 0, 255), 4)
        cv2.circle(img, (b[2], b[3]), 1, (0, 255, 255), 4)
        cv2.circle(img, (b[4], b[5]), 1, (255, 0, 255), 4)
        cv2.circle(img, (b[6], b[7]), 1, (0, 255, 0), 4)
        cv2.circle(img, (b[8], b[9]), 1, (255, 0, 0), 4)


        pt1 = (xmin , ymin )
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    # detection[0].decode() +
                    # " [" + str(round(detection[1] * 100, 2)) + "]",
                     str(round(detection[1] * 100, 2)) ,
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames


    configPath = "./cfg/mbv2_yolov3_face.cfg"
    # configPath = "./cfg/lite_yolov3_face.cfg"
    weightPath = "./backup/mbv2_yolov3_face_last.weights"
    metaPath = "./data/face.data"

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    import glob
    imgs = glob.glob("./test_imgs/input/*.*p*g")
    for img_path in imgs:
        print(img_path)
        # img_path  = "data/face3.jpeg"
        img  = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,_ = img.shape
        ratio_w = darknet.network_width(netMain) * 1.0 / w
        ratio_h = darknet.network_height(netMain) * 1.0 / h

        img_resized = cv2.resize(img_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,img_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.15 ,nms= 0.35)
        # print(detections)
        image = cvDrawBoxes(detections, img , ratio_w, ratio_h)
        cv2.imwrite(img_path.replace("input","output"),image)

if __name__ == "__main__":
    YOLO()
