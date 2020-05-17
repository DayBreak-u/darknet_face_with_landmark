#!/usr/bin/ python3
# -*- coding: utf-8 -*-
# @Time    : 2019-10-17
# @Author  : vealocia
# @FileName: evaluation_on_widerface.py

import math
import os
import sys

import cv2
sys.path.append('../')
import darknet_landmark as darknet

val_image_root = "/mnt/data1/yanghuiyu/dlmodel/Fd/RetinaFace/data/retinaface/val/images/"  # path to widerface valuation image root
val_result_txt_save_root = "./widerface_evaluate/widerface_evaluation/"  # result directory
# val_result_img_save_root = "./result_imgs/"  # result directory



def cvDrawBoxes(detections, img ,ratio_w , ratio_h ):
    for detection in detections:
        xmin, ymin, xmax, ymax = detection[:4]

        pt1 = (xmin , ymin )
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
        # cv2.putText(img,
        #             detection[0].decode() +
        #             " [" + str(round(detection[1] * 100, 2)) + "]",
        #             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             [0, 255, 0], 2)
    return img
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

netMain = None
metaMain = None
altNames = None
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


counter = 0
for parent, dir_names, file_names in os.walk(val_image_root):
    for file_name in file_names:
        if not file_name.lower().endswith('jpg'):
            continue
        im = cv2.imread(os.path.join(parent, file_name), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape
        ratio_w = darknet.network_width(netMain) * 1.0 / w
        ratio_h = darknet.network_height(netMain) * 1.0 / h

        img_resized = cv2.resize(img_rgb,
                                 (darknet.network_width(netMain),
                                  darknet.network_height(netMain)),
                                 interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.05, nms=0.3)



        boxes = []
        for detection in detections:
            x, y, w, h, score = detection[2][0], \
                                detection[2][1], \
                                detection[2][2], \
                                detection[2][3], \
                                float(detection[1])

            xmin, ymin, xmax, ymax = convertBack(
                float(x) / ratio_w, float(y) / ratio_h, float(w) / ratio_w, float(h) / ratio_h)
            boxes.append([xmin, ymin, xmax, ymax, score])



        event_name = parent.split('/')[-1]
        if not os.path.exists(os.path.join(val_result_txt_save_root, event_name)):
            os.makedirs(os.path.join(val_result_txt_save_root, event_name))
        fout = open(os.path.join(val_result_txt_save_root, event_name, file_name.split('.')[0] + '.txt'), 'w')


        # if not os.path.exists(os.path.join(val_result_img_save_root, event_name)):
        #     os.makedirs(os.path.join(val_result_img_save_root, event_name))

        # image = cvDrawBoxes(boxes, im, ratio_w, ratio_h)
        # cv2.imwrite(os.path.join(val_result_img_save_root, event_name, file_name.split('.')[0] + '.jpg'), image)

        fout.write(file_name.split('.')[0] + '\n')
        fout.write(str(len(boxes)) + '\n')
        for i in range(len(boxes)):
            bbox = boxes[i]

            fout.write('%d %d %d %d %.03f' % (math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1]), bbox[4] if bbox[4] <= 1 else 1) + '\n')
        fout.close()
        counter += 1
        print('[%d] %s is processed.' % (counter, file_name))
