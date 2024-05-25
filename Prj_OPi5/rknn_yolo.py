import os
import cv2
import sys
import argparse
from rknnlite.api import RKNNLite
import numpy as np
import urllib
import traceback
import time
import datetime as dt
import multiprocessing
from web import serve_result

serial_name="/dev/ttyS6" #"/dev/ttyS6"

RKNN_MODEL = "best.rknn"

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

CLASSES = ("BIPV","BIPV_sand","fend")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def box_process(position, IMG_SIZE, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = sigmoid(position[:,:2,:,:])*2 - 0.5
    box_wh = pow(sigmoid(position[:,2:4,:,:])*2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2  # bottom right y

    return xyxy


first_frame = True

def post_process(input_data,IMG_SIZE,anchors):

    #[[[10,13], [16,30], [33,23]],[[30,61], [62,45], [59,119]],[[116,90], [156,198], [373,326]]]
    #anchors = [anchors[i] for i in masks]
    boxes, scores, classes_conf = [], [], []
    # 1*255*h*w -> 3*85*h*w
    #print(input_data[1].shape)
    #print("len(anchors[0])",len(anchors[0]))
    input_data = [_in.reshape([len(anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
    #print(input_data[1].shape)
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:,:4,:,:], IMG_SIZE, anchors[i]))
        scores.append(sigmoid(input_data[i][:,4:5,:,:]))
        classes_conf.append(sigmoid(input_data[i][:,5:,:,:]))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    
    

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    
    
    global temp_scores
    global first_frame
    if not first_frame:
        pre_scores=temp_scores
    else :
        pre_scores=0
    temp_scores=scores
    scores+=pre_scores*0.8

    # nms
    nboxes, nclasses, nscores = [], [], []

    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes,CLASSES):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def letter_box(im, new_shape, pad_color=(0,0,0), info_need=False):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    # Scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
 
    # Compute padding
    ratio = r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
 
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border
    
    if info_need is True:
        return im, ratio, (dw, dh)
    else:
        return im



def Inference(rknn,IMG_SIZE,anchors,img):

    print('--> Running model')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = letter_box(im= img.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    #img = cv2.resize(img_src, (640, 512), interpolation=cv2.INTER_LINEAR) # direct resize
    input = np.expand_dims(img, axis=0)
    outputs = rknn.inference([input])
    boxes, classes, scores = post_process(outputs,IMG_SIZE,anchors)
    print('done')
    
    return boxes, classes, scores

if __name__ == '__main__':

    # 创建RKNN对象
    rknn = RKNNLite()
    
    #加载RKNN模型
    print('--> Load RKNN model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')
 
     # 初始化 runtime 环境
    print('--> Init runtime environment')
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
 
    # 数据处理
    while (cap.isOpened()):
        
        ret, img = cap.read()

        if not ret:
            break
 
        # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (1920, 1080))
        #cv2.imshow("img", img)
        pad_color = (0,0,0)
        img1=img
        img1 = letter_box(im= img.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        #img = cv2.resize(img_src, (640, 512), interpolation=cv2.INTER_LINEAR) # direct resize
        input = np.expand_dims(img1, axis=0)
 
        outputs = rknn.inference([input])
        
        boxes, classes, scores = post_process(outputs)
 
        img_p = img.copy()
 
        if boxes is not None:
            
            draw(img_p, boxes, scores, classes)
        # show output
        cv2.imshow("post process result", img_p)
        #print(dt.datetime.utcnow() - start)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    rknn.release()
