import cv2
import time
from rknnpool import rknnPoolExecutor
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
from ctypes import c_bool
from web import serve_result
from uart import My_Serial,process_serial,Car_Cmd
from rknn_yolo import draw,Inference
# 图像处理函数，实际应用过程中需要自行修改


serial_name="/dev/ttyS6" #"/dev/ttyS6""/dev/ttyCH341USB0"

RKNN_MODEL_1 = "./weights/best_480p.rknn"
RKNN_MODEL_2 = "./weights/best.rknn"
RKNN_MODEL_3 = "./weights/best3.rknn"

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

anchors_1 = [[[10,13], [16,30], [33,23]],[[30,61], [62,45], [59,119]],[[116,90], [156,198], [373,326]]]

anchors_2 = [[[37,34], [66,52], [96,58]], [[105,111], [150,177], [1294,298]], [[1437,541], [1617,654],[1497,850]]]
#[[[75,83], [104,105], [120,133]], [[157,185], [681,653], [1448,584]], [[1461,859], [1812,812],[1509,986]]]


IMG_SIZE_1 = (640, 480)   # (width, height), such as (1280, 736)
IMG_SIZE_2 = (1920, 1088)

CLASSES = ("BIPV","BIPV_sand","fend")
CLASSES2 = ("mark")
# 推理线程数
TPEs_1 = 6
TPEs_2 = 9

Kp,Kd=100,1
err_last=0

def pid(Cx):
    global err_last
    err=Cx-0.5
    err_D=err-err_last
    output=abs(float(Kp*err +Kd*err_D))
    return output

def process_for_run(pool,cap,image_queue,data_queue,cmd_queue,fps,car_status,detect_status,control,first_time_1):
    if first_time_1 is True:
        #修改摄像头分辨率为480P
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,IMG_SIZE_1[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,IMG_SIZE_1[1])
        # 清理之前提交的旧任务
        pool.clear()
        # 初始化异步所需要的帧
        for i in range(TPEs_1 + 1):
            ret, img = cap.read()
            if not ret:
                cap.release()
                del pool
                exit(-1)
            pool.put(img)
        print("fisrt RUN")
        #左转360度寻找光伏板
        speed=0x02
        cmd_queue.put(Car_Cmd.TURN_LEFT)
        cmd_queue.put(int(speed))
    else:
        data=""
        ret, img = cap.read()
        pool.put(img)
        boxes, classes, scores, flag = pool.get()
        
        if classes is not None :
            if control is True:
                print("classes:",classes)
                BIPV_where=np.where(classes == 0)[0]
                print(BIPV_where)
                if len(BIPV_where) > 0:
                    box=boxes[BIPV_where][0]
                    if box is not None:
                        print(box)
                        left, top, right, bottom = box
                        area_per=(bottom-top)*(right-left)/IMG_SIZE_1[0]/IMG_SIZE_1[1]
                        center_x,center_y=(right+left)/2/IMG_SIZE_1[0],(int(bottom)+int(top))/2/IMG_SIZE_1[1]
                        if 0.4 < center_x < 0.6:
                            if left <= 0.2 or right >=0.8:
                                #data="停止前进" #串口发送停止指令
                                cmd_queue.put(Car_Cmd.STOP)
                                cmd_queue.put(int(0))
                                car_status.value=False
                            else:
                                #data="小车直行"
                                cmd_queue.put(Car_Cmd.RUN)
                                cmd_queue.put(int(0))
                                #pass  #小车直行
                        elif center_x < 0.4:#物体偏左
                            #data="小车向左微调"
                            speed=0x01#pid(center_x)
                            cmd_queue.put(Car_Cmd.TURN_LEFT)
                            cmd_queue.put(speed)
                            #pass  #小车向左微调
                        elif center_x > 0.6 :
                            #data="小车向右微调"
                            speed=0x01#pid(center_x)
                            cmd_queue.put(Car_Cmd.TURN_RIGHT)
                            cmd_queue.put(speed)
                            #pass  #小车向右微调
                        #data=center_x
            else:
                cmd_queue.put(Car_Cmd.STOP)
                cmd_queue.put(int(0))
        # draw process result and fps
        cv2.putText(img, f'fps: {fps}',
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 125, 125), 2)
 
        if boxes is not None:
            draw(img, boxes, scores, classes,CLASSES)
            
        ret, img = cv2.imencode('.jpg', img)

        image_queue.put(img)
        data_queue.put(data)

def process_for_inference(pool,cap,image_queue,data_queue,cmd_queue,fps,first_time_2=True):
    if first_time_2:
        #修改摄像头分辨率为1080P
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        # 清理之前提交的旧任务
        pool.clear()
        cmd_queue.put(Car_Cmd.STOP)
        cmd_queue.put(int(0))
        # 初始化异步所需要的帧
        for i in range(TPEs_2 + 1):
            ret, img = cap.read()
            if not ret:
                cap.release()
                del pool
                exit(-1)
            pool.put(img)
            print("fisrt DETECT")
    else:
        data=""
        ret, img = cap.read()
        pool.put(img)
        boxes, classes, scores, flag = pool.get()


        if classes is not None :
            
            sand=len(np.where(classes == 1)[0])
            fend=len(np.where(classes == 2)[0])
            print("存在")
            if fend>0 or sand>0:
                data="光伏板存在"
                if fend>0:
                    data+="  遮挡物  "
                if sand>0:
                    data+="  沙子  "

        # draw process result and fps
        cv2.putText(img, f'fps: {fps}',
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 125, 125), 2)
 
        if boxes is not None:
            draw(img, boxes, scores, classes,CLASSES)
            print("aaaaaaaa")

        ret, img = cv2.imencode('.jpg', img)
        
        image_queue.put(img)
        data_queue.put(data)


if __name__ == '__main__':

    
    # 创建队列用于主进程和子进程间通信
    image_queue = multiprocessing.Queue(15)
    data_queue = multiprocessing.Queue(15)
    cmd_queue = multiprocessing.Queue(6)

    #创建一个可以共享于主进程和子进程的标志
    car_status = multiprocessing.Value(c_bool, False)
    detect_status = multiprocessing.Value(c_bool, False)

    # 创建并启动子进程
    server_process = multiprocessing.Process(target=serve_result, args=(image_queue,data_queue,cmd_queue,car_status,detect_status))
    server_process.start()

    uart_prouartcess = multiprocessing.Process(target=process_serial, args=(serial_name,cmd_queue))
    uart_prouartcess.start()

    # 初始化rknn池
    pool_1 = rknnPoolExecutor(
        rknnModel=RKNN_MODEL_1,
        img_size=IMG_SIZE_1,
        anchors=anchors_1,
        TPEs=TPEs_1,
        func=Inference)
    
    pool_2 = rknnPoolExecutor(
        rknnModel=RKNN_MODEL_2,
        img_size=IMG_SIZE_2,
        anchors=anchors_2,
        TPEs=TPEs_2,
        func=Inference)
        
    

    cap = cv2.VideoCapture(0)
    #修改摄像头格式为mjpg，和分辨率为1080P
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    cap.set(cv2.CAP_PROP_FPS, 30)  
 
    fps,frames,loopTime,initTime = 0,0,time.time(),time.time()
    # 数据处理
    data='无'
    first_time_1,first_time_2=True,True
    control=True
    while (cap.isOpened()):

        if car_status.value is True:
                process_for_run(pool_1,cap,image_queue,data_queue,cmd_queue,fps,car_status,detect_status,control,first_time_1)
                control=not control
                print(control)
                first_time_1=False
                first_time_2=True
            
                print("run")
        elif detect_status.value is True:
            process_for_inference(pool_2,cap,image_queue,data_queue,cmd_queue,fps,first_time_2)
            first_time_2=False
            first_time_1=True
            print("detect")
        else:
            
                
            ret, img = cap.read()
            cv2.putText(img, f'fps: {fps}',
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 125, 125), 2)
                    
            ret, img = cv2.imencode('.jpg', img)
            
            image_queue.put(img)
            data_queue.put(data)
            print("normal")
            
        frames+=1
        if frames==30:
            fps = round(30/(time.time() - loopTime))
            frames=0
            loopTime=time.time()
        


    # When everything done, release the video capture object
    cap.release()

    pool_1.release()
    pool_2.release()
