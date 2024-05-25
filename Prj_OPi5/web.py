import time
import cv2
import base64
import multiprocessing
from ctypes import c_bool
from flask import  request, jsonify, Flask, render_template, Response
from uart import Car_Cmd

####################################################################################################
def serve_result(image_queue,data_queue,cmd_queue,car_status,detect_status):
    
    app = Flask(__name__)

    def generate_frames():
        while True:
            if not image_queue.empty():
                st=time.time()
                print(image_queue.qsize())
                img = image_queue.get()
                print("queue:",time.time()-st,"s")
                # 将处理后的帧转换为JPEG格式
                #ret, img = cv2.imencode('.jpg', img)
                frame = img.tobytes()
                print("frames:",time.time()-st,"s")
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    def generate_additional_data():
        while True:
            if not data_queue.empty():
                st=time.time()
                data = data_queue.get()
                event = f'data: {data}\n\n'
                yield event.encode('utf-8')
                ##time.sleep(1)  # 控制数据发送频率

    @app.route('/')
    def index():
        return render_template('web.html')


    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
        
    @app.route('/data')
    def additional_data_feed():

        return Response(generate_additional_data(), mimetype='text/event-stream')

    
    @app.route('/button', methods=['POST'])
    def detect_event():
        data = request.json
        action = data.get('action')
        key = data.get('key')
        if action == 'detect':
            if key == 'start':
                car_status.value=False
                detect_status.value=True
            elif key == 'stop':
                detect_status.value=False
        elif action == 'car':
            if key == 'start':
                car_status.value=True
            elif key == 'stop':
                car_status.value=False
                cmd_queue.put(Car_Cmd.STOP)
                cmd_queue.put(int(0))
        print(f"Received key action: {action} for key: {key}")
        # 根据需求处理这些信息，例如记录日志或更新状态
        return jsonify({"status": "success"}),200

    @app.route('/keypress', methods=['POST'])
    def keypress_event():
        data = request.json
        key_action = data.get('action')
        key_pressed = data.get('key')
        if key_action == 'keydown':
            if key_pressed == 'W':
                cmd_queue.put(Car_Cmd.RUN)
                cmd_queue.put(int(10))
            elif key_pressed == 'A':
                cmd_queue.put(Car_Cmd.TURN_LEFT)
                cmd_queue.put(int(10))
            elif key_pressed == 'S':
                cmd_queue.put(Car_Cmd.RUN_BACK)
                cmd_queue.put(int(10))
            elif key_pressed == 'D':
                cmd_queue.put(Car_Cmd.TURN_RIGHT)
                cmd_queue.put(int(10))
        else:
            cmd_queue.put(Car_Cmd.STOP)
            cmd_queue.put(int(0))
        print(f"Received key action: {key_action} for key: {key_pressed}")
        # 根据需求处理这些信息，例如记录日志或更新状态
        return jsonify({"status": "success"})
    

    # 运行Flask应用
    app.run(host='0.0.0.0', port=5050) # , debug=True --host=0.0.0.0 --port=5000
