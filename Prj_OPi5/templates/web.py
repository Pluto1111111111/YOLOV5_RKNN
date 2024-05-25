import os
import urllib
import traceback
import time
import datetime as dt
import sys
from flask import redirect, render_template_string, request, url_for,jsonify
import numpy as np
import cv2



####################################################################################################
if __name__ == '__main__':
    from flask import Flask, render_template, Response
    from flask import Flask, Response
    import base64
    app = Flask(__name__)

    def generate_frames():
        data=0
        while True:
            if data==100:
                data=0
            data+=1
            img=cv2.imread('img10.jpg')
            ret,img3=cv2.imencode('.jpg',img)
            img1=img3.tobytes()
            img2=base64.b64encode(img1).decode('utf-8')
            #print("suceess!!!")

            # 构造SSE事件
            event = f'data: {{"data": "{data}","image":"{img2}"}}\n\n'
            yield event

            # 使用生成器生成视频帧
            #  yield (b'--frame\r\n'
                #   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')


    @app.route('/')
    def index():
        return render_template('web.html')


    @app.route('/stream')
    def video_feed():
        return Response(generate_frames(),
                        mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache'})
        #return Response(generate_frames(),
                    #    mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/start', methods=['POST'])
    def your_endpoint():
        if request.method == 'POST':
            # message = request.form.get('message')
            print(f"succeed")
            # 这里可以处理接收到的数据，比如保存到数据库等
            response = {"status": "success", "message": "Data received successfully"}
            return jsonify(response), 200
            return jsonify(response)
    
    # 运行Flask应用
    app.run(debug=True,host='0.0.0.0', port=4050) # , debug=True --host=0.0.0.0 --port=5000

# ==================================
