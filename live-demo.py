from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import numpy as np
import time
from templateTracking import blockBasedTracking


app = Flask(__name__,template_folder='./templates')

camera = cv2.VideoCapture(0)
template = None
template_start_point = None
method = cv2.TM_SQDIFF_NORMED
prev = time.time()
capturing = True
frame_rate = 10
def track():
    global template,template_start_point,prev
    if template_start_point is None:
        template_start_point = np.array(camera.read()[1].shape)//2
        print(camera.read()[1].shape)
    while capturing:
        time_elapsed = time.time() - prev
        if time_elapsed > 1./frame_rate:
            for i in range(4):
                camera.read()
            success, frame = camera.read()  # read the camera frame
            prev = time.time()
            if template is not None:
                frame,template,template_start_point = blockBasedTracking(frame, template, template_start_point, method)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

@app.route("/", methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    global template
    if request.method == 'POST':
        f = request.files['file']
        scale = float(request.form['scale'])
        template = cv2.imdecode(np.fromstring(f.read(), np.uint8), 0)
        template = cv2.resize(template, None, fx= scale, fy= scale, interpolation= cv2.INTER_LINEAR)
        print(template.shape)
    return redirect(url_for('index'), code=302)

@app.route('/reset')
def reset():
    global template,template_start_point
    template = None
    template_start_point = None
    return redirect(url_for('index'), code=302)

@app.route('/set/<mthd>')
def set(mthd):
    global method
    if mthd=="sqdiff":
        method = cv2.TM_SQDIFF
    if mthd=="norm":
        method = cv2.TM_SQDIFF_NORMED
    return redirect(url_for('index'), code=302)

@app.route('/video_feed')
def video_feed():
    return Response(track(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
   app.run(debug = True)
