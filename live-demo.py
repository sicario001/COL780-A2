from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import numpy as np
import time
from blockBasedTracker import blockBasedTracking
from templateTrackingAffine import affineLkInit,affineLkTracking,hyper_params


app = Flask(__name__,template_folder='./templates')

camera = cv2.VideoCapture(0)
template = None
trackMethod = "block"
affineParams = {'initialized':False,'pyr_layers':3}
blockParams = {'initialized':False}
prev = time.time()
capturing = True
frame_rate = 5

def track():
    global template,prev,hyper_params
    if template is not None:
        if trackMethod == 'block':
            if not blockParams['initialized']:
                blockParams['template_start_point'] = np.array(camera.read()[1].shape)//2
                blockParams['method'] = cv2.TM_SQDIFF_NORMED
                print(camera.read()[1].shape)
        else:
            if not affineParams['initialized']:
                affineParams['template_start_point'] = np.array(camera.read()[1].shape)[::-1]//2
                affineParams['init_template'] = template.copy()
                affineParams['init_template_start_point'] = np.array(camera.read()[1].shape)[::-1]//2
                affineParams['template_box'] = np.array([(0,0),np.array(template.shape)[::-1]])
                affineParams['frame_0_pyr'],affineParams['coord_pyr'],affineParams['Jacobian_pyr'] = affineLkInit(template,affineParams['pyr_layers'],affineParams['template_box'])
    while capturing:
        time_elapsed = time.time() - prev
        if time_elapsed > 1./frame_rate:
            for i in range(4):
                camera.read()
            success, frame = camera.read()  # read the camera frame
            prev = time.time()
            if template is not None:
                if trackMethod == 'block':
                    frame,template,blockParams['template_start_point'] = blockBasedTracking(frame, template, blockParams['template_start_point'], blockParams['method'])
                else:
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    _, _, affineParams['init_template_start_point'],affineParams['rect_bound'],frame = affineLkTracking(affineParams['coord_pyr'],affineParams['Jacobian_pyr'],affineParams['template_box'],affineParams['frame_0_pyr'],frame,affineParams['pyr_layers'],affineParams['init_template'],affineParams['template_start_point'],affineParams['init_template_start_point'])
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
    global template, affineParams, blockParams
    template = None
    affineParams = {'initialized':False,'pyr_layers':3}
    blockParams = {'initialized':False}
    return redirect(url_for('index'), code=302)

@app.route('/set/<mthd>')
def set(mthd):
    global blockParams
    if mthd=="sqdiff":
        blockParams['method'] = cv2.TM_SQDIFF
    if mthd=="norm":
        blockParams['method'] = cv2.TM_SQDIFF_NORMED
    return redirect(url_for('index'), code=302)

@app.route('/video_feed')
def video_feed():
    return Response(track(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
   app.run(debug = True)
