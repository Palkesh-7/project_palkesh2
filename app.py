# from numpy.core.fromnumeric import argmax
# from unittest import result
from flask import Flask
# from pyngrok import ngrok
from flask import Flask,request, render_template, Response
import cv2
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
import datetime
import os
import tensorflow as tf
# import imutils
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
# from IPython.display import display, Javascript
# from google.colab.output import eval_js
# from base64 import b64decode
# port_no = 4500
global model,capture,frame


# ngrok.set_auth_token("2FXfgow8vdsajOzrZSPGILt2Irr_QWRaa8qxxrxoanfdKm6V")
# public_url = ngrok.connect(port_no).public_url

app = Flask(__name__)
# run_with_ngrok(app)
class_names = ['Potato___Early_blight',
              'Potato___Late_blight',
              'Potato___healthy']

# class_names = ['Pepper__bell___Bacterial_spot',
#               'Pepper__bell___healthy',
#               'Potato___Early_blight',
#               'Potato___Late_blight',
#               'Potato___healthy',
#               'Tomato_Bacterial_spot',
#               'Tomato_Early_blight',
#               'Tomato_Late_blight',
#               'Tomato_Leaf_Mold',
#               'Tomato_Septoria_leaf_spot',
#               'Tomato_Spider_mites_Two_spotted_spider_mite',
#               'Tomato__Target_Spot',
#               'Tomato__Tomato_YellowLeaf__Curl_Virus',
#               'Tomato__Tomato_mosaic_virus',
#               'Tomato_healthy']

model = load_model('/content/potatoes.h5')
 
model.make_predict_function()

try:
    os.mkdir('static/shot')
except OSError as error:
    pass

try:
    os.mkdir('static/upload')
except OSError as error:
    pass


camera = cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read()
        if success:   
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass





def predict_label(image):
    

    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))

    image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}

# routes
@app.route("/")
def main():
    return render_template("index.html")


@app.route('/capture',methods=['POST','GET'])
def capture():
    return render_template('capture.html')
    
    
@app.route('/video_feed')
def video_feed():
  
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def requests():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
            success, frame = camera.read()
            
            
            if(capture):
                capture=0
                now = datetime.datetime.now()
                img_path1 = os.path.sep.join(['static/shots', "shot_{}.jpg".format(str(now).replace(":",''))])
                cv2.imwrite(img_path1, frame)
                camera.release()
                cv2.destroyAllWindows()

                result = predict_label(img_path1)



                return render_template("output.html", prediction=result["class"],confidence =result["confidence"],img_path = img_path1)


                
    elif request.method=='GET':
        return render_template('capture.html')
    return render_template('capture.html')
 



@app.route("/submit", methods=['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/upload" + img.filename	
		img.save(img_path)

		result1 = predict_label(img_path)

	return render_template("output.html", prediction=result1["class"],confidence = result1["confidence"],img_path = img_path)



# print(f"click here for website by globle link  {public_url}")
# app.run(port=port_no)

if __name__ =='__main__':
    app.run(debug = True)