import pandas as pd
import joblib
import cv2
import numpy as np
import skimage
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load('mymodel 1.pkl')

def process_image(image):
    # Load the selected image using OpenCV
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Convert the image to the HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calculate the mean hue, saturation, and brightness values of the image
    h, s, v = cv2.split(hsv_img)
    mean_hue = np.mean(h)
    mean_saturation = np.mean(s)
    mean_brightness = np.mean(v)

    # Convert the image to grayscale
    gray_img = skimage.color.rgb2gray(img)

    # Calculate the threshold value for binarization using Otsu's method
    threshold_value = threshold_otsu(gray_img)

    # Binarize the image using the threshold value
    binary_img = gray_img < threshold_value

    # Label the connected regions in the binary image
    labeled_img = label(binary_img)

    # Calculate the region properties of the labeled regions
    region_props = regionprops(labeled_img)

    # Extract texture features using gray-level co-occurrence matrices (GLCM)
    glcm = graycomatrix(gray_img.astype(np.uint8), distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    energy = graycoprops(glcm, 'energy')

    # Calculate the porosity of the soil
    porosity = 1 - (region_props[0].area / (gray_img.shape[0] * gray_img.shape[1]))

    l=[mean_hue,mean_saturation,mean_brightness,contrast[0][0],energy[0][0],porosity]

    df=pd.DataFrame([l])

    y_pred2=model.predict(df)
    x=y_pred2[0]

    if x == 0:
        result = '''
        <style>
        body{
        background-image: url('https://foundationfar.org/wp-content/uploads/2022/05/healthy-soil-e1652963401278.jpeg');
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-size: cover; 
        }
        h2{
        color: white;
        }
        </style>
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <H2>
            This is Black soil<br/>
            Best suited to grow: Wheat<br/>
            Fertilizer to use: Urea
        </H2>
        </div>
        '''
    elif x == 1:
        result = '''
        <style>
        body{
        background-image: url('https://media.istockphoto.com/id/843942930/photo/fantasy-christmas-background.jpg?s=170667a&w=0&k=20&c=wazc4tx7gWDgG0aGCcfm1TWKtr4PkWwKH-d-qZB1Y48=');
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-size: cover; 
        }
        </style>
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <H2>
             This is Cinder soil<br/>
             Best suited to grow: Strawberries<br/>
             Fertilizer to use: Ammonium Nitrate
        </H2>
        </div>
      '''
    elif x == 2:
        result ='''
        <style>
        body{
        background-image: url('https://media.istockphoto.com/id/843942930/photo/fantasy-christmas-background.jpg?s=170667a&w=0&k=20&c=wazc4tx7gWDgG0aGCcfm1TWKtr4PkWwKH-d-qZB1Y48=');
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-size: cover; 
        }
        </style>
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <H2>
            This is Laterite soil<br/>
            Best suited to grow: Tea or Coffee<br/>
            Fertilizer to use: NPK
        </H2>
        </div>
      '''
    elif x == 3:
        result = '''
        <style>
        body{
        background-image: url('https://media.istockphoto.com/id/843942930/photo/fantasy-christmas-background.jpg?s=170667a&w=0&k=20&c=wazc4tx7gWDgG0aGCcfm1TWKtr4PkWwKH-d-qZB1Y48=');
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-size: cover; 
        }
        </style>
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <H2>
            This is Peat soil<br/>
            Best suited to grow: Onion<br/>
            Fertilizer to use: Diammonium phosphate
        </H2>
        </div>
        '''
    else:
        result = '''
        <style>
        body{
        background-image: url('https://media.istockphoto.com/id/843942930/photo/fantasy-christmas-background.jpg?s=170667a&w=0&k=20&c=wazc4tx7gWDgG0aGCcfm1TWKtr4PkWwKH-d-qZB1Y48=');
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-size: cover; 
        }
        </style>
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <H2>
            This is Yellow soil<br/>
            Best suited to grow: Orange<br/>
            Fertilizer to use: Citrus Food fertilizer
        </H2>
        </div>
        '''
    

    return result

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        print('received', file.filename)
        print("begin")
        return str(process_image(file))

    return '''  
    <!doctype html>
    <title>Farm Smart</title>

    <style>
        body {
          background-image: url('https://media.istockphoto.com/id/843942930/photo/fantasy-christmas-background.jpg?s=170667a&w=0&k=20&c=wazc4tx7gWDgG0aGCcfm1TWKtr4PkWwKH-d-qZB1Y48=');
          background-repeat: no-repeat;
          background-attachment: fixed;
          background-size: cover; 
        }
        </style>
    <b><font size="14"><font face="times new roman"><center><br><br><br> FARM SMART </b></font></font></center><br><br><br><br><br>
  
    <h3>
        <b><font size="8"><font face="times new roman"><center> UPLOAD YOUR IMAGE </b></font></font></center>
    </h3>

    <center>
    <br>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>

    </center>
    '''



@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    result = process_image(file)
    response = {'result': result}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)