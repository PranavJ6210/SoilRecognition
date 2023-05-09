import pandas as pd
import joblib
import cv2
import numpy as np
import skimage
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from flask import Flask, request, jsonify, render_template
import io

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
        result = {
            "soilType": "This is Black soil",
            "recommendedCrop": "Best suited to grow: Wheat",
            "fertiliser": "Fertilizer to use: Urea"
        }
    elif x == 1:
        result = {
            "soilType": "This is Cinder soil",
            "recommendedCrop": "Best suited to grow: Strawberries",
            "fertiliser": "Fertilizer to use: Ammonium Nitrate"
        }
    elif x == 2:
        result = {
            "soilType": "This is Laterite soil",
            "recommendedCrop": "Best suited to grow: Tea or Coffee",
            "fertiliser": "Fertilizer to use: NPK"
        }
    elif x == 3:
        result = {
            "soilType": "This is Peat soil",
            "recommendedCrop": "Best suited to grow: Onion",
            "fertiliser": "Fertilizer to use: Diammonium phosphate"
        }
    else:
        result = {
            "soilType": "This is Yellow soil",
            "recommendedCrop": "Best suited to grow: Orange",
            "fertiliser": "Fertilizer to use: Citrus Food fertilizer"
        }

    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    result = process_image(file)
    response = {'result': result}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)