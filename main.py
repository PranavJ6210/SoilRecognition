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
        <!DOCTYPE html>
<html>

<head>
    <title>Results Page</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&family=Poppins:wght@600&display=swap" rel="stylesheet">
    <style>
        body{
            background-color: aquamarine;
        }
        .column-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .column {
            flex: 1;
            text-align: center;
            padding: 20px;
        }

        h1 {
            font-family: 'Libre Baskerville';
        }

        p {
            font-family: 'Poppins';
            font-size: 18px;
        }

        .square-image {
            width: 300px;
            height: 300px;
            object-fit: cover;
        }

        footer {
            font-family: 'poppins';
            text-align: center;
            padding: 20px;
            background-color: #333;
            color: white;
        }
    </style>
</head>

<body>
    <div class="column-container">
        <div class="column">
            <h1>Your Soil<br> Type</h1>
            <p><br>It's a Black Soil<br></p>
            <img src="https://housing.com/news/wp-content/uploads/2023/03/Black-cotton-soil-Properties-types-formation-and-benefits-f.jpg" alt="Soil Type" class="square-image">
        </div>
        <div class="column">
            <h1>Recommended <br>Crops</h1>
            <p><br>Best suited to grow Wheat<br></p>
            <img src="https://cdn.britannica.com/90/94190-050-C0BA6A58/Cereal-crops-wheat-reproduction.jpg" alt="Recommended Crops" class="square-image">
        </div>
        <div class="column">
            <h1>Recommended <br>Fertilizer</h1>
            <p><br>Urea<br></p>
            <img src="https://img.tradeford.com/pimages/l/1/1043391.jpg" alt="Recommended Fertilizer" class="square-image">
        </div>
    </div>

    <footer>
        Made with <span style="color: red;">&hearts;</span> B.Tech AI & DS
    </footer>
</body>

</html>

        '''
    elif x == 1:
        result = '''
       <!DOCTYPE html>
<html>

<head>
    <title>Results Page</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&family=Poppins:wght@600&display=swap" rel="stylesheet">
    <style>
        body{
            background-color: aquamarine;
        }
        .column-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .column {
            flex: 1;
            text-align: center;
            padding: 20px;
        }

        h1 {
            font-family: 'Libre Baskerville';
        }

        p {
            font-family: 'Poppins';
            font-size: 18px;
        }

        .square-image {
            width: 300px;
            height: 300px;
            object-fit: cover;
        }

        footer {
            font-family: 'poppins';
            text-align: center;
            padding: 20px;
            background-color: #333;
            color: white;
        }
    </style>
</head>

<body>
    <div class="column-container">
        <div class="column">
            <h1>Your Soil<br> Type</h1>
            <p><br>It's a Cinder Soil<br></p>
            <img src="https://livingcolorgardencenter.net/wp-content/uploads/2021/09/LC-BLOG-Sweet-Strawberries-planting-strawberries-1024x512.jpg" alt="Soil Type" class="square-image">
        </div>
        <div class="column">
            <h1>Recommended <br>Crops</h1>
            <p><br>Best suited to grow Strawberries<br></p>
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXT9w0lmE7qutLG7Bj2fXSZVf-a3lkvkhZZQ&usqp=CAU" alt="Recommended Crops" class="square-image">
        </div>
        <div class="column">
            <h1>Recommended <br>Fertilizer</h1>
            <p><br>Ammonium Nitrate<br></p>
            <img src="https://media.istockphoto.com/id/1268317883/photo/farmers-hand-in-blue-glove-holds-white-fertilizer-for-plants.jpg?s=612x612&w=0&k=20&c=xy0lH5sBI-mYJOTWUbPCC3RaRkDBwNNkCWr_28eqFP8=" alt="Recommended Fertilizer" class="square-image">
        </div>
    </div>

    <footer>
        Made with <span style="color: red;">&hearts;</span> B.Tech AI & DS
    </footer>
</body>

</html>

      '''
    elif x == 2:
        result ='''
        <!DOCTYPE html>
<html>

<head>
    <title>Results Page</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&family=Poppins:wght@600&display=swap" rel="stylesheet">
    <style>
        body{
            background-color: aquamarine;
        }
        .column-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .column {
            flex: 1;
            text-align: center;
            padding: 20px;
        }

        h1 {
            font-family: 'Libre Baskerville';
        }

        p {
            font-family: 'Poppins';
            font-size: 18px;
        }

        .square-image {
            width: 300px;
            height: 300px;
            object-fit: cover;
        }

        footer {
            font-family: 'poppins';
            text-align: center;
            padding: 20px;
            background-color: #333;
            color: white;
        }
    </style>
</head>

<body>
    <div class="column-container">
        <div class="column">
            <h1>Your Soil<br> Type</h1>
            <p><br>It's a Laterite Soil<br></p>
            <img src="https://i0.wp.com/studyandupdates.in/wp-content/uploads/2022/01/depositphotos_56050223-stock-photo-laterite-soil-texture.jpg?fit=1023%2C682&ssl=1" alt="Soil Type" class="square-image">
        </div>
        <div class="column">
            <h1>Recommended <br>Crops</h1>
            <p><br>Best suited to grow Tea/Coffee<br></p>
            <img src="https://c8.alamy.com/comp/2BKAWK9/coffee-beans-and-black-tea-leaves-in-a-round-saucer-2BKAWK9.jpg" alt="Recommended Crops" class="square-image">
        </div>
        <div class="column">
            <h1>Recommended <br>Fertilizer</h1>
            <p><br>NPK<br></p>
            <img src="https://www.dfgrupo.com/wp-content/uploads/2022/04/NPK-Nitrofoska-DFGRUPO.jpg" alt="Recommended Fertilizer" class="square-image">
        </div>
    </div>

    <footer>
        Made with <span style="color: red;">&hearts;</span> B.Tech AI & DS
    </footer>
</body>

</html>


      '''
    elif x == 3:
        result = '''
        <!DOCTYPE html>
<html>

<head>
    <title>Results Page</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&family=Poppins:wght@600&display=swap" rel="stylesheet">
    <style>
        body{
            background-color: aquamarine;
        }
        .column-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .column {
            flex: 1;
            text-align: center;
            padding: 20px;
        }

        h1 {
            font-family: 'Libre Baskerville';
        }

        p {
            font-family: 'Poppins';
            font-size: 18px;
        }

        .square-image {
            width: 300px;
            height: 300px;
            object-fit: cover;
        }

        footer {
            font-family: 'poppins';
            text-align: center;
            padding: 20px;
            background-color: #333;
            color: white;
        }
    </style>
</head>

<body>
    <div class="column-container">
        <div class="column">
            <h1>Your Soil<br> Type</h1>
            <p><br>It's a Peat Soil<br></p>
            <img src="https://cdn.britannica.com/42/119342-050-0C297D28/peat-bed-Avon-Park-Florida.jpg" alt="Soil Type" class="square-image">
        </div>
        <div class="column">
            <h1>Recommended <br>Crops</h1>
            <p><br>Best suited to grow Onions<br></p>
            <img src="https://www.almanac.com/sites/default/files/styles/or/public/image_nodes/shutterstock_2203736779.jpg?itok=NDoeIRX7" alt="Recommended Crops" class="square-image">
        </div>
        <div class="column">
            <h1>Recommended <br>Fertilizer</h1>
            <p><br>Diammonium Phosphate<br></p>
            <img src="https://royalglobalenergy.com/wp-content/uploads/2019/05/diammonium-phosphate-dap_39396-14.jpg" alt="Recommended Fertilizer" class="square-image">
        </div>
    </div>

    <footer>
        Made with <span style="color: red;">&hearts;</span> B.Tech AI & DS
    </footer>
</body>

</html>


        '''
    else:
        result = '''
        <!DOCTYPE html>
<html>

<head>
    <title>Results Page</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&family=Poppins:wght@600&display=swap" rel="stylesheet">
    <style>
        body{
            background-color: aquamarine;
        }
        .column-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .column {
            flex: 1;
            text-align: center;
            padding: 20px;
        }

        h1 {
            font-family: 'Libre Baskerville';
        }

        p {
            font-family: 'Poppins';
            font-size: 18px;
        }

        .square-image {
            width: 300px;
            height: 300px;
            object-fit: cover;
        }

        footer {
            font-family: 'poppins';
            text-align: center;
            padding: 20px;
            background-color: #333;
            color: white;
        }
    </style>
</head>

<body>
    <div class="column-container">
        <div class="column">
            <h1>Your Soil<br> Type</h1>
            <p><br>It's a Yellow Soil<br></p>
            <img src="https://media.istockphoto.com/id/1227007177/photo/yellow-sand-dry-soil-texture-and-background-the-background-of-the-red-soil-abstract-land.jpg?s=170667a&w=0&k=20&c=nMfv79Y3UGqQi8dLHRfVpnhWBBHZVWKEY09Kbdm_NtE=" alt="Soil Type" class="square-image">
        </div>
        <div class="column">
            <h1>Recommended <br>Crops</h1>
            <p><br>Best suited to grow Orange<br></p>
            <img src="https://sethlui.com/wp-content/uploads/2021/06/Oranges.jpg" alt="Recommended Crops" class="square-image">
        </div>
        <div class="column">
            <h1>Recommended <br>Fertilizer</h1>
            <p><br>NPK<br></p>
            <img src="https://www.dfgrupo.com/wp-content/uploads/2022/04/NPK-Nitrofoska-DFGRUPO.jpg" alt="Recommended Fertilizer" class="square-image">
        </div>
    </div>

    <footer>
        Made with <span style="color: red;">&hearts;</span> B.Tech AI & DS
    </footer>
</body>

</html>


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
    <!DOCTYPE html>
<html>

<head>
    <title>Farm Smart - Soil Recognition AI</title>
    <link rel="stylesheet" type="text/css" href="style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap" rel="stylesheet">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: poppins, sans-serif;
            background-color: #8DDFCB;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-size: cover;
        }

        .logo {
            font-size: 36px;
            margin-top: 100px;
            text-align: center;
        }

        .hero {
            text-align: center;
            padding: 100px 100px;
        }

        .hero h1 {
            font-family: 'Libre Baskerville', serif;
            font-size: 36px;
            margin-bottom: 20px;
        }

        .hero p {
            font-size: 18px;
            margin-bottom: 40px;
        }

        .hero img {
            max-width: 60%;
            height: auto;
        }

        .form-container {
            text-align: center;
            padding: 50px 0;
        }
        .form-container{
            font-family: 'Libre Baskerville', serif;
        }

        .upload-form {
            margin: 0 auto;
            max-width: 100px;
        }

        .upload-form input[type="file"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
        }

        .upload-form input[type="submit"] {
            background-color: #333;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }

        .upload-form input[type="submit"]:hover {
            background-color: #555;
        }
        .mission-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 100px 100px;
        }

        .mission-text {
            flex: 1;
            text-align: left;
            padding: 0 20px;
        }

        .mission-text h2 {
            font-family: 'Libre Baskerville', serif;
            font-size: 24px;
            margin-bottom: 20px;
        }

        .mission-text p {
            font-size: 18px;
            margin-bottom: 20px;
        }

        .mission-image {
            flex: 1;
            max-width: 60%;
        }

        .access-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 100px 100px;
            
        }

        .access-image {
            flex: 1;
            max-width: 60%;
        }

        .access-text {
            
            flex: 1;
            text-align: left;
            padding: 0 20px;
        }

        .access-text h2 {
            font-family: 'Libre Baskerville', serif;
            font-size: 24px;
            margin-bottom: 20px;
        }

        .access-text p {
            font-size: 18px;
            margin-bottom: 20px;
        }
        footer {
            font-family: 'poppins';
            text-align: center;
            padding: 20px;
            background-color: #333;
            color: white;
        }
    </style>
</head>

<body>
    <div class="logo">FARM SMART</div>

    <div class="hero">
        <h1>Introducing our cutting-edge Soil Recognition AI System!</h1>
        <p>With this revolutionary technology, you can effortlessly determine the composition of your soil just by uploading an image.<br>But that's not all – our system goes the extra mile by offering tailored recommendations for crop selection <br>and fertilizer application to optimize your agricultural endeavors.</p>
        <img src="https://greenerideal.com/wp-content/uploads/2023/08/smart-farm-monitoring.jpg" alt="Soil Recognition AI" />
    </div>

    <div class="form-container">
        <h2>Upload Your Soil Image</h2>
        <form action="" method="post" enctype="multipart/form-data" class="upload-form">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    </div>

    <div class="mission-container">
        <div class="mission-text">
            <h2>Mission we are working on</h2>
            <p>With our AI model, farmers can boost their agricultural efficiency, reduce resource wastage, and increase overall profitability. <br><br>It empowers them to make data-driven decisions and adapt to changing environmental conditions, enhancing sustainability.</p>
            <ul>
                <li><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-camera-search" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
   <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
   <path d="M11.5 20h-6.5a2 2 0 0 1 -2 -2v-9a2 2 0 0 1 2 -2h1a2 2 0 0 0 2 -2a1 1 0 0 1 1 -1h6a1 1 0 0 1 1 1a2 2 0 0 0 2 2h1a2 2 0 0 1 2 2v2.5"></path>
   <path d="M14.757 11.815a3 3 0 1 0 -3.431 4.109"></path>
   <path d="M18 18m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0"></path>
   <path d="M20.2 20.2l1.8 1.8"></path>
</svg>   Upload Image</li>
                <li><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-bulb-filled" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
   <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
   <path d="M4 11a1 1 0 0 1 .117 1.993l-.117 .007h-1a1 1 0 0 1 -.117 -1.993l.117 -.007h1z" stroke-width="0" fill="currentColor"></path>
   <path d="M12 2a1 1 0 0 1 .993 .883l.007 .117v1a1 1 0 0 1 -1.993 .117l-.007 -.117v-1a1 1 0 0 1 1 -1z" stroke-width="0" fill="currentColor"></path>
   <path d="M21 11a1 1 0 0 1 .117 1.993l-.117 .007h-1a1 1 0 0 1 -.117 -1.993l.117 -.007h1z" stroke-width="0" fill="currentColor"></path>
   <path d="M4.893 4.893a1 1 0 0 1 1.32 -.083l.094 .083l.7 .7a1 1 0 0 1 -1.32 1.497l-.094 -.083l-.7 -.7a1 1 0 0 1 0 -1.414z" stroke-width="0" fill="currentColor"></path>
   <path d="M17.693 4.893a1 1 0 0 1 1.497 1.32l-.083 .094l-.7 .7a1 1 0 0 1 -1.497 -1.32l.083 -.094l.7 -.7z" stroke-width="0" fill="currentColor"></path>
   <path d="M14 18a1 1 0 0 1 1 1a3 3 0 0 1 -6 0a1 1 0 0 1 .883 -.993l.117 -.007h4z" stroke-width="0" fill="currentColor"></path>
   <path d="M12 6a6 6 0 0 1 3.6 10.8a1 1 0 0 1 -.471 .192l-.129 .008h-6a1 1 0 0 1 -.6 -.2a6 6 0 0 1 3.6 -10.8z" stroke-width="0" fill="currentColor"></path>
</svg>   Get Recommendation</li>
                <li><svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-rotate-clockwise-2" width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
   <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
   <path d="M9 4.55a8 8 0 0 1 6 14.9m0 -4.45v5h5"></path>
   <path d="M5.63 7.16l0 .01"></path>
   <path d="M4.06 11l0 .01"></path>
   <path d="M4.63 15.1l0 .01"></path>
   <path d="M7.16 18.37l0 .01"></path>
   <path d="M11 19.94l0 .01"></path>
</svg>   Get Started</li>
            </ul>

        </div>
        <div class="mission-image">
            <img src="https://www.iberdrola.com/documents/20125/40267/SmartFarming_746x419.jpg/48e1b07b-1a90-7d42-98b0-3d41e508ce7f?t=1627035238757" alt="Mission Image" />
        </div>
    </div>

    <div class="access-container">
        <div class="access-image">
            <img src="https://sbnri.com/blog/wp-content/uploads/2023/01/iStock-543212762.jpg" alt="Access Image" />
        </div>
        <div class="access-text">
            <h2>Everyone should have access to smart farming</h2>
            <p>Our model is continuously updated with the latest agricultural research and data, ensuring that farmers always have access to the most relevant and accurate information for their specific needs.</p>
            <p>Embracing our AI model in farming is not just a step forward; it's a leap towards a more sustainable and productive future in agriculture, benefiting both farmers and the global food supply chain.</p>
        </div>
    </div>
    <footer>
        Made with <span style="color: red;">&hearts;</span> B.Tech AI & DS
    </footer>

</body>

</html>


    '''



@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    result = process_image(file)
    response = {'result': result}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)