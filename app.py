from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Call analysis functions here
        results = {
            'segmentation': skin_segmentation(filepath),
            'texture_analysis': skin_texture_analysis(filepath),
            'uv_damage': uv_damage_detection(filepath),
            'mole_counter': mole_counter(filepath),
            'tone_analysis': skin_tone_analysis(filepath),
            'burn_severity': burn_severity_estimation(filepath),
            'scar_detection': scar_detection(filepath),
            'crack_detection': skin_crack_detection(filepath),
            'moisture_visualization': skin_moisture_visualization(filepath),
            'pore_detection': skin_pore_detection(filepath)
        }

        return render_template('results.html', results=results, uploaded_image=filepath)
    return jsonify({'error': 'Invalid file format'})

# Analysis Functions

def skin_segmentation(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented.jpg')
    cv2.imwrite(segmented_path, segmented)
    return {'output_image': segmented_path}

def skin_texture_analysis(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    texture_path = os.path.join(app.config['UPLOAD_FOLDER'], 'texture.jpg')
    cv2.imwrite(texture_path, edges)
    return {'output_image': texture_path}

def uv_damage_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    uv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uv_damage.jpg')
    cv2.imwrite(uv_path, enhanced)
    return {'output_image': uv_path}

def mole_counter(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mole_count = len(contours)
    return {'count': mole_count}

def skin_tone_analysis(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv[:, :, 0])
    return {'average_hue': avg_color}

def burn_severity_estimation(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    severity = np.sum(thresh > 0) / (thresh.size) * 100
    return {'severity_percentage': severity}

def scar_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    scar_path = os.path.join(app.config['UPLOAD_FOLDER'], 'scar.jpg')
    cv2.imwrite(scar_path, edges)
    return {'output_image': scar_path}

def skin_crack_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cracks = cv2.Canny(image, 100, 200)
    crack_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cracks.jpg')
    cv2.imwrite(crack_path, cracks)
    return {'output_image': crack_path}

def skin_moisture_visualization(image_path):
    image = cv2.imread(image_path)
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    moisture_path = os.path.join(app.config['UPLOAD_FOLDER'], 'moisture.jpg')
    cv2.imwrite(moisture_path, enhanced)
    return {'output_image': moisture_path}

def skin_pore_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pore_count = len(contours)
    return {'count': pore_count}



# Import necessary libraries
#from flask import Flask, render_template, request
from keras.models import load_model
#import numpy as np
#import cv2
#import os


# Load the Keras model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()



# Define the route for processing the image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded. Please upload an image."

    file = request.files['file']

    if file.filename == '':
        return "No selected file."

    if file:
        # Save the uploaded image temporarily
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Process the image
        image = cv2.imread(filepath)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1

        # Predict using the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Return the result
        return render_template(
            'result2.html', 
            class_name=class_name, 
            confidence_score=round(confidence_score * 100, 2),
            image_path=filepath
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
