# Import required libraries for the application
import streamlit as st  # Streamlit is used to create a web app interface
import cv2  # OpenCV is used for image processing
import numpy as np  # NumPy is used for numerical operations on arrays
import os  # OS module is used for file and directory operations
from werkzeug.utils import secure_filename  # Securely handle file names
from keras.models import load_model  # Load pre-trained Keras models
import gdown  # Used for downloading files from Google Drive

# Load Keras model and labels
# Google Drive file ID for the pre-trained model
file_id = "18g6wdMpRS81GXpqIDY4PazCpXy_apUiz"  
# Define the output filename for the downloaded model
output = "keras_Model.h5"

# Download the model from Google Drive
gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", output, quiet=False)

# Read class labels from a text file
class_names = open("labels.txt", "r").readlines()

# Load the pre-trained model without re-compiling it
model = load_model(output, compile=False)

# Define the folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
# Define allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to display the uploaded image and analysis results
def display_results(image_path, results):
    st.image(image_path, caption="Uploaded Image", use_column_width=True)  # Show uploaded image
    for key, value in results.items():
        if isinstance(value, dict) and 'output_image' in value:
            st.image(value['output_image'], caption=key, use_column_width=True)  # Display processed images
        else:
            st.write(f"{key}: {value}")  # Show analysis results as text

# Function for skin segmentation analysis
def skin_segmentation(image_path):
    image = cv2.imread(image_path)  # Read the uploaded image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)  # Define lower bound for skin color
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)  # Define upper bound for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)  # Create a mask for skin regions
    segmented = cv2.bitwise_and(image, image, mask=mask)  # Apply the mask to the image
    segmented_path = os.path.join(UPLOAD_FOLDER, 'segmented.jpg')  # Save path for segmented image
    cv2.imwrite(segmented_path, segmented)  # Save the segmented image
    return {'output_image': segmented_path}  # Return the result path

# Function for skin texture analysis using edge detection
def skin_texture_analysis(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    edges = cv2.Canny(image, 100, 200)  # Apply Canny edge detection
    texture_path = os.path.join(UPLOAD_FOLDER, 'texture.jpg')  # Save path for texture analysis image
    cv2.imwrite(texture_path, edges)  # Save the texture analysis image
    return {'output_image': texture_path}

# Function for UV damage detection
def uv_damage_detection(image_path):
    image = cv2.imread(image_path)  # Read the uploaded image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    enhanced = cv2.equalizeHist(gray)  # Enhance the contrast
    uv_path = os.path.join(UPLOAD_FOLDER, 'uv_damage.jpg')  # Save path for UV analysis image
    cv2.imwrite(uv_path, enhanced)  # Save the UV damage analysis image
    return {'output_image': uv_path}


# Function for skin tone analysis
def skin_tone_analysis(image_path):
    image = cv2.imread(image_path)  # Read the uploaded image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    avg_color = np.mean(hsv[:, :, 0])  # Calculate average hue value
    return {'average_hue': avg_color}  # Return average hue as skin tone

# Function for burn severity estimation
def burn_severity_estimation(image_path):
    image = cv2.imread(image_path)  # Read the uploaded image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # Threshold for burn regions
    severity = np.sum(thresh > 0) / (thresh.size) * 100  # Calculate percentage of burn severity
    return {'severity_percentage': severity}

# Function for scar detection
def scar_detection(image_path):
    image = cv2.imread(image_path)  # Read the uploaded image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)  # Define lower bound for skin color
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)  # Define upper bound for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)  # Create a mask for skin regions
    segmented = cv2.bitwise_and(image, image, mask=mask)  # Apply the mask to the image
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    edges = cv2.Canny(gray, 50, 150)  # Apply edge detection
    scar_path = os.path.join(UPLOAD_FOLDER, 'scar.jpg')  # Save path for scar detection image
    cv2.imwrite(scar_path, edges)  # Save the scar detection image
    return {'output_image': scar_path}


# Function for visualizing skin moisture
def skin_moisture_visualization(image_path):
    image = cv2.imread(image_path)  # Read the uploaded image
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)  # Enhance image details
    moisture_path = os.path.join(UPLOAD_FOLDER, 'moisture.jpg')  # Save path for moisture visualization
    cv2.imwrite(moisture_path, enhanced)  # Save the moisture visualization image
    return {'output_image': moisture_path}



# Streamlit app UI
st.title('Skin Analysis and Prediction')  # App title

# File uploader widget in Streamlit
file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if file:  # If a file is uploaded
    # Save the uploaded file securely
    filename = secure_filename(file.name)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(file.getbuffer())

    # Display the uploaded image
    st.image(filepath, caption="Uploaded Image", use_column_width=True)

    # Perform different skin analysis tasks
    results = {
        'segmentation': skin_segmentation(filepath),
        'texture_analysis': skin_texture_analysis(filepath),
        'uv_damage': uv_damage_detection(filepath),
        'tone_analysis': skin_tone_analysis(filepath),
        'burn_severity': burn_severity_estimation(filepath),
        'scar_detection': scar_detection(filepath),
        'moisture_visualization': skin_moisture_visualization(filepath),
    
    }

    # Display analysis results
    display_results(filepath, results)

    # Prepare the image for prediction
    image = cv2.imread(filepath)  # Read the uploaded image
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)  # Resize to model input size
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)  # Reshape for model input
    image = (image / 127.5) - 1  # Normalize pixel values

    # Predict the skin condition using the pre-trained model
    prediction = model.predict(image)
    index = np.argmax(prediction)  # Get the index of the highest probability
    class_name = class_names[index].strip()  # Get the corresponding class label
    confidence_score = prediction[0][index]  # Get the confidence score

    # Display prediction results
    st.write(f"Prediction: {class_name}")
    st.write(f"Confidence: {confidence_score * 100:.2f}%")
