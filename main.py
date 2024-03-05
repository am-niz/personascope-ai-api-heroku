from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
import io  # Needed for io.BytesIO

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

# Load your models
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model(r'my_model (1).h5')
age_model = tf.keras.models.load_model(r'alancheri_age_model_50epochs.h5')
emotion_model = tf.keras.models.load_model(r'emotion_detection_model_100epochs.h5')
gender_labels = ['Male', 'Female']
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict_gender_from_image(image_data, model, face_classifier):
    pil_image = Image.open(io.BytesIO(image_data))

    # Load the image from the path
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) 

    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    gender = None
    
    # Loop over the faces detected
    for (x, y, w, h) in faces:
        
        # Extract the region of interest (ROI) in grayscale
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize the ROI for the gender model
        roi_resized = cv2.resize(roi_gray, (128, 128), interpolation=cv2.INTER_AREA)
        roi_reshaped = roi_resized.reshape(1, 128, 128, 1)  # Reshape for the model
        
        # Predict the gender
        gender_predict = model.predict(roi_reshaped)
        gender = gender_labels[round(gender_predict[0][0][0])]

    if gender is None:
        gender = "Unknown"
        
    return gender


def predict_age_from_image(image_path, age_model, face_classifier):

    pil_image = Image.open(io.BytesIO(image_path))

    # Load the image from the path
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) 

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    age = None
    
    
    # Process each face found
    for (x, y, w, h) in faces:

        # Draw a rectangle around the face (optional, to visualize)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region from the original (colored) image
        roi_color = frame[y:y+h, x:x+w]
        
        # Resize the extracted face region to the expected input size of your age model
        roi_resized = cv2.resize(roi_color, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Normalize the pixels in the face region
        roi_normalized = roi_resized.astype('float32') / 255.0
        
        # Expand dimensions to match the model's input shape
        roi_ready = np.expand_dims(roi_normalized, axis=0)
        
        # Predict the age
        age_predict = age_model.predict(roi_ready)
        age = round(age_predict[0, 0])

    if age is None:
        age = -1     
        
    return age


def predict_emotion_from_image(image_path, emotion_model, face_classifier):

    pil_image = Image.open(io.BytesIO(image_path))

    # Load the image from the path
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) 
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    emotion = None
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Extract the region of interest (ROI) in grayscale
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Preprocess the ROI for the emotion model
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        
        # Predict the emotion
        preds = emotion_model.predict(roi)[0]
        emotion = class_labels[preds.argmax()]

    if emotion is None:
        emotion = "Unknown"
        
    return emotion



@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()  # Read image bytes

        # Ensure image_array is processed correctly if your model expects a specific shape, color channels, etc.

        gender = predict_gender_from_image(contents, model, face_classifier)
        age = predict_age_from_image(contents, age_model, face_classifier)
        emotion = predict_emotion_from_image(contents, emotion_model, face_classifier)

        return JSONResponse(content={"Gender": gender, "Age": age, "Emotion": emotion}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)

