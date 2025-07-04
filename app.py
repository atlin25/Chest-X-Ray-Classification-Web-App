from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import base64
from PIL import Image
import io
import pydicom
from skimage.transform import resize
import google.generativeai as genai
import traceback
import base64
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

genai.configure(api_key="AIzaSyBl7YflSQYesVVXqzlYRfoGO0owfWAhgrs")
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-flash") # pro if i have the quota for it

CLASS_LABELS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis"
]

app = Flask(__name__)
CORS(app)

def focal_loss(gamma=2.0, alpha=0.35):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    return loss

model = load_model("best_model.h5", custom_objects={"loss": focal_loss(gamma=2.0, alpha=0.35)})

def preprocess_image(image_bytes):
    try:
        dicom = pydicom.dcmread(io.BytesIO(image_bytes))
        img = dicom.pixel_array.astype(np.float32)
        img_range = np.max(img) - np.min(img)
        if img_range == 0:
            img = np.zeros_like(img)
        else:
            img = (img - np.min(img)) / img_range
        img_resized = resize(img, (256, 256), anti_aliasing=True)
        img_rgb = np.stack([img_resized]*3, axis=-1)
    except Exception as e:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((256, 256))
            img_rgb = np.array(img).astype(np.float32) / 255.0
        except Exception as e2:
            raise ValueError(f"Unsupported image format. DICOM error: {e}, PIL error: {e2}")

    return np.expand_dims(img_rgb, axis=0).astype(np.float32)

def generate_summary(prediction):
    if prediction is None or not np.isfinite(prediction).all():
        return "Sorry, the prediction could not be interpreted. Please try again with a valid image."

    prediction_text = "\n".join(
        f"- {label}: {round(score * 100, 1)}%"
        for label, score in zip(CLASS_LABELS, prediction)
    )

    prompt = f"""
    A medical image classification model processed a chest X-ray and returned the following prediction:
    {prediction_text}

    Write a short, friendly summary of what this means for a non-technical user. Suggest what they should do next in a responsible tone. Avoid giving a definitive diagnosis.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini API error:", e)
        return "An error occurred while generating insights. Please try again later."

@app.route('/api/classify', methods=['POST'])

def classify():
    data = request.get_json()
    if 'image_base64' not in data:
        return jsonify({'error': 'No image_base64 provided'}), 400

    try:
        # Extract base64 part (remove prefix if present)
        header, encoded = data['image_base64'].split(',', 1) if ',' in data['image_base64'] else (None, data['image_base64'])
        
        # Fix base64 padding here:
        encoded = fix_base64_padding(encoded)

        # Now decode safely
        image_bytes = base64.b64decode(encoded)

        input_tensor = preprocess_image(image_bytes)
        raw_prediction = model.predict(input_tensor)[0]
        print("Raw prediction:", raw_prediction)

        prediction_list = raw_prediction.tolist()
        summary = generate_summary(prediction_list)
        confidence = round(float(max(raw_prediction)) * 100, 2)
        primary_label = CLASS_LABELS[raw_prediction.argmax()]

        return jsonify({
            'prediction': prediction_list,
            'labels': CLASS_LABELS,
            'summary': summary,
            'confidence': f"{confidence}%",
            'primary_label': primary_label
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def fix_base64_padding(b64_string):
    b64_string = b64_string.strip()
    missing_padding = len(b64_string) % 4
    if missing_padding != 0:
        b64_string += '=' * (4 - missing_padding)
    return b64_string

@app.route('/api/gemini_insights', methods=['POST'])
def gemini_insights():
    data = request.get_json()

    prediction = data.get('prediction')
    labels = data.get('labels')
    summary = data.get('summary')

    # Compose prompt for Gemini model
    prompt = f"""
    The model predicted the following on a chest X-ray:
    {dict(zip(labels, prediction))}
    Summary:
    {summary}

    Provide a clear, responsible medical insight based on these results.
    """

    # Call your Gemini model API or function here.
    # For example, if you have a Gemini client object:
    response = gemini_model.generate_content(prompt)

    # Extract the text from the response (adjust depending on your Gemini API client)
    insight_text = response.text.strip()

    return jsonify({'insight': insight_text})

if __name__ == "__main__":
    app.run(debug=True)

