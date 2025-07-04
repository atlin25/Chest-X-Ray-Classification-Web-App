# Chest X-Ray Classification Web App
A deep learning model for classifying chest X-ray images to detect abnormalities using the [VinBigData Chest X-ray Abnormalities Detection dataset](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection).

## Overview
This project consists of a custom-trained CNN model to classify chest X-ray images for 14 abnormalities and a React + Flask web interface that allows users to upload X-rays, receive predictions, and get detailed language-based insights. The model targets 99% accuracy but currently achieves ~80% due to class imbalance, dataset validity, dataset size, and GPU limitations. Key features include:
- Multi-label classification using TensorFlow/Keras with focal loss and data augmentation.
- Integration with Gemini Pro API for user-friendly prediction summaries.
- Flask/React web app for image uploads and result visualization.

## Model Architecture and Pipeline
1. Preprocessing
   - Loads and preprocesses DICOM, Jpeg, and Png formats.
   - Applies normalization, resizing, and caching.
   - Data Augmentation to address imbalance: rotation, contrast, brightness, flipping.
   - multi-label stratified K-fold splitting for balanced train/val sets.
2. Custom CNN
   - Built using TensorFlow and Keras with a standard structure.
   - Conv-BatchNorm-ReLU blocks.
   - Global average pooling.
   - Dropout regularization.
   - Fully connected layer with sigmoid output for multi-label prediction.
3. Additional Techniques (addressing imbalance)
   - Focal Loss to counter class imbalance.
   - Class weights dependant on label frequency.
   - Callbacks such as EarlyStopping, Learning rate reduction on plateau, Model checkpointing for best validation loss.
   - Threshold optimization, fine tuning training metrics (AUC, PR AUC, Precision, Recall).

## Installation and Setup
### Prerequisites
- Python 3.10
- Git

### Dependencies
The project requires the following Python libraries (tested versions):
- `tensorflow==2.11.0`
- `numpy==1.26.4`
- `pandas==2.2.3`
- `pydicom==3.0.1`
- `scikit-learn==1.6.1`
- `scikit-image==0.25.2`
- `pillow==11.2.1`
- `google-generativeai==0.8.5`

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<atlin25>/<Chest-X-ray-Classification-Web-App>.git
   cd <Chest-X-ray-Classification-Web-App>
2. Backend Setup
   - python3 -m venv venv
   - source venv/bin/activate # On Windows: venv\Scripts\activate
3. Frontend Setup
   - cd frontend
   - npm install
4. To Run App
   - source venv/bin/activate
   - cd backend
   - python app.py
5. In another Session
   - cd frontend
   - npm start


