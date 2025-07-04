import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pydicom
from PIL import Image
from skimage.transform import resize
import random
from scipy.ndimage import rotate
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# === Paths and constants ===
data_dir = "/home/tkmra/MLL/Capstone/vinbigdata-chest-xray-abnormalities-detection"
train_img_dir = os.path.join(data_dir, "train")
csv_path = os.path.join(data_dir, "trimmed_train.csv")
cache_dir = os.path.join(data_dir, "cache")
model_dir = os.path.join(data_dir, "model")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 16

# === Load labels and binarize ===
labels = pd.read_csv(csv_path)
class_names = sorted([c for c in labels['class_name'].unique() if c != 'No finding'])
num_classes = len(class_names)

mlb = MultiLabelBinarizer(classes=class_names)
mlb.fit([class_names])

grouped = labels.groupby('image_id')['class_name'].apply(list).reset_index()
grouped['class_name'] = grouped['class_name'].apply(lambda x: [label for label in x if label != 'No finding'])

image_id = grouped['image_id'].values
encoded_labels = mlb.transform(grouped['class_name'])

# === Compute class weights ===
def get_class_weights(encoded_labels):
    label_counts = np.sum(encoded_labels, axis=0)
    total = encoded_labels.shape[0]
    class_weights = total / (label_counts + 1e-6)
    return tf.constant(class_weights, dtype=tf.float32)

class_weights_tensor = get_class_weights(encoded_labels)

# === Weighted binary crossentropy ===
def weighted_binary_crossentropy(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    loss = - (
        class_weights_tensor * y_true * tf.math.log(y_pred) +
        (1 - y_true) * tf.math.log(1 - y_pred)
    )
    return tf.reduce_mean(loss)

# === Preprocessing & caching ===
def load_and_preprocess(image_id):
    cache_path = os.path.join(cache_dir, image_id + ".npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    # Detect file extension
    possible_exts = [".dicom", ".dcm", ".jpg", ".jpeg", ".png"]
    for ext in possible_exts:
        img_path = os.path.join(train_img_dir, image_id + ext)
        if os.path.exists(img_path):
            break
    else:
        raise FileNotFoundError(f"No supported image file found for ID {image_id}")

    # Load image based on type
    if ext in [".dicom", ".dcm"]:
        dicom = pydicom.dcmread(img_path)
        img = dicom.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize 0-1
    else:
        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0     # Normalize 0-1

    # Resize and prepare 3-channel image
    img_resized = resize(img, (256, 256), anti_aliasing=True)
    if img_resized.ndim == 2:  # Grayscale
        img_resized = np.stack([img_resized]*3, axis=-1)

    np.save(cache_path, img_resized)
    return img_resized

def augment_image(img_resized):
    # Horizontal flip
    if random.random() < 0.5:
        img_resized = np.fliplr(img_resized)
    
    # Vertical flip
    if random.random() < 0.5:
        img_resized = np.flipud(img_resized)

    # Random rotation
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        img_resized = rotate(img_resized, angle, reshape=False, mode='nearest')

    # Brightness adjustment
    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        img_resized = np.clip(img_resized * factor, 0.0, 1.0)

    # Contrast adjustment
    if random.random() < 0.5:
        mean = np.mean(img_resized)
        contrast_factor = random.uniform(0.8, 1.2)
        img_resized = np.clip((img_resized - mean) * contrast_factor + mean, 0.0, 1.0)

    return img_resized
def data_generator(image_ids, labels, batch_size):
    dataset_size = len(image_ids)
    while True:
        idxs = np.arange(dataset_size)
        np.random.shuffle(idxs)
        for start in range(0, dataset_size, batch_size):
            batch_idx = idxs[start:start+batch_size]
            batch_images = []
            batch_labels = []
            for i in batch_idx:
                img = load_and_preprocess(image_ids[i])
                # augment with 50% chance
                if np.random.rand() < 0.5:
                    img = augment_image(img)
                batch_images.append(img)
                batch_labels.append(labels[i])
            yield np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)

# === Train/Validation Split ===
mskf = MultilabelStratifiedKFold(n_splits=5, random_state=42, shuffle=True)
train_idx, val_idx = next(mskf.split(image_id, encoded_labels))
train_ids, val_ids = image_id[train_idx], image_id[val_idx]
train_labels, val_labels = encoded_labels[train_idx], encoded_labels[val_idx]

train_gen = data_generator(train_ids, train_labels, BATCH_SIZE)
val_gen = data_generator(val_ids, val_labels, BATCH_SIZE)

steps_per_epoch = len(train_ids) // BATCH_SIZE
validation_steps = len(val_ids) // BATCH_SIZE

# === Custom CNN Model ===
def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding = 'same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), padding = 'same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), padding = 'same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), padding = 'same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model

# === Compile and Train ===
model = create_model()
def focal_loss(gamma=2.0, alpha=0.35):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    return loss

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=focal_loss(gamma=2.0, alpha=0.35),
    metrics=[
    tf.keras.metrics.AUC(),
    tf.keras.metrics.Precision(thresholds=0.4),
    tf.keras.metrics.Recall(thresholds=0.3),
    tf.keras.metrics.AUC(curve='PR', name='pr_auc')]
)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,               # number of epochs with no improvement before stopping
    restore_best_weights=True # restores weights from the best epoch
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
)

checkpoint_cb = ModelCheckpoint(
    filepath='best_model.h5',     # You can change the name/path as needed
    monitor='val_loss',           # You can also use 'val_auc' or 'val_pr_auc'
    save_best_only=True,          # Save only the best model
    save_weights_only=False,      # Save the full model
    mode='min',                   # Use 'min' for loss, 'max' for metrics like AUC
    verbose=1                     # Print when saving
)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[early_stopping, reduce_lr, checkpoint_cb]
)



model_save_path = os.path.join(model_dir, "multi_label")

import matplotlib.pyplot as plt

def plot_training(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for metric in metrics:
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(metric.upper())
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.show()

plot_training(history)

# === Run model on validation set without TTA ===
val_preds = []
val_true = []

for img_id, true_labels in zip(val_ids, val_labels):
    img = load_and_preprocess(img_id)
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch, verbose=0)[0]
    val_preds.append(pred)
    val_true.append(true_labels)

val_preds = np.array(val_preds)
val_true = np.array(val_true)

# === Search optimal thresholds per class based on F1 score ===
#optimal_thresholds = []
#for i in range(num_classes):
#    best_thresh = 0.5
#    best_f1 = 0
#    for thresh in np.linspace(0.1, 0.9, 81):  # Try thresholds from 0.1 to 0.9
#        preds_bin = (val_preds[:, i] >= thresh).astype(int)
#        f1 = f1_score(val_true[:, i], preds_bin, zero_division=0)
#        if f1 > best_f1:
#            best_f1 = f1
#            best_thresh = thresh
#    optimal_thresholds.append(best_thresh)
#
#optimal_thresholds = np.array(optimal_thresholds)
#print("\nOptimal thresholds per class:")
#for cls, thresh in zip(class_names, optimal_thresholds):
#    print(f"{cls}: {thresh:.2f}")
#
## === Final evaluation using optimal thresholds ===
#final_preds_bin = (val_preds >= optimal_thresholds).astype(int)
#
#final_auc = roc_auc_score(val_true, val_preds, average='macro')
#final_precision = precision_score(val_true, final_preds_bin, average='macro', zero_division=0)
#final_recall = recall_score(val_true, final_preds_bin, average='macro', zero_division=0)
#
#print("\nEvaluation using optimal thresholds:")
#print(f"AUC:       {final_auc:.4f}")
#print(f"Precision: {final_precision:.4f}")
#print(f"Recall:    {final_recall:.4f}")
#
