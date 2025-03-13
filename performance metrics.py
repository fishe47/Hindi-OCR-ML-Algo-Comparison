import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
cnn_model = load_model("cnn_model_final.keras")
with open("lgbm_model.pkl", "rb") as f:
    lgbm_model = pickle.load(f)
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Define test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
testGenerator = test_datagen.flow_from_directory(
    "DevanagariHandwrittenCharacterDataset/Test",
    target_size=(32, 32),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False
)

# Flatten data for LightGBM and XGBoost
def flatten_data(generator):
    data, labels = [], []
    for batch_images, batch_labels in generator:
        flat_images = batch_images.reshape(batch_images.shape[0], -1)
        data.extend(flat_images)
        labels.extend(np.argmax(batch_labels, axis=1))
        if len(data) >= generator.samples:
            break
    return np.array(data), np.array(labels)

X_test, y_test = flatten_data(testGenerator)

# Helper function to evaluate models
def evaluate_model(name, y_true, y_pred, inference_time):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"--- {name} Performance Metrics ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print("\n")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix for {name}:\n", cm)
    plot_confusion_matrix(cm, name)
    
    return accuracy, precision, recall, f1

# Helper function to plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=testGenerator.class_indices.keys(), yticklabels=testGenerator.class_indices.keys())
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Evaluate CNN Model
start_time = time.time()  # Record start time for CNN inference
cnn_predictions = cnn_model.predict(testGenerator)
cnn_predicted_classes = np.argmax(cnn_predictions, axis=1)
cnn_inference_time = time.time() - start_time  # Calculate inference time
evaluate_model("CNN", y_test, cnn_predicted_classes, cnn_inference_time)

# Evaluate LightGBM Model
start_time = time.time()  # Record start time for LightGBM inference
lgbm_predictions = lgbm_model.predict(X_test)
lgbm_inference_time = time.time() - start_time  # Calculate inference time
evaluate_model("LightGBM", y_test, lgbm_predictions, lgbm_inference_time)

# Evaluate XGBoost Model
start_time = time.time()  # Record start time for XGBoost inference
xgb_predictions = xgb_model.predict(X_test)
xgb_inference_time = time.time() - start_time  # Calculate inference time
evaluate_model("XGBoost", y_test, xgb_predictions, xgb_inference_time)
