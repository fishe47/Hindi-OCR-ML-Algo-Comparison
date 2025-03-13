from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import pickle
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Load the CNN model
cnn_model = load_model("best_cnn_model.keras")  

# Load the trained LightGBM and XGBoost models
with open("lgbm_model.pkl", "rb") as f:
    lgbm_model = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Define the labels (characters or classes)
labels = ["ka", "kha", "ga", "gha", "kna", "cha", "chha", "ja", "jha", "yna", "t`a", "t`ha",
          "d`a", "d`ha", "adna", "ta", "tha", "da", "dha", "na", "pa", "pha", "ba", "bha",
          "ma", "yaw", "ra", "la", "waw", "sha", "shat", "sa", "ha", "aksha", "tra", "gya"]

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))
    image = image.astype("float") / 255.0
    return image

def predict_with_all_models(image_path):
    # Preprocess image
    image = preprocess_image(image_path)
    cnn_input = np.expand_dims(img_to_array(image), axis=0)

    # CNN Prediction
    cnn_prediction = cnn_model.predict(cnn_input)
    cnn_label = labels[np.argmax(cnn_prediction)]

    # Flatten the image for LightGBM and XGBoost
    flat_image = image.flatten().reshape(1, -1)

    # LightGBM Prediction
    lgbm_label = labels[lgbm_model.predict(flat_image)[0]]

    # XGBoost Prediction
    xgb_label = labels[xgb_model.predict(flat_image)[0]]

    # Print the predictions from all three models
    print(f"CNN Prediction: {cnn_label}")
    print(f"LightGBM Prediction: {lgbm_label}")
    print(f"XGBoost Prediction: {xgb_label}")

# Test image path
image_path = r"C:\Users\Sriraj.k.k.Sarkar\Desktop\imp stuff\Hindi-OCR-master\Hindi-OCR-minor-project\DevanagariHandwrittenCharacterDataset\Train\character_21_pa\1637.png"
# Make predictions with all models
predict_with_all_models(image_path)
