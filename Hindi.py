import os
import numpy as np
import h5py
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Define data generators
trainDataGen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1.0/255)

trainGenerator = trainDataGen.flow_from_directory(
    "DevanagariHandwrittenCharacterDataset/Train",
    target_size=(32, 32),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

testGenerator = test_datagen.flow_from_directory(
    "DevanagariHandwrittenCharacterDataset/Test",
    target_size=(32, 32),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

# CNN Model Definition
cnn_model = Sequential([
    Conv2D(32, (3, 3), input_shape=(32, 32, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(trainGenerator.num_classes, activation='softmax')
])

cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for Early Stopping and Model Checkpointing
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_cnn_model.keras", save_best_only=True, monitor='val_accuracy')

]

# Train the CNN model with callbacks
cnn_model.fit(trainGenerator, epochs=30, validation_data=testGenerator, callbacks=callbacks)

# Save the trained CNN model
cnn_model.save("cnn_model_final.keras")

# Flatten images for LightGBM and XGBoost
def flatten_data(generator, batch_size=32):
    data, labels = [], []
    for batch_images, batch_labels in generator:
        flat_images = batch_images.reshape(batch_images.shape[0], -1)
        data.extend(flat_images)
        labels.extend(np.argmax(batch_labels, axis=1))
        if len(data) >= generator.samples:
            break
    return np.array(data), np.array(labels)

X_train, y_train = flatten_data(trainGenerator, batch_size=32)
X_test, y_test = flatten_data(testGenerator, batch_size=32)

# LightGBM Model
lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train, y_train)
lgbm_predictions = lgbm_model.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, lgbm_predictions)
print(f"LightGBM Accuracy: {lgbm_accuracy:.2f}")

# Save LightGBM model
with open("lgbm_model.pkl", "wb") as f:
    pickle.dump(lgbm_model, f)

# XGBoost Model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")

# Save XGBoost model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# Evaluate and print final CNN Model Accuracy on Test Set
cnn_accuracy = cnn_model.evaluate(testGenerator)[1]
print(f"CNN Accuracy: {cnn_accuracy:.2f}")