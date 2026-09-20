import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Feature Extraction (32-D)
def extract_best_features(file_path, segment_duration=5.0, hop_duration=1.0):
    try:
        audio, sr = librosa.load(file_path)
        total_duration = librosa.get_duration(y=audio, sr=sr)
        max_energy = -np.inf
        best_features = None

        for start in np.arange(0, total_duration - segment_duration + 1, hop_duration):
            end = start + segment_duration
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = audio[start_sample:end_sample]

            if len(segment) < int(segment_duration * sr):
                continue

            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)

            mfcc_mean = np.mean(mfcc.T, axis=0)
            chroma_mean = np.mean(chroma.T, axis=0)
            contrast_mean = np.mean(contrast.T, axis=0)

            combined = np.hstack([mfcc_mean, chroma_mean, contrast_mean])
            energy = np.sum(mfcc_mean ** 2)

            if energy > max_energy:
                max_energy = energy
                best_features = combined

        return best_features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load Dataset from Folders
def load_features_from_folders(base_path=r'C:/Users/user/Desktop/animal_sounds'):
    features = []
    labels = []
    class_folders = ['monkeys', 'squirrels', 'no_wilds']

    for label in class_folders:
        folder_path = os.path.join(base_path, label)
        if not os.path.exists(folder_path):
            print(f" Folder not found: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                feat = extract_best_features(file_path)
                if feat is not None:
                    features.append(feat)
                    labels.append(label)

    return np.array(features), np.array(labels)

# Load and Prepare the Data
features, labels = load_features_from_folders()
print(f"Loaded {len(features)} samples with {features.shape[1]} features each.")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y_categorical = to_categorical(y_encoded)
joblib.dump(le, r"C:/Users/user/Desktop/hardware/label_encoder.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Define and Train Neural Network
model = Sequential([
Dense(128, activation='relu', input_shape=(features.shape[1],)),
Dense(64, activation='relu'),
Dense(3, activation='softmax')  # 3 classes
])
#model = Sequential([
 #   Dense(128, activation='relu', input_shape=(features.shape[1],)),
  #  Dropout(0.3),
   # Dense(64, activation='relu'),
    #Dropout(0.2),
    #Dense(3, activation='softmax')
#])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

plt.figure(figsize=(6, 5))

# Evaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"\n Neural Network Accuracy: {acc * 100:.2f}%")

# Confusion Matrix
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[early_stop])
os.makedirs(r"C:/Users/user/Desktop/hardware", exist_ok=True)
model.save(r"C:/Users/user/Desktop/hardware/animal_detector.h5")
print(" Model saved as animal_detector.h5", flush=True)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# Plot Training Curves

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
