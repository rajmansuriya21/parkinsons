import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler  # Changed to StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from scipy.stats import skew, kurtosis

def load_mri_data(mri_path="/home/wot-raj/parkinsons_multimodal/data/MRI_data"):
    mri_data = []
    labels = []
    preferred_types = ['t1_', 't2_', 'FLAIR']
    normal_path = os.path.join(mri_path, "normal")
    pd_path = os.path.join(mri_path, "parkinson")
    
    for file in os.listdir(normal_path):
        if file.endswith(".png") and any(t in file for t in preferred_types):
            img = cv2.imread(os.path.join(normal_path, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64)) / 255.0  # Reduced size to prevent overfitting
                mri_data.append(img)
                labels.append(0)
    
    for file in os.listdir(pd_path):
        if file.endswith(".png") and any(t in file for t in preferred_types):
            img = cv2.imread(os.path.join(pd_path, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64)) / 255.0
                mri_data.append(img)
                labels.append(1)
    
    mri_data = np.array(mri_data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)  # Changed to int32 for clarity
    print(f"Loaded {len(mri_data)} MRI images (Normal: {np.sum(labels==0)}, PD: {np.sum(labels==1)})")
    return mri_data, labels

def load_other_datasets():
    datasets = {}
    paths = {
        "telemonitoring": "/home/wot-raj/parkinsons_multimodal/data/parkinsons_telemonitoring.csv",
        "uci": "/home/wot-raj/parkinsons_multimodal/data/parkinsons.csv",
        "speech_train": "/home/wot-raj/parkinsons_multimodal/data/parkinsons_speech_train.csv",
        "speech_test": "/home/wot-raj/parkinsons_multimodal/data/parkinsons_speech_test.csv",
        "gait": "/home/wot-raj/parkinsons_multimodal/data/combined_gait_data.csv"
    }
    for name, path in paths.items():
        if os.path.exists(path):
            try:
                datasets[name] = pd.read_csv(path, low_memory=False)
                datasets[name] = datasets[name].apply(pd.to_numeric, errors='coerce')
                print(f"{name.capitalize()} Data Loaded Successfully!")
            except Exception as e:
                print(f"Error loading {name}:", e)
                datasets[name] = pd.DataFrame()
        else:
            print(f"{name.capitalize()} Data File Not Found!")
            datasets[name] = pd.DataFrame()
    return datasets

def preprocess_voice_data(telemonitoring_data, uci_data):
    if telemonitoring_data.empty or uci_data.empty:
        return np.empty((0, 10)), np.array([])
    voice_features = pd.concat([
        telemonitoring_data.drop(columns=["motor_UPDRS", "total_UPDRS", "subject#", "age", "sex", "test_time"], errors='ignore'),
        uci_data.drop(columns=["name", "status"], errors='ignore')
    ], axis=1)
    voice_features.fillna(voice_features.mean(), inplace=True)
    labels = uci_data["status"].values.astype(np.int32)
    return voice_features.values, labels

def preprocess_speech_data(speech_train, speech_test):
    if speech_train.empty and speech_test.empty:
        return np.empty((0, 4))
    speech_data = pd.concat([speech_train, speech_test], axis=0)
    speech_data = speech_data.select_dtypes(include=[np.number])
    speech_data.fillna(speech_data.mean(), inplace=True)
    features = np.array([[np.mean(row.dropna()), np.std(row.dropna()), skew(row.dropna()), kurtosis(row.dropna())]
                        for _, row in speech_data.iterrows()], dtype=np.float32)
    return features

def preprocess_gait_data(gait_data):
    if gait_data.empty:
        return np.empty((0, 4))
    gait_data = gait_data.select_dtypes(include=[np.number])
    gait_data.fillna(gait_data.mean(), inplace=True)
    features = np.array([[np.mean(row.dropna()), np.std(row.dropna()), skew(row.dropna()), kurtosis(row.dropna())]
                        for _, row in gait_data.iterrows()], dtype=np.float32)
    return features

def augment_mri_data(mri_data, labels):
    augmented_data = []
    augmented_labels = []
    for img, label in zip(mri_data, labels):
        augmented_data.append(img)
        augmented_labels.append(label)
        # Add random rotation (up to 15 degrees)
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        augmented_data.append(rotated)
        augmented_labels.append(label)
        # Add random flip
        if np.random.rand() > 0.5:
            flipped = np.flip(img, axis=1)
            augmented_data.append(flipped)
            augmented_labels.append(label)
    return np.array(augmented_data), np.array(augmented_labels)

def augment_non_mri_data(features, labels):
    noise = np.random.normal(0, 0.01, features.shape)  # Increased noise for diversity
    augmented = np.clip(features + noise, -3, 3)  # Clip to prevent outliers
    return np.vstack([features, augmented]), np.concatenate([labels, labels])

def preprocess():
    # Load data
    mri_data, mri_labels = load_mri_data()
    datasets = load_other_datasets()
    telemonitoring_data = datasets["telemonitoring"]
    uci_data = datasets["uci"]
    speech_train = datasets["speech_train"]
    speech_test = datasets["speech_test"]
    gait_data = datasets["gait"]
    
    # Use UCI data for consistent labels
    num_samples = min(uci_data.shape[0], 195)
    labels = uci_data["status"].values[:num_samples].astype(np.int32)
    
    # Split data before augmentation to prevent leakage
    if len(mri_data) > 0:
        indices = np.random.permutation(len(mri_data))[:num_samples]
        mri_data = mri_data[indices]
        mri_labels = mri_labels[indices]
        # Split MRI data
        mri_train, mri_test, mri_labels_train, mri_labels_test = train_test_split(
            mri_data, mri_labels, test_size=0.25, random_state=42, stratify=mri_labels
        )
        # Augment training data only
        mri_train, mri_labels_train = augment_mri_data(mri_train, mri_labels_train)
    else:
        mri_train = np.zeros((num_samples * 2, 64, 64), dtype=np.float32)
        mri_test = np.zeros((num_samples // 2, 64, 64), dtype=np.float32)
        mri_labels_train = np.concatenate([labels, labels])
        mri_labels_test = labels[:num_samples // 2]
    
    # Process voice data
    voice_features, voice_labels = preprocess_voice_data(
        telemonitoring_data.iloc[:num_samples], uci_data.iloc[:num_samples]
    )
    if voice_features.size > 0:
        voice_train, voice_test, voice_labels_train, voice_labels_test = train_test_split(
            voice_features, voice_labels, test_size=0.25, random_state=42, stratify=voice_labels
        )
        voice_train, voice_labels_train = augment_non_mri_data(voice_train, voice_labels_train)
    else:
        voice_train = np.zeros((len(labels) * 2, 10), dtype=np.float32)
        voice_test = np.zeros((len(labels) // 2, 10), dtype=np.float32)
        voice_labels_train = np.concatenate([labels, labels])
        voice_labels_test = labels[:len(labels) // 2]
    
    # Process speech data
    speech_features = preprocess_speech_data(speech_train.iloc[:num_samples], speech_test.iloc[:num_samples])
    if speech_features.size > 0:
        speech_train, speech_test = train_test_split(
            speech_features, test_size=0.25, random_state=42
        )
        speech_train, speech_labels_train = augment_non_mri_data(speech_train, labels[:len(speech_train)])
    else:
        speech_train = np.zeros((len(labels) * 2, 4), dtype=np.float32)
        speech_test = np.zeros((len(labels) // 2, 4), dtype=np.float32)
        speech_labels_train = np.concatenate([labels, labels])
    
    # Process gait data
    gait_features = preprocess_gait_data(gait_data.iloc[:num_samples])
    if gait_features.size > 0:
        gait_train, gait_test = train_test_split(
            gait_features, test_size=0.25, random_state=42
        )
        gait_train, gait_labels_train = augment_non_mri_data(gait_train, labels[:len(gait_train)])
    else:
        gait_train = np.zeros((len(labels) * 2, 4), dtype=np.float32)
        gait_test = np.zeros((len(labels) // 2, 4), dtype=np.float32)
        gait_labels_train = np.concatenate([labels, labels])
    
    # Align samples
    min_samples_train = min(len(mri_train), len(voice_train), len(speech_train), len(gait_train))
    mri_train = mri_train[:min_samples_train]
    voice_train = voice_train[:min_samples_train]
    speech_train = speech_train[:min_samples_train]
    gait_train = gait_train[:min_samples_train]
    labels_train = mri_labels_train[:min_samples_train]
    
    min_samples_test = min(len(mri_test), len(voice_test), len(speech_test), len(gait_test))
    mri_test = mri_test[:min_samples_test]
    voice_test = voice_test[:min_samples_test]
    speech_test = speech_test[:min_samples_test]
    gait_test = gait_test[:min_samples_test]
    labels_test = mri_labels_test[:min_samples_test]
    
    # Combine non-MRI features
    non_mri_train = np.hstack([voice_train, speech_train, gait_train])
    non_mri_test = np.hstack([voice_test, speech_test, gait_test])
    
    # Apply PCA if needed
    if non_mri_train.shape[1] > 10:  # Reduced components
        pca = PCA(n_components=10)
        non_mri_train = pca.fit_transform(non_mri_train)
        non_mri_test = pca.transform(non_mri_test)
    
    # Normalize non-MRI features
    non_mri_scaler = StandardScaler()
    non_mri_train = non_mri_scaler.fit_transform(non_mri_train)
    non_mri_test = non_mri_scaler.transform(non_mri_test)
    
    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_mri_flat = mri_train.reshape(mri_train.shape[0], -1)
    X_train_combined = np.hstack([X_train_mri_flat, non_mri_train])
    X_train_combined, y_train = smote.fit_resample(X_train_combined, labels_train)
    X_train_mri = X_train_combined[:, :64*64].reshape(-1, 64, 64, 1)
    X_train_non_mri = X_train_combined[:, 64*64:].reshape(-1, 10)
    
    # Reshape test data
    X_test_mri = mri_test.reshape(-1, 64, 64, 1)
    X_test_non_mri = non_mri_test
    y_test = labels_test.astype(np.int32)
    
    # Debug: Check data ranges and distributions
    print("MRI Train Range (min, max):", X_train_mri.min(), X_train_mri.max())
    print("Non-MRI Train Range (min, max):", X_train_non_mri.min(), X_train_non_mri.max())
    print("MRI Test Range (min, max):", X_test_mri.min(), X_test_mri.max())
    print("Non-MRI Test Range (min, max):", X_test_non_mri.min(), X_test_non_mri.max())
    print("Train Class Distribution:", np.bincount(y_train))
    print("Test Class Distribution:", np.bincount(y_test))
    
    # Verify no NaNs
    assert not np.any(np.isnan(X_train_mri)), "NaNs in X_train_mri"
    assert not np.any(np.isnan(X_train_non_mri)), "NaNs in X_train_non_mri"
    assert not np.any(np.isnan(X_test_mri)), "NaNs in X_test_mri"
    assert not np.any(np.isnan(X_test_non_mri)), "NaNs in X_test_non_mri"
    assert not np.any(np.isnan(y_train)), "NaNs in y_train"
    assert not np.any(np.isnan(y_test)), "NaNs in y_test"
    
    # Compute class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] * 0.5 for i in range(len(class_weights))}  # Scale down weights
    
    print("Final Processed Data Shapes:")
    print("Train MRI:", X_train_mri.shape, "Train Non-MRI:", X_train_non_mri.shape, "Train Labels:", y_train.shape)
    print("Test MRI:", X_test_mri.shape, "Test Non-MRI:", X_test_non_mri.shape, "Test Labels:", y_test.shape)
    print("Class Weights:", class_weight_dict)
    
    return (X_train_mri, X_train_non_mri), (X_test_mri, X_test_non_mri), y_train, y_test, class_weight_dict

if __name__ == "__main__":
    print("Starting preprocessing pipeline...")
    preprocess()