import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Conv2D, MaxPooling2D, Flatten, Input, Concatenate, Conv1D, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import pickle
from preprocess import preprocess

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Uncomment for CPU-only

@tf.keras.utils.register_keras_serializable()
def swish(x):
    return tf.nn.swish(x)

@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1 - 1e-7)  # Prevent log(0)
        loss_1 = -y_true * alpha * tf.keras.backend.pow((1 - y_pred), gamma) * tf.keras.backend.log(y_pred)
        loss_0 = -(1 - y_true) * (1 - alpha) * tf.keras.backend.pow(y_pred, gamma) * tf.keras.backend.log(1 - y_pred)
        return tf.keras.backend.mean(loss_1 + loss_0)
    return loss

def cosine_decay(epoch, lr):
    initial_lr = 0.0003
    min_lr = 0.00005
    decay_epochs = 100
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / decay_epochs))
    return min_lr + (initial_lr - min_lr) * cosine_decay

def build_cnn_model(input_shape=(128, 128, 1)):
    """Optimized CNN for MRI data."""
    model = Sequential([
        Input(shape=input_shape),
        GaussianNoise(0.01),  # Add noise for robustness
        Conv2D(16, (3, 3), activation=swish, padding='same', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),  # Increased dropout
        Conv2D(32, (3, 3), activation=swish, padding='same', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(64, activation=swish, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),  # Reduced size
        Dropout(0.4)
    ])
    return model

def build_lstm_model(input_shape):
    """Optimized LSTM for voice/speech data."""
    model = Sequential([
        Input(shape=input_shape),
        GaussianNoise(0.01),  # Add noise for robustness
        LSTM(32, return_sequences=False, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),  # Simplified to one layer
        BatchNormalization(),
        Dropout(0.4),  # Increased dropout
        Dense(32, activation=swish, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))  # Reduced size
    ])
    return model

def build_tcn_model(input_shape):
    """Optimized TCN for gait data."""
    model = Sequential([
        Input(shape=input_shape),
        GaussianNoise(0.01),  # Add noise for robustness
        Conv1D(32, kernel_size=3, padding='causal', activation=swish, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),  # Simplified to one layer
        BatchNormalization(),
        Dropout(0.4),  # Increased dropout
        Flatten(),
        Dense(32, activation=swish, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))  # Reduced size
    ])
    return model

def build_fusion_model(cnn_model, lstm_model, tcn_model):
    """Optimized late fusion of all modalities."""
    combined_input = Concatenate()([cnn_model.output, lstm_model.output, tcn_model.output])
    x = Dense(64, activation=swish, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))(combined_input)  # Reduced size
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Increased dropout
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[cnn_model.input, lstm_model.input, tcn_model.input], outputs=output)
    optimizer = AdamW(learning_rate=0.0003, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss=focal_loss(alpha=0.25, gamma=2.0),  # Adjusted alpha
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])  # Added AUC
    return model

def train():
    """Train multimodal model."""
    (X_train_mri, X_train_non_mri), (X_test_mri, X_test_non_mri), y_train, y_test, class_weight_dict = preprocess()
    
    # Split non-MRI into voice/speech and gait
    feature_dim = X_train_non_mri.shape[1] // 2
    X_train_voice = X_train_non_mri[:, :feature_dim].reshape(-1, feature_dim, 1)
    X_train_gait = X_train_non_mri[:, feature_dim:].reshape(-1, feature_dim, 1)
    X_test_voice = X_test_non_mri[:, :feature_dim].reshape(-1, feature_dim, 1)
    X_test_gait = X_test_non_mri[:, feature_dim:].reshape(-1, feature_dim, 1)
    
    # Reshape MRI
    X_train_mri = X_train_mri.reshape(-1, 128, 128, 1)
    X_test_mri = X_test_mri.reshape(-1, 128, 128, 1)
    
    # Build models
    cnn_model = build_cnn_model()
    lstm_model = build_lstm_model((feature_dim, 1))
    tcn_model = build_tcn_model((feature_dim, 1))
    fusion_model = build_fusion_model(cnn_model, lstm_model, tcn_model)
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(cosine_decay)
    
    # Train
    history = fusion_model.fit(
        [X_train_mri, X_train_voice, X_train_gait], y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy, test_auc = fusion_model.evaluate([X_test_mri, X_test_voice, X_test_gait], y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
    
    # Save model
    fusion_model.save("parkinsons_multimodal_model_final_v8.keras")  # Updated version
    print("Model saved as 'parkinsons_multimodal_model_final_v8.keras'")
    
    # Save history
    with open("training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.show()

if __name__ == "__main__":
    train()