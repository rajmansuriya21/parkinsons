import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import shap
import lime
import lime.lime_tabular
from preprocess import preprocess

@tf.keras.utils.register_keras_serializable()
def swish(x):
    return tf.nn.swish(x)

@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha=0.5, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1 - 1e-7)
        loss_1 = -y_true * alpha * tf.keras.backend.pow((1 - y_pred), gamma) * tf.keras.backend.log(y_pred)
        loss_0 = -(1 - y_true) * (1 - alpha) * tf.keras.backend.pow(y_pred, gamma) * tf.keras.backend.log(1 - y_pred)
        return tf.keras.backend.mean(loss_1 + loss_0)
    return loss

def grad_cam(model, img, layer_name):
    """Generate Grad-CAM heatmap for MRI."""
    img = img.reshape(1, 128, 128, 1)
    grad_model = Model(inputs=[model.inputs[0]], outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img])
        loss = predictions[:, 1]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap + 1e-10)
    return heatmap

def evaluate():
    """Evaluate the multimodal model."""
    (X_train_mri, X_train_non_mri), (X_test_mri, X_test_non_mri), y_train, y_test, _ = preprocess()
    
    # Reshape
    feature_dim = X_test_non_mri.shape[1] // 2
    X_train_mri = X_train_mri.reshape(-1, 128, 128, 1)
    X_test_mri = X_test_mri.reshape(-1, 128, 128, 1)
    X_train_voice = X_train_non_mri[:, :feature_dim].reshape(-1, feature_dim, 1)
    X_train_gait = X_train_non_mri[:, feature_dim:].reshape(-1, feature_dim, 1)
    X_test_voice = X_test_non_mri[:, :feature_dim].reshape(-1, feature_dim, 1)
    X_test_gait = X_test_non_mri[:, feature_dim:].reshape(-1, feature_dim, 1)
    
    # Load model
    model = load_model(
        "parkinsons_multimodal_model_final_v7.keras",
        custom_objects={"swish": swish, "focal_loss": focal_loss(alpha=0.5, gamma=2.0)}
    )
    
    # Predict
    y_pred_proba = model.predict([X_test_mri, X_test_voice, X_test_gait])
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No PD", "PD"], yticklabels=["No PD", "PD"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.show()
    
    # Load history
    with open("training_history.pkl", "rb") as f:
        history = pickle.load(f)
    
    # Plot training/validation
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Test Accuracy: {accuracy:.2f}')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()
    
    # SHAP for non-MRI
    def predict_fn(x):
        x_voice = x[:, :feature_dim].reshape(-1, feature_dim, 1)
        x_gait = x[:, feature_dim:].reshape(-1, feature_dim, 1)
        x_mri = np.repeat(X_test_mri[:1], len(x), axis=0)
        return model.predict([x_mri, x_voice, x_gait])
    
    explainer = shap.KernelExplainer(predict_fn, X_train_non_mri[:50])
    shap_values = explainer.shap_values(X_test_non_mri[:10])
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_non_mri[:10], feature_names=[f"Feature_{i}" for i in range(X_test_non_mri.shape[1])])
    plt.savefig("shap_summary.png")
    plt.show()
    
    # LIME for non-MRI
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_non_mri,
        feature_names=[f"Feature_{i}" for i in range(X_train_non_mri.shape[1])],
        class_names=["No PD", "PD"],
        mode="classification"
    )
    for i in range(3):
        exp = lime_explainer.explain_instance(X_test_non_mri[i], predict_fn, num_features=10)
        exp.save_to_file(f"lime_explanation_{i}.html")
        print(f"LIME explanation {i} saved.")
    
    # Grad-CAM for MRI
    for i in range(3):
        heatmap = grad_cam(model, X_test_mri[i], "conv2d_2")  # Adjust layer name if needed
        plt.figure(figsize=(6, 6))
        plt.imshow(X_test_mri[i].reshape(128, 128), cmap='gray')
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title(f"Grad-CAM for Test Image {i}")
        plt.savefig(f"grad_cam_{i}.png")
        plt.show()

if __name__ == "__main__":
    evaluate()