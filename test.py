import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

# load data
# data = np.load("test_mocap_dataset.npz")
data = np.load("mocap_dataset.npz")
X_test = data["X_test"]
y_test = data["y_test"]

# load normalization parameters
norm = np.load("cnn_simple_norm_params.npz")
mean = norm["mean"]
std  = norm["std"]
X_test_norm = (X_test - mean) / std

# load model
cnn = keras.models.load_model("cnn_simple.h5")

loss, accuracy = cnn.evaluate(X_test_norm, y_test, batch_size=8)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# confusion matrix

y_pred_prob = cnn.predict(X_test_norm, batch_size=8)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
