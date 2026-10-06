import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# last inn dataen
data = np.load("mocap_dataset.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]

num_classes = len(np.unique(y_train))

# normalize
mean = X_train.mean(axis=(0, 1), keepdims=True)
std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
X_train_norm = (X_train - mean) / std
X_val_norm   = (X_val   - mean) / std

T = X_train_norm.shape[1]         
num_features = X_train_norm.shape[2]

inputs = keras.Input(shape=(T, num_features))
x = layers.Conv1D(filters=2**4, kernel_size=5, padding="same", activation="relu")(inputs)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer='sgd', 
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        mode="min"
    ),
]

history = model.fit(
    X_train_norm, y_train,
    validation_data=(X_val_norm, y_val),
    epochs=100,
    batch_size=8,
    callbacks=callbacks
)

model.save("cnn_simple.h5")
np.savez("cnn_simple_norm_params.npz", mean=mean, std=std)

def plot_history(history):
    plt.figure(figsize=(10, 4))

    # loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    # accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.show()

plot_history(history)
