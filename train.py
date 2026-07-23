import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ==========================
# Dataset Paths
# ==========================
train_path = "data set/train"
test_path = "data set/test"
model_path = "model/emotion_model.h5"

os.makedirs("model", exist_ok=True)

# ==========================
# Data Preprocessing
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=128,
    class_mode="categorical",
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=128,
    class_mode="categorical",
    shuffle=False
)

print("="*40)
print("Classes :", train_data.class_indices)
print("Train Images :", train_data.samples)
print("Test Images :", test_data.samples)
print("="*40)

# ==========================
# CNN Model
# ==========================
model = Sequential()

model.add(Conv2D(32,(3,3),padding="same",activation="relu",input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.30))

model.add(Conv2D(256,(3,3),padding="same",activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.40))

model.add(Flatten())

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(7,activation="softmax"))

# ==========================
# Compile
# ==========================
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Callbacks
# ==========================
checkpoint = ModelCheckpoint(
    model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ==========================
# Train Model
# ==========================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=50,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1
)

# ==========================
# Save Model
# ==========================
model.save(model_path)

print("\n✅ Model Saved Successfully!")
loss, acc = model.evaluate(test_data, verbose=0)
print(f"Validation Accuracy: {acc*100:.2f}%")