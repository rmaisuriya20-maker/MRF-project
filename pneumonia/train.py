"""
Pneumonia X-Ray Pipeline — Stage 2: Training

- Transfer learning with pretrained ResNet50.
- Freeze base layers; add GlobalAveragePooling + Dense(sigmoid).
- Binary Cross Entropy loss, Adam optimizer.
- Metrics: Accuracy, Precision, Recall (recall emphasized for healthcare).
"""

import os
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Local
from preprocess import (
    load_image_paths_and_labels,
    remove_corrupted,
    train_val_test_split,
    load_and_preprocess,
    augment_image,
    IMG_SIZE,
)
from config import (
    DATA_ROOT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    MODEL_SAVE_PATH,
)


def build_model(input_shape=(224, 224, 3), freeze_base: bool = True) -> Model:
    """
    ResNet50 (ImageNet weights), freeze base, GlobalAveragePooling, Dense(1, sigmoid).
    """
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    if freeze_base:
        base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inputs=base.input, outputs=x)
    return model


def tf_dataset_from_paths(
    paths: list[str],
    labels: list[int],
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
):
    """Build tf.data.Dataset from (paths, labels). Preprocess and optionally augment in graph."""
    def load_and_preprocess_tf(path: str, label: int):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.keras.layers.RandomRotation(0.1)(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1024, len(paths)))
    ds = ds.map(
        lambda p, l: (tf.py_function(
            lambda x: load_and_preprocess(x.decode("utf-8")),
            [p],
            tf.float32,
        ), l),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def numpy_dataset(paths: list[str], labels: list[int], augment: bool = False):
    """Load full dataset into memory (for smaller datasets)."""
    X = np.array([load_and_preprocess(p) for p in paths], dtype=np.float32)
    y = np.array(labels, dtype=np.float32).reshape(-1, 1)
    if augment:
        for i in range(len(X)):
            X[i] = augment_image(X[i])
    return X, y


def train(
    data_root: str = None,
    model_save_path: str = None,
    epochs: int = None,
    batch_size: int = None,
):
    """
    Load data, split, build model, train. Saves best model by val_recall.
    """
    data_root = data_root or str(DATA_ROOT)
    model_save_path = model_save_path or MODEL_SAVE_PATH
    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(data_root).mkdir(parents=True, exist_ok=True)
    (Path(data_root) / "NORMAL").mkdir(parents=True, exist_ok=True)
    (Path(data_root) / "PNEUMONIA").mkdir(parents=True, exist_ok=True)

    # 1. Load and clean
    paths, labels = load_image_paths_and_labels(data_root)
    if not paths:
        raise FileNotFoundError(f"No images under {data_root}. Expected NORMAL/ and PNEUMONIA/ subdirs.")
    paths, labels = remove_corrupted(paths, labels)
    print(f"Loaded {len(paths)} images after removing corrupted.")

    # 2. Split
    (train_p, train_l, val_p, val_l, test_p, test_l) = train_val_test_split(
        paths, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    if len(val_p) == 0 and len(train_p) > 0:
        val_p, val_l = train_p[:1], train_l[:1]
    print(f"Train {len(train_p)} / Val {len(val_p)} / Test {len(test_p)}")

    # 3. Datasets (in-memory for simplicity; use tf_dataset_from_paths for large data)
    X_train = np.array([load_and_preprocess(p) for p in train_p], dtype=np.float32)
    y_train = np.array(train_l, dtype=np.float32).reshape(-1, 1)
    X_val = np.array([load_and_preprocess(p) for p in val_p], dtype=np.float32)
    y_val = np.array(val_l, dtype=np.float32).reshape(-1, 1)

    # 4. Model
    model = build_model(input_shape=(*IMG_SIZE, 3), freeze_base=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    # 5. Callbacks: save best by val_recall (emphasize recall)
    model_checkpoint = ModelCheckpoint(
        model_save_path,
        monitor="val_recall",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early = EarlyStopping(monitor="val_recall", patience=3, mode="max", restore_best_weights=True, verbose=1)

    # 6. Train (with augmentation via ImageDataGenerator; data already in [0,1])
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode="nearest",
    )
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[model_checkpoint, early],
        verbose=1,
    )

    print(f"Best model saved to {model_save_path}")
    return model, history


if __name__ == "__main__":
    train()
