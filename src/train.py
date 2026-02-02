# ===============================
# Training & Fine-Tuning
# ===============================

import numpy as np
import tensorflow as tf # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from config import *

def get_class_weights(train_gen):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    return dict(enumerate(weights))


def train_model(model, train_gen, val_gen):

    class_weights = get_class_weights(train_gen)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_CRITERIA,
        restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stop]
    )

    # Fine-tuning
    model.layers[1].trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNING_EPOCHS,
        class_weight=class_weights
    )

    return history, history_fine
