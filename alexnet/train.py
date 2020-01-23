import tensorflow as tf
import os
import pandas as pd

layer = tf.keras.layers
seq = tf.keras.models.Sequential


def build_model():
    model = seq([
        layer.Conv2D(96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
        layer.MaxPooling2D(pool_size=(3, 3), strides=(2,2)),

        layer.Conv2D(256, kernel_size=(5,5), strides=(1,1), activation='relu'),
        layer.MaxPooling2D(pool_size=(3,3), strides=(2,2)),

        layer.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu'),

        layer.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu'),

        layer.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        layer.MaxPooling2D(pool_size=(3,3), strides=(2,2)),

        layer.Flatten(),
        layer.Dense(4096, input_shape=(224*224*3,), activation='relu'),
        layer.Dropout(0.4),

        layer.Dense(4096, activation='relu'),
        layer.Dense(4096, activation='relu'),
        layer.Dropout(0.4),

        layer.Dense(1, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def generate_data():
    filenames = os.listdir("data")
    classes = []
    for filename in filenames:
        type = filename.split('.')[0]
        if type == "cat":
            classes.append("0")
        else:
            classes.append("1")

    return pd.DataFrame({
        "file": filenames,
        "class": classes
    })
