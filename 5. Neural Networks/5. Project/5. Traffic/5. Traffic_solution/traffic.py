import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")
    imgs, lbls = load_data(sys.argv[1])

    lbls = tf.keras.utils.to_categorical(lbls)
    xTrain, xTest, yTrain, yTest = train_test_split(
        np.array(imgs), np.array(lbls), testSize=TEST_SIZE
    )

    model = get_model()
    model.fit(xTrain, yTrain, epochs=EPOCHS)
    model.evaluate(xTest,  yTest, verbose=2)

    if len(sys.argv) == 3:
        file = sys.argv[2]
        model.save(file)
        print(f"Model saved: {file}")


def load_data(data_dir):
    imgs = []
    lbls = []

    for i in range(NUM_CATEGORIES):
        url = os.path.join(data_dir, str(i))
        for image in os.listdir(url):
            img = cv2.imread(os.path.join(data_dir, str(i), image))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            imgs.append(img)
            lbls.append(i)
    return imgs, lbls


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()