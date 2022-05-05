import ray
import matplotlib.pyplot as plt
import numpy as np
import os
from ray.train import Trainer
import time
import tensorflow as tf
import cv2
import PIL
import re


# For Google Colab:
# !pip install ray
# import psutil
# ray._private.utils.get_system_memory = lambda: psutil.virtual_memory().total


class CnnModel:
    """Base CNN - LeNet-5"""
    def __init__(self, *args, **kwargs):
        self.use_ray = kwargs.pop("use_ray", False)
        if self.use_ray:
            self.num_workers = kwargs.pop("num_workers", os.cpu_count())
        self.batch_size = kwargs.pop("batch_size", 64)
        self.epochs = kwargs.pop("epochs", 3)
        self.steps_per_epoch = kwargs.pop("steps_per_epoch", 70)
        self.verbose = kwargs.pop("verbose", 2)
        self.fit_time = None
        self.model = None
        self.image_size = 28
        self.name = "CNN (img_size: %d; %s; batch_size: %d; num_workers: %d)" % (
            self.image_size, "ray_train" if self.use_ray else "no_ray",
            self.batch_size, self.num_workers if self.use_ray
            else 1)

    @staticmethod
    def preprocess(x, y):
        x = tf.cast(x, dtype=tf.float32) / 255
        x = tf.expand_dims(x, axis=3)
        y = tf.cast(y, dtype=tf.int32)
        y = tf.one_hot(y, depth=10)
        return x, y

    def get_dataset(self):
        (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(self.batch_size).map(
            self.preprocess)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_size).map(self.preprocess)
        return train_dataset, test_dataset

    def build_and_compile_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, kernel_size=5, strides=1, activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(16, kernel_size=5, strides=1, activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation="relu"),
            tf.keras.layers.Dense(84, activation="relu"),
            tf.keras.layers.Dense(10)
        ])
        self.model.build(input_shape=(None, 28, 28, 1))
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def ray_train(self):
        def train_distributed():
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            train_dataset, test_dataset = self.get_dataset()
            train_dataset = strategy.experimental_distribute_dataset(train_dataset)
            with strategy.scope():
                self.build_and_compile_model()
            tmp_time = time.time()
            self.model.fit(train_dataset, epochs=self.epochs, validation_data=test_dataset,
                           steps_per_epoch=self.steps_per_epoch, verbose=self.verbose)
            print("\n\nTIME: %f" % (time.time() - tmp_time))
            return self.model

        trainer = ray.train.Trainer(backend="tensorflow", num_workers=self.num_workers)
        config = {"lr": 1e-3, "batch_size": self.batch_size / 2, "epochs": self.epochs,
                  "steps_per_epoch": self.steps_per_epoch}
        trainer.start()
        tmp_time = time.time()
        self.model = trainer.run(train_distributed, config=config)[0]
        self.fit_time = time.time() - tmp_time
        trainer.shutdown()
        return self.model

    def train(self):
        if self.use_ray:
            return self.ray_train()
        tmp_time = time.time()
        self.build_and_compile_model()
        train_dataset, test_dataset = self.get_dataset()
        self.model.fit(train_dataset, epochs=self.epochs, validation_data=test_dataset,
                       steps_per_epoch=self.steps_per_epoch)
        self.fit_time = time.time() - tmp_time

    def prepare_image(self, img_array):
        new_array = cv2.resize(img_array / 255, (self.image_size, self.image_size))
        return new_array.reshape(-1, self.image_size, self.image_size, 1)

    def predict(self, image, invert=True, only_number=False):
        """
        Predict the probabilities of numbers from an image.
        :type image: str | numpy.ndarray (png-file link or image array)
        :type invert: bool
        :type only_number: bool
        @rtype numpy.ndarray | None (number probabilities)
        """
        if isinstance(image, str) and re.search(r"\.(png|jpg)$", image):
            try:
                image = PIL.Image.open(image)
            except FileNotFoundError:
                raise ValueError("Invalid image file link")
            else:
                image = image.resize((self.image_size, self.image_size), PIL.Image.ANTIALIAS)
                image = image.convert('L')
                if invert:
                    image = PIL.ImageOps.invert(image)
                image = np.asarray(image)
        if isinstance(image, np.ndarray):
            res = self.model.predict(self.prepare_image(image))
        else:
            raise ValueError("Invalid image parameter type")
        if only_number:
            return list(res[0]).index(max(res[0]))
        return res


class CnnModel224(CnnModel):
    def __init__(self, *args, **kwargs):
        super(CnnModel224, self).__init__(*args, **kwargs)
        self.classes = kwargs.pop("classes", ['cat', 'dog'])
        self.train_path = "CnnModel224_dataset/train"
        self.valid_path = "CnnModel224_dataset/val"
        self.test_path = "CnnModel224_dataset/test"
        self.image_size = 224

    def get_dataset(self):
        train_batches = tf.keras.preprocessing.image_dataset_from_directory(self.train_path,
                                                                            labels="inferred",
                                                                            label_mode="int",
                                                                            color_mode="rgb",
                                                                            batch_size=self.batch_size,
                                                                            image_size=(
                                                                                self.image_size, self.image_size),
                                                                            shuffle=True,
                                                                            seed=123,
                                                                            validation_split=0.1,
                                                                            subset="training")
        valid_batches = tf.keras.preprocessing.image_dataset_from_directory(self.train_path,
                                                                            labels="inferred",
                                                                            label_mode="int",
                                                                            color_mode="rgb",
                                                                            batch_size=self.batch_size,
                                                                            image_size=(
                                                                                self.image_size, self.image_size),
                                                                            shuffle=True,
                                                                            seed=123,
                                                                            validation_split=0.1,
                                                                            subset="validation")
        return train_batches, valid_batches

    def build_and_compile_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                                   input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=2, activation='softmax'),
        ])
        self.model.build(input_shape=(224, 224, 3))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()


def main():
    X = [30, 60, 90, 120]
    total_res = []
    for cnn_model in (CnnModel224, CnnModel):
        res = []
        for batch_size in X:
            _res = []
            model_dist1 = cnn_model(use_ray=1, batch_size=batch_size, epochs=1, num_workers=2, steps_per_epoch=70,
                                    verbose=0)
            model_dist2 = cnn_model(use_ray=1, batch_size=batch_size, epochs=1, num_workers=4, steps_per_epoch=70,
                                    verbose=0)
            model_dist3 = cnn_model(use_ray=1, batch_size=batch_size, epochs=1, num_workers=8, steps_per_epoch=70,
                                    verbose=0)
            model = cnn_model(use_ray=0, batch_size=batch_size, epochs=1, steps_per_epoch=10, verbose=0)
            models = (model_dist1, model_dist2, model_dist3, model)
            for _model in models:
                _model.train()
                _res.append(_model.fit_time)
                print(_model.name, _model.fit_time)
            res.append(_res)
        total_res.append(res)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Train time')
    ax2.set_title("CNN (input: img 224x224)")
    ax1.set_title("CNN (input: img 28x28)")
    for Y_index in range(len(total_res)):
        fig.axes[Y_index].plot(X, [i[3] for i in total_res[Y_index]])
        fig.axes[Y_index].plot(X, [i[0] for i in total_res[Y_index]])
        fig.axes[Y_index].plot(X, [i[1] for i in total_res[Y_index]])
        fig.axes[Y_index].plot(X, [i[2] for i in total_res[Y_index]])
        fig.axes[Y_index].set_xlabel('batch size')
        fig.axes[Y_index].set_ylabel('time')
        fig.axes[Y_index].legend(['cpu_1', 'cpu_2', 'cpu_4', 'cpu_8'])
    plt.show()


if __name__ == "__main__":
    main()
