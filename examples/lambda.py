import json
import os

if __name__ == "__main__":
    import sys

    sys.path.append("/".join(sys.path[0].split("/")[:-1]))

from examples.itrainer import ITrainer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def mul(x):
    return x * x


class Trainer(ITrainer):
    def dataload(self):
        self.batch_size = 128
        self.num_classes = 10
        self.evaluate_batch_size = 500
        self.evaluate_num_classes = 10
        self.epochs = 5

        self.img_rows, self.img_cols = 28, 28
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.keras.datasets.mnist.load_data()

        # create dataset
        x_train, y_train, x_test, y_test = (
            self.x_train[:1000],
            self.y_train[:1000],
            self.x_test[:1000],
            self.y_test[:1000],
        )
        if self.keras.backend.image_data_format() == "channels_first":
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        self.create_dataset = x_train, y_train, x_test, y_test, input_shape

        # train dataset
        x_train, y_train, x_test, y_test = self.x_train[:500], self.y_train[:500], self.x_test[:100], self.y_test[:100]

        if self.keras.backend.image_data_format() == "channels_first":
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        self.train_dataset = x_train, y_train, x_test, y_test

        # evaluate dataset
        x_test, y_test = self.x_test[150:250], self.y_test[150:250]

        if self.keras.backend.image_data_format() == "channels_first":
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
        else:
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)

        x_test = x_test.astype("float32")
        self.evaluate_dataset = x_test, y_test

    @staticmethod
    def save_model(model, path_dir="InitialModel"):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        model.save_weights(os.path.join(path_dir, "model_weights.h5"), overwrite=True)
        with open(os.path.join(path_dir, "model.config"), "w") as f:
            json.dump(model.to_json(), f)

    def create_model(self):
        x_train, y_train, x_test, y_test, input_shape = self.create_dataset

        print(f"{x_train.shape[0]} train samples")
        print(f"{x_test.shape[0]} test samples")

        y_train = self.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = self.keras.utils.to_categorical(y_test, self.num_classes)

        input_layer = self.keras.layers.Input(shape=input_shape)
        x = self.keras.layers.Lambda(lambda _x: _x / 255)(input_layer)
        x = self.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
        x = self.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        x = self.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = self.keras.layers.Lambda(mul)(x)
        x = self.keras.layers.Dropout(0.25)(x)
        x = self.keras.layers.Flatten()(x)
        x = self.keras.layers.Dense(128, activation="relu")(x)
        x = self.keras.layers.Dropout(0.5)(x)
        output_layer = self.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        model = self.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss=self.keras.losses.categorical_crossentropy,
            optimizer=self.keras.optimizers.Adadelta(),
            metrics=["accuracy"],
        )
        results = model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(x_test, y_test),
        )
        self.save_model(model)

        train_accuracy = results.history["accuracy"][-1]
        validation_accuracy = results.history["val_accuracy"][-1]

        return train_accuracy, validation_accuracy

    def train_func(self, model):
        x_train, y_train, x_test, y_test = self.train_dataset

        print(f"x_train shape: {x_train.shape}")
        print(f"{x_train.shape[0]} train samples")
        print(f"{x_test.shape[0]} test samples")

        y_train = self.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = self.keras.utils.to_categorical(y_test, self.num_classes)

        model.compile(loss=self.keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
        early_stop = self.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=3,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        reduce_lr = self.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=1, min_lr=0.001, verbose=1
        )
        results = model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[early_stop, reduce_lr],
        )

        train_accuracy = results.history["accuracy"][-1]
        test_accuracy = results.history["val_accuracy"][-1]
        return train_accuracy, test_accuracy

    def evaluate_func(self, model):
        x_test, y_test = self.evaluate_dataset

        print(f"{x_test.shape[0]} test samples")

        y_test = self.keras.utils.to_categorical(y_test, self.evaluate_num_classes)
        model.compile(loss=self.keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
        results = model.evaluate(x_test, y_test, batch_size=self.evaluate_batch_size)
        return results[-1]


if __name__ == "__main__":
    from clients.kerases import get_framework

    # Use what you need
    # keras = get_framework("keras")
    keras = get_framework("tf.keras")

    trainer = Trainer(keras=keras)
    trainer.create_model()
    print("Model successfully created")
