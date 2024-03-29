import os

if __name__ == "__main__":
    import sys

    sys.path.append("/".join(sys.path[0].split("/")[:-1]))

from examples.itrainer import ITrainer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Trainer(ITrainer):
    def load_data(self):
        self.batch_size = 128
        self.num_classes = 10
        self.evaluate_batch_size = 500
        self.evaluate_num_classes = 10
        self.epochs = 15
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

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        self.create_dataset = x_train, y_train, x_test, y_test, input_shape

        # train dataset
        x_train, y_train, x_test, y_test = (
            self.x_train[:500],
            self.y_train[:500],
            self.x_test[:100],
            self.y_test[:100],
        )

        if self.keras.backend.image_data_format() == "channels_first":
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        self.train_dataset = x_train, y_train, x_test, y_test

        # evaluate dataset
        x_test, y_test = self.x_test[150:250], self.y_test[150:250]

        if self.keras.backend.image_data_format() == "channels_first":
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
        else:
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)

        x_test = x_test.astype("float32") / 255

        self.evaluate_dataset = x_test, y_test

    def create_model(self):
        epochs = 5
        x_train, y_train, x_test, y_test, input_shape = self.create_dataset

        print(f"{x_train.shape[0]} train samples")
        print(f"{x_test.shape[0]} test samples")

        y_train = self.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = self.keras.utils.to_categorical(y_test, self.num_classes)

        input_0 = self.keras.layers.Input(shape=input_shape)
        x_0 = self.keras.layers.Conv2D(32, kernel_size=(4, 4), activation="relu")(input_0)
        x_0 = self.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_0)

        input_1 = self.keras.layers.Input(shape=input_shape)
        x_1 = self.keras.layers.Conv2D(64, kernel_size=(4, 4), activation="relu")(input_1)
        x_1 = self.keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu")(x_1)
        x_1 = self.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_1)

        x = self.keras.layers.Concatenate()([x_0, x_1])
        x = self.keras.layers.Dropout(0.25)(x)
        x = self.keras.layers.Flatten()(x)
        x = self.keras.layers.Dense(128, activation="relu")(x)
        x = self.keras.layers.Dropout(0.5)(x)
        Output = self.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        model = self.keras.models.Model(inputs=[input_0, input_1], outputs=Output)
        model.compile(
            loss=self.keras.losses.categorical_crossentropy,
            optimizer=self.keras.optimizers.Adadelta(),
            metrics=["accuracy"],
        )
        results = model.fit(
            [x_train, x_train],
            y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=([x_test, x_test], y_test),
        )
        self.save_model(model)

        train_accuracy = results.history["accuracy"][-1]
        validation_accuracy = results.history["val_accuracy"][-1]

        return train_accuracy, validation_accuracy

    def train_func(self, model):
        x_train, y_train, x_test, y_test = self.train_dataset

        print("x_train shape:", x_train.shape)
        print(f"{x_train.shape[0]} train samples")
        print(f"{x_test.shape[0]} test samples")

        y_train = self.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = self.keras.utils.to_categorical(y_test, self.num_classes)

        model.compile(loss=self.keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
        results = model.fit(
            [x_train, x_train],
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=([x_test, x_test], y_test),
        )

        train_accuracy = results.history["accuracy"][-1]
        validation_accuracy = results.history["val_accuracy"][-1]
        return train_accuracy, validation_accuracy

    def evaluate_func(self, model):
        x_test, y_test = self.evaluate_dataset

        print(f"{x_test.shape[0]} test samples")

        y_test = self.keras.utils.to_categorical(y_test, self.evaluate_num_classes)
        model.compile(loss=self.keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
        results = model.evaluate([x_test, x_test], y_test, batch_size=self.evaluate_batch_size)
        test_accuracy = results[-1]
        return test_accuracy


if __name__ == "__main__":
    from clients.kerases import get_framework

    # Use what you need
    # keras = get_framework("keras")
    keras = get_framework("tf.keras")

    trainer = Trainer(keras=keras)
    trainer.create_model()
    print("Model successfully created")
