import json
import os

import run_population as krp
from examples.itrainer import ITrainer


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

x_train, y_train, x_test, y_test = krp.x_train, krp.y_train, krp.x_test, krp.y_test


class Trainer(ITrainer):
    @staticmethod
    def save_model(model, path_dir="InitialModel"):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        model.save_weights(os.path.join(path_dir, "model_weights.h5"), overwrite=True)
        with open(os.path.join(path_dir, "model.config"), "w") as f:
            json.dump(model.to_json(), f)

    def create_model(self):
        pass

    def train_func(self, model):
        batch_size = 100
        epochs = 5

        model.compile(loss=self.keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
        early_stop = self.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=1, verbose=0, mode="auto"
        )
        reduce_lr = self.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=0.001)
        results = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[early_stop, reduce_lr],
            verbose=2,
        )

        train_accuracy = results.history["accuracy"][-1]
        test_accuracy = results.history["val_accuracy"][-1]
        return train_accuracy, test_accuracy

    def evaluate_func(self, model):
        batch_size = 200

        model.compile(loss=self.keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
        results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
        return results[-1]
