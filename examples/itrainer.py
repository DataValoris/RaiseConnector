import json
import os
from abc import ABC, abstractmethod
from typing import Tuple, Any


class ITrainer(ABC):
    keras = None

    def __init__(self, keras) -> None:
        self.keras = keras
        self.batch_size = None
        self.num_classes = None
        self.evaluate_batch_size = None
        self.evaluate_num_classes = None
        self.img_rows = None
        self.img_cols = None
        self.epochs = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.create_dataset = None
        self.train_dataset = None
        self.evaluate_dataset = None
        self.load_data()

    @abstractmethod
    def load_data(self) -> None:
        pass

    @classmethod
    def save_model(cls, model, path_dir: str = "InitialModel") -> None:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        model.save_weights(os.path.join(path_dir, "model_weights.h5"), overwrite=True)
        with open(os.path.join(path_dir, "model.config"), "w") as f:
            json.dump(model.to_json(), f)

    def create_model(self) -> Any:
        pass

    @abstractmethod
    def train_func(self, model) -> Tuple[float, float]:
        """Trains the model and returns train fitness and validation fitness"""
        pass

    @abstractmethod
    def evaluate_func(self, model) -> float:
        """Evaluates the model and returns test fitness"""
        pass
