class ITrainer:
    keras = None

    def __init__(self, keras):
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
        self.dataload()

    def dataload(self):
        pass

    @staticmethod
    def save_model(model, path_dir):
        pass

    def create_model(self):
        pass

    def train_func(self, model):
        pass

    def evaluate_func(self, model):
        pass
