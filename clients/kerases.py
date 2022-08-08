try:
    import keras
except ImportError:
    keras = None
try:
    from tensorflow import keras as tfkeras
except ImportError:
    tfkeras = None


def get_framework(name):
    if name == "keras":
        if keras is not None:
            return keras
        else:
            print("The project uses a library 'keras' that is not installed.")
            exit(0)
    elif name == "tf.keras":
        if tfkeras is not None:
            return tfkeras
        else:
            print("The project uses a library 'tensorflow' that is not installed.")
            exit(0)
