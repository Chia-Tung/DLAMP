from importlib import import_module


def get_builder(model_name: str):
    return getattr(import_module("..", __name__), "{}Builder".format(model_name))
