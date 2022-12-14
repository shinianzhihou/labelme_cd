from . import yolov5_cd

class ObjectCD(object):
    def __init__(
        self,
        model_name,
        model_args,
    ):
        super().__init__()
        model = eval(model_name)(**model_args)

        self.model = model
        self.model_name = model_name
        self.model_args = model_args

    def __call__(self, x):
        output = self.model(x)
        return output

def object_cd(**kwargs):
    return ObjectCD(**kwargs)
