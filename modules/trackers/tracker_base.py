import sklearn
from packaging import version

if version.parse(sklearn.__version__) > version.parse("0.22"):
    pass
else:
    pass

import numpy as np

np.set_printoptions(suppress=True)

_FOR_USER_IMPLEMENTERS = "the sign of unimplemented method"


def for_user_implementers(obj):
    setattr(obj, _FOR_USER_IMPLEMENTERS, "meaningless")
    return obj


class ConstructFirstMethod(object):
    """
        A Template Class of Tracking Method
        That methods need to be implemented by subclass should put @for_user_implementers
    """

    def __init__(self):
        # inspection methods of interfaces which should be implemented
        self.check_callbacks()

    def run(self):
        # ...
        self.preprocess()
        self.tracking()
        self.postprocess()

    @classmethod
    def check_callbacks(cls):
        for p in cls.__base__.__dict__.keys():
            if hasattr(cls.__base__.__dict__.get(p), _FOR_USER_IMPLEMENTERS):
                if p not in cls.__dict__.keys():
                    raise ValueError(f"Hi, interface [{p}] need to implemented ~")

    # @for_user_implementers
    def preprocess(self):
        ...

    # @for_user_implementers
    def tracking(self):
        ...

    # @for_user_implementers
    def postprocess(self):
        ...
