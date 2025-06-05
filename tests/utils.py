import random

import numpy as np


def set_seed(frontend, seed):
    random.seed(seed)
    np.random.seed(seed)
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        frontend.torch.set_default_dtype(frontend.torch.float64)
        frontend.torch.manual_seed(seed)
    elif frontend.__name__ == "qiboml.interfaces.keras":
        frontend.keras.backend.set_floatx("float64")
        frontend.tf.keras.backend.set_floatx("float64")
        frontend.keras.utils.set_random_seed(seed)
        frontend.tf.config.experimental.enable_op_determinism()
