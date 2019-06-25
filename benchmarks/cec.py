import os
import sys
import numpy as np

import benchmarks.cec2013.cec13 as f13
import benchmarks.cec2017.cec17 as f17


def func_wrapper(func, func_id):
    def wrapped(x):
        
        origin_shape = x.shape
        dim = origin_shape[-1]
        
        if type(x) is np.ndarray:
            x = x.reshape((-1, dim)).tolist()
        if func == "cec13":
            tmp = f13.eval(x, func_id+1)
        elif func == "cec17":
            tmp = f17.eval(x, func_id+1)
        else:
            raise Exception("No such benchmark!")

        return np.array(tmp).reshape(origin_shape[:-1])

    return wrapped

cec13 = [func_wrapper("cec13", func_id) for func_id in range(28)]
cec17 = [func_wrapper("cec17", func_id) for func_id in range(30)]
