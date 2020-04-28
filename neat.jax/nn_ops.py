import math
import jax.numpy as np


def relu(value):
    if value > 0:
        return value
    return 0.0

def sigmoid(value):
    return 1 / (1 + math.e ** -value)

def softmax(value):
    pass

