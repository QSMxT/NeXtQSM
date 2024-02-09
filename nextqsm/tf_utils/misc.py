import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import math


def get_base_path(prefix=""):
    base_path = str(Path(__file__).parent.parent.parent) + "/"
    return base_path


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def add_padding(volumes, pad_size):
    assert (len(volumes.shape) == 5 and len(pad_size) == 3)
    padded_volumes = []
    shape = volumes.shape[1:]
    for volume in volumes:
        # Add one if shape is not EVEN
        padded = np.pad(volume[:, :, :, 0], [(int(shape[0] % 2 != 0), 0), (int(shape[1] % 2 != 0), 0), (int(shape[2] % 2 != 0), 0)], 'constant', constant_values=0.0)

        # Calculate how much padding to give
        val_x = (pad_size[0] - padded.shape[0]) // 2
        val_y = (pad_size[1] - padded.shape[1]) // 2
        val_z = (pad_size[2] - padded.shape[2]) // 2

        # Append padded volume
        padded_volumes.append(np.pad(padded, [(val_x,), (val_y,), (val_z,)], 'constant', constant_values=0.0))

    padded_volumes = np.array(padded_volumes)
    assert (padded_volumes.shape[1] == pad_size[0] and padded_volumes.shape[2] == pad_size[1] and padded_volumes.shape[3] == pad_size[2])

    return np.expand_dims(padded_volumes, -1), np.array(shape[:-1]), np.array([val_x, val_y, val_z])


def remove_padding(volumes, orig_shape, values):
    assert (len(volumes.shape) == 5 and len(orig_shape) == 3 and len(values) == 3)
    # Remove padding
    if values[0] != 0:
        volumes = volumes[:, values[0]:-values[0], :, :]
    if values[1] != 0:
        volumes = volumes[:, :, values[1]:-values[1], :]
    if values[2] != 0:
        volumes = volumes[:, :, :, values[2]:-values[2]]

    volumes = volumes[:, int(orig_shape[0] % 2 != 0):, int(orig_shape[1] % 2 != 0):, int(orig_shape[2] % 2 != 0):]
    assert (volumes.shape[1] == orig_shape[0] and volumes.shape[2] == orig_shape[1] and volumes.shape[3] == orig_shape[2])

    return volumes


def get_how_much_to_pad(shape, multiple):
    pad = []
    is_same_shape = True
    for val in shape:
        pad.append(math.ceil(val/multiple) * multiple)
        if pad[-1] != val:
            is_same_shape = False

    return pad, is_same_shape


def get_act_function(label):
    return tf.keras.layers.ReLU

