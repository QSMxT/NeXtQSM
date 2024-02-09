import numpy as np
import nibabel as nib
import tensorflow as tf

from nextqsm.tf_utils import misc

def load_testing_volume(paths):
    data, meta = {}, {}

    for label in paths:
        if paths[label] is None:
            data[label] = None
            continue

        volume = nib.load(paths[label])
        data[label] = volume.get_fdata()

        # Add metadata
        if "affine" not in meta:
            meta["affine"] = volume.affine
            meta["header"] = volume.header
            meta["orig_shape"] = volume.shape

            # Check if padding needed
            pad_size, is_same_shape = misc.get_how_much_to_pad(meta["orig_shape"], 64)

        # Convert to float32 and expand_dims
        data[label] = np.float32(np.array([np.expand_dims(data[label], -1)]))
        assert (len(data[label].shape) == 5 and data[label].shape[0] == 1 and data[label].shape[-1] == 1)

        # Add padding if needed
        if not is_same_shape:
            data[label], _, meta["pad_added"] = misc.add_padding(data[label], pad_size)

    # Multiply by mask if not None
    if data["mask"] is not None:
        for label in paths:
            if data[label] is not None and label != "mask":
                data[label] *= data["mask"]
    else:
        data["mask"] = tf.ones_like(data["source"])

    return {paths["source"].split("/")[-1].split(".")[0]: [data]}, meta