import tensorflow as tf
import numpy as np


class QSM:

    def __init__(self, voxel_size=(1, 1, 1), b_vec=(0, 0, 1)):
        self.voxel_size = voxel_size
        self.b_vec = b_vec

    def get_dipole_kernel_fourier(self, shape):
        assert (len(shape) == 3 and shape[0] % 2 == 0 and shape[1] % 2 == 0 and shape[2] % 2 == 0)
        ry, rx, rz = np.meshgrid(np.arange(-shape[1] / 2, shape[1] / 2.),
                                 np.arange(-shape[0] / 2., shape[0] / 2.),
                                 np.arange(-shape[2] / 2., shape[2] / 2.))

        rx = (rx / np.max(np.abs(rx))) / (2*self.voxel_size[0])
        ry = (ry / np.max(np.abs(ry))) / (2*self.voxel_size[1])
        rz = (rz / np.max(np.abs(rz))) / (2*self.voxel_size[2])

        r2 = np.power(rx, 2) + np.power(ry, 2) + np.power(rz, 2)

        kernel = 1. / float(3.) - np.divide(np.power(rx * self.b_vec[0] + ry * self.b_vec[1] + rz * self.b_vec[2], 2), r2 + np.finfo(np.float).eps)

        return np.float32(kernel)


    def forward_operation_fourier(self, y, kernel):
        assert (len(y.shape) == 5 and len(kernel.shape) == 3)
        y = y[:, :, :, :, 0]
        scaling = tf.dtypes.cast(tf.math.sqrt(tf.dtypes.cast(tf.size(y), tf.float32)), tf.complex64)
        y_fft = tf.signal.fft3d(tf.dtypes.cast(y, tf.complex64)) / scaling
        phase = (y_fft * tf.dtypes.cast(tf.signal.fftshift(kernel), tf.complex64))
        phase = tf.math.real(tf.signal.ifft3d(phase) * scaling)

        return tf.expand_dims(phase, axis=-1)


def get_kernel_features(folder):
    b_vec = (0, 0, 1)
    voxel_size = (1., 1., 1.)
    return b_vec, voxel_size
