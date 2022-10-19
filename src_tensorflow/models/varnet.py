import tensorflow as tf

from tf_utils import UNet, misc


class VarNet(tf.keras.Model):

    def __init__(self, params):
        super(VarNet, self).__init__()
        self.params = params
        self.nets, self.lambdas = self.init_layers()

    def init_layers(self):
        lambdas = []

        nets = UNet(1, self.params["vn_n_layers"], self.params["vn_starting_filters"], 3, self.params["vn_kernel_initializer"], self.params["vn_batch_norm"],
                    0., misc.get_act_function(self.params["vn_act_func"]), 1, False, False, None)

        for n_step in range(self.params["vn_n_steps"]):
            lambdas.append(tf.Variable(self.params["vn_l_init"], name='L-' + str(n_step), trainable=True, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0., 500)))

        return nets, lambdas

    def call(self, x, training):
        return self.nets(x, training=training)
