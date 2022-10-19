import tensorflow as tf

from tf_utils import MetricsManager
from processing import qsm


class Solver:

    def __init__(self, params):
        self.params = params
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])
        self.loss_obj = tf.keras.losses.MeanAbsoluteError()
        self.loss_manager = MetricsManager()

        self.operator = qsm.QSM()

    @tf.function
    def test_step(self, bf_model, model, x, source, freq, chi, mask, kernel, weight_vn):
        metrics = {"E_D": [], "L": []}

        with tf.GradientTape(persistent=True) as tape:
            # Background field
            bf_logits = bf_model(source, training=False) * mask
            metrics["bf_loss"] = self.loss_obj(freq, bf_logits)
            metrics["bf_rmse"] = self.loss_manager.metrics["NRMSE"](freq, bf_logits, mask)

            # Dipole inversion
            x.assign(bf_logits)
            for step in range(self.params["vn_n_steps"]):
                E_D = self.loss_manager.metrics[self.params["vn_dt_loss"]](bf_logits, self.operator.forward_operation_fourier(x, kernel), mask)
                metrics["E_D"].append(E_D)

                E_R = tf.reduce_mean(tf.math.abs(model(x, training=False)))

                E_i = model.lambdas[step] * E_D + E_R
                metrics["L"].append(model.lambdas[step])

                dE_dx = tape.gradient(E_i, x)
                x = x - dE_dx
                x = x * mask
        
        metrics["vn_loss"] = self.loss_obj(chi, x)
        metrics["vn_rmse"] = self.loss_manager.metrics["NRMSE"](chi, x, mask)
        return bf_logits, x, metrics
