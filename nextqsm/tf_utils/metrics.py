import tensorflow as tf


class MetricsManager:

    def __init__(self):
        self.metrics = {
            "NRMSE": self.NRMSE,
            "RMSE": self.RMSE
        }

    def NRMSE(self, labels, logits, mask):
        labels = labels * mask if mask is not None else labels
        logits = logits * mask if mask is not None else logits

        mask = tf.ones_like(labels) if mask is None else mask

        true_flat = tf.keras.layers.Flatten()(labels)
        fake_flat = tf.keras.layers.Flatten()(logits)
        mask_flat = tf.keras.layers.Flatten()(mask)

        # Get only elements in mask
        true_new = tf.boolean_mask(true_flat, mask_flat)
        fake_new = tf.boolean_mask(fake_flat, mask_flat)

        # Demean
        true_demean = true_new - tf.math.reduce_mean(true_new)
        fake_demean = fake_new - tf.math.reduce_mean(fake_new)

        return 100 * tf.norm(true_demean - fake_demean) / tf.norm(true_demean)

    def RMSE(self, labels, logits, mask):
        return 100 * tf.norm(labels - logits) / tf.norm(labels)
