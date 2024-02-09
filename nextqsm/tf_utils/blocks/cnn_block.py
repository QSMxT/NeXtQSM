import tensorflow as tf


class CNN(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides=1, padding="SAME", output_padding=None, data_format=None, dilation_rate=1, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, batch_norm=False, activation=None, dropout=0., up=False):
        super(CNN, self).__init__()

        self.parts = []
        if not up:
            self.parts.append(tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
                                                     dilation_rate=dilation_rate, activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                                     bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                                     activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint))
        else:
            self.parts.append(tf.keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, output_padding=output_padding,
                                                              data_format=data_format, dilation_rate=dilation_rate, activation=None, use_bias=use_bias,
                                                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                                              bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                                                              bias_constraint=bias_constraint))

        self.parts.append(tf.keras.layers.BatchNormalization()) if batch_norm else None
        self.parts.append(activation()) if activation is not None else None
        self.parts.append(tf.keras.layers.Dropout(dropout)) if dropout > 0. else None

    def call(self, x, training):
        for layer in self.parts:
            x = layer(x, training=training)
        return x
