import tensorflow as tf

from tf_utils.blocks.cnn_block import CNN


class UNet(tf.keras.Model):
    def __init__(self, n_classes, n_layers, starting_filters, k_size, init, batch_norm, dropout, activation, conv_per_layer, max_pool,
                 upsampling, kernel_regularizer, max_n_filters=512):
        super(UNet, self).__init__()

        self.n_layers = n_layers
        self.conv_per_layer = conv_per_layer
        self.max_pool = max_pool
        self.upsampling = upsampling

        if kernel_regularizer is not None:
            if kernel_regularizer[0] == "L1":
                kernel_regularizer = tf.keras.regularizers.l1(kernel_regularizer[1])
            elif kernel_regularizer[0] == "L2":
                kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer[1])
            else:
                raise NotImplementedError(kernel_regularizer)

        self.encoder = []
        for i in range(n_layers):
            # Set maximum number of filters
            n_filters = max_n_filters if starting_filters * (2 ** i) > max_n_filters else starting_filters * (2 ** i)

            # First layer does not have Batch Norm
            is_batch_norm = i != 0 and batch_norm

            if max_pool and i != 0:
                self.encoder.append(tf.keras.layers.MaxPool3D())

            # How many CNN layers at each stage
            for j in range(conv_per_layer):
                strides = 2 if j == 0 and i != 0 and max_pool is False else 1
                self.encoder.append(CNN(n_filters, k_size, strides=strides, kernel_initializer=init, batch_norm=is_batch_norm, dropout=False,
                                        activation=activation, kernel_regularizer=kernel_regularizer))

        self.decoder = []
        for i in range(n_layers - 2, -1, -1):
            n_filters = max_n_filters if starting_filters * (2 ** i) > max_n_filters else starting_filters * (2 ** i)

            if upsampling:
                self.decoder.append(tf.keras.layers.UpSampling3D())

            for j in range(conv_per_layer):
                strides = 2 if j == 0 and upsampling is False else 1
                self.decoder.append(CNN(n_filters, k_size, strides=strides, kernel_initializer=init, batch_norm=batch_norm, dropout=dropout, activation=activation,
                                        kernel_regularizer=kernel_regularizer, up=True))

        self.last_conv = CNN(n_classes, 3, kernel_initializer=init, batch_norm=None, dropout=0., activation=None)

    def call(self, x, training):

        """
            Encoder
        """
        skips = []
        for i in range(self.n_layers):

            if self.max_pool and i != 0:
                x = self.encoder[i * self.conv_per_layer + (i - 1)](x, training=training)

            for j in range(self.conv_per_layer):
                x = self.encoder[i * (self.conv_per_layer + int(self.max_pool)) + j](x, training=training)
            skips.append(x)

        """
            Decoder
        """
        skips = list(reversed(skips[:-1]))
        for i in range(len(range(self.n_layers - 2, -1, -1))):

            if self.upsampling:
                x = self.decoder[i * (self.conv_per_layer + int(self.upsampling))](x, training=training)
                x = tf.keras.layers.Concatenate()([x, skips[i]])

            for j in range(self.conv_per_layer):
                x = self.decoder[i * (self.conv_per_layer + int(self.upsampling)) + j + int(self.upsampling)](x, training=training)

                if j == 0 and self.upsampling is False:
                    x = tf.keras.layers.Concatenate()([x, skips[i]])

        return self.last_conv(x)

    def summary(self, input_shape):
        """
        :param input_shape: (32, 32, 1)
        """
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))
        model.summary(line_length=130)

