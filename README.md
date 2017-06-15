# about-tensorflow-batchnorm
the usage of tensorflow batchnorm
1. pay attention to the 'training' phase or 'test' phase of batch_norm of tensorflow version
2. pay attention to updating the moving averages of mean and var 
if you choose to use the implementation of BN in https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py, 
  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y
note that you add the 'moving_mean' and 'moving_variance' in your saver's variable list, otherwise you may get excellent results at training phase but you get much worse results at test stage when you set mode='test'. This is because you do not save the accumulated mean and var  but inference needs them.
What's more, to train a model with batchnorm, you code should like this:
