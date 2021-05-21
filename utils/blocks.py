import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np

from typing import List, Optional, Sequence, Text, Tuple, Union

class SpectralNormMixin(tf.keras.layers.Layer):
  """Mixin class for SpectralNorm layers."""

  def _create_singular_vectors(
      self,
      kernel: tf.Tensor,
      name: Optional[str] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Builds left and right singular vectors for kernel."""
    shape = kernel.shape.as_list()
    dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    u_name = 'u' if name is None else '{}_u'.format(name)
    u = self.add_weight(
        name=u_name,
        shape=[1, shape[-1]],
        dtype=dtype,
        initializer=tf.initializers.TruncatedNormal(stddev=0.02),
        trainable=False)
    v_name = 'v' if name is None else '{}_v'.format(name)
    v = self.add_weight(
        name=v_name,
        shape=(1, np.prod(shape[:-1])),
        initializer=tf.initializers.TruncatedNormal(stddev=0.02),
        trainable=False,
        dtype=dtype)
    return u, v

  def _power_iteration(self,
                       kernel: tf.Tensor,
                       u: tf.Tensor,
                       v: tf.Tensor,
                       training: bool = False) -> tf.Tensor:
    """Runs power iteration for kernel."""
    shape = kernel.shape.as_list()
    kernel = tf.reshape(kernel, [-1, shape[-1]])
    if training:
      v_hat = tf.nn.l2_normalize(tf.matmul(u, kernel, transpose_b=True))
      u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, kernel))
      u.assign(u_hat)
      v.assign(v_hat)
    return tf.squeeze(tf.matmul(tf.matmul(v, kernel), u, transpose_b=True))


class Conv2DSpectralNorm(tf.keras.layers.Conv2D,
                         SpectralNormMixin):
  """SpectralNorm version of tf.keras.layers.Conv2D.

  This class reproduces the pytorch spectral normalization used in the original
  FaceDictGAN implementation.

  -https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html#spectral_norm
  -https://github.com/csxmli2016/DFDNet
  """

  def build(self, input_shape: Tuple[int, int, int, int]) -> None:
    """Builds U, V for singular value decompositon."""
    super().build(input_shape)
    u, v = self._create_singular_vectors(self.kernel)
    self.u = u
    self.v = v

    self.built = True

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Computes output of the block."""
    sigma = self._power_iteration(
        self.kernel, self.u, self.v, training=training)

    outputs = self._convolution_op(inputs, self.kernel / sigma)
    if self.use_bias:
      outputs = tf.nn.bias_add(
          outputs, self.bias, data_format=self._tf_data_format)
    if self.activation is None:
      return outputs
    return self.activation(outputs)

class SameBlock2d(tf.keras.Model):
  def __init__(self, num_features):
    super(SameBlock2d, self).__init__()
    self.conv = layers.Conv2D(num_features, (7, 7), strides=1, padding="same", use_bias=True)
    self.batch_norm = layers.BatchNormalization(epsilon=1e-5)
  
  def call(self, input_layer):
    block = self.conv(input_layer)
    block = self.batch_norm(block)
    block = layers.ReLU()(block)

    return block

class ResBlock2d(tf.keras.Model):
  def __init__(self, num_features):
    super(ResBlock2d, self).__init__()
    self.conv1 = layers.Conv2D(num_features, (3, 3), strides=1, padding="same", use_bias=True)
    self.conv2 = layers.Conv2D(num_features, (3, 3), strides=1, padding="same", use_bias=True)
    self.batch_norm1 = layers.BatchNormalization(epsilon=1e-5)
    self.batch_norm2 = layers.BatchNormalization(epsilon=1e-5)
  
  def call(self, input_layer):
    block = self.batch_norm1(input_layer)
    block = layers.ReLU()(block)
    block = self.conv1(block)
    block = self.batch_norm2(block)
    block = layers.ReLU()(block)
    block = self.conv2(block)

    block += input_layer

    return block

class DownBlock2d(tf.keras.Model):
  def __init__(self, num_features, norm, pool):
    super(DownBlock2d, self).__init__()
    self.norm = norm
    self.pool = pool
    self.conv = layers.Conv2D(num_features, (4, 4), strides=1, padding="valid")
    self.instance_norm = tfa.layers.InstanceNormalization(axis=3, epsilon=1e-5)
  
  def call(self, input_layer):
    block = self.conv(input_layer)
    if self.norm:
      block = self.instance_norm(block)
    block = layers.LeakyReLU(alpha=0.2)(block)
    if self.pool:
      block = layers.AveragePooling2D()(block)

    return block


class DownBlock2dWoadv(tf.keras.Model):
  def __init__(self, num_features, norm, pool):
    super(DownBlock2dWoadv, self).__init__()
    self.norm = norm
    self.pool = pool
    self.conv = Conv2DSpectralNorm(num_features, (4, 4), strides=1, padding="valid")
    self.instance_norm = tfa.layers.InstanceNormalization(axis=3, epsilon=1e-5)
  
  def call(self, input_layer):
    block = self.conv(input_layer)

    if self.norm:
      block = self.instance_norm(block)
    block = layers.LeakyReLU(alpha=0.2)(block)
    if self.pool:
      block = layers.AveragePooling2D()(block)

    return block

class DownBlock(tf.keras.Model):
  def __init__(self, num_features):
    super(DownBlock, self).__init__()
    self.padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
    self.conv = layers.Conv2D(num_features, (3, 3), strides=1, padding="same", use_bias=True)
    self.batch_norm = layers.BatchNormalization(epsilon=1e-5)
  
  def call(self, input_layer):

    block = self.conv(input_layer)
    block = self.batch_norm(block)
    block = layers.ReLU()(block)
    block = layers.AveragePooling2D()(block)

    return block

class UpBlock(tf.keras.Model):
  def __init__(self, num_features):
    super(UpBlock, self).__init__()
    self.padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
    self.conv = layers.Conv2D(num_features, (3, 3), strides=1, padding="same", use_bias=True)
    self.batch_norm = layers.BatchNormalization(epsilon=1e-5)
  
  def call(self, input_layer):
    block = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(input_layer)
    block = self.conv(block)
    block = self.batch_norm(block)
    block = layers.ReLU()(block)

    return block