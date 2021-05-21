import tensorflow as tf
from tensorflow.keras import layers
import time
import numpy as np
import torch
import imageio
from skimage.transform import resize

from networks.full_generator import FullGenerator
from networks.full_discriminator import FullDiscriminator
from networks.keypoint_detector import KeypointDetector
from networks.generator import Generator
from networks.discriminator import DiscriminatorWoadv as Discriminator

from utils.util import detach_keypoint

input_image = np.random.uniform(low=0.0, high=1.0, size = [1,256,256,3])
keypoint_detector = KeypointDetector()
output_kp = keypoint_detector(input_image, training=False)

# source_image = imageio.imread('/home/jinghui/first-order-model/yandong.jpeg')
source_image = imageio.imread('/home/jinghui/First-Order-Motion-TensorFlow/videos/id10007/image-00001.png')
source_image = resize(source_image, (256, 256))[..., :3]
source_image = source_image.reshape((1,256,256,3))

# driving_image = imageio.imread('/home/jinghui/first-order-model/yuting.png')
driving_image = imageio.imread('/home/jinghui/First-Order-Motion-TensorFlow/videos/id10008/image-00001.png')
driving_image = resize(driving_image, (256, 256))[..., :3]
driving_image = driving_image.reshape((1,256,256,3))

generator = Generator()


kp_source = keypoint_detector(source_image, training=True)
kp_driving = keypoint_detector(driving_image, training=True)

out_generator = generator(input_image, kp_source=output_kp, kp_driving=output_kp)

print('----------------finish generator initial-----------')
print('----------------finish generator initial-----------')
print('----------------finish generator initial-----------')
print('----------------finish generator initial-----------')

keypoint_detector.load_weights('./checkpoints/kp_detector_woadv/kp_detector')
generator.load_weights('./checkpoints/generator_woadv/generator')

# with tf.GradientTape(persistent=True) as tape:
kp_source = keypoint_detector(source_image, training=True)
kp_driving = keypoint_detector(driving_image, training=True)
out_generator = generator(source_image, kp_source=kp_source, kp_driving=kp_driving, training=True)

# print(kp_source)
# print(kp_driving)
# print('----------------after down_block -----------')
# feature_map_trans = tf.transpose(out_generator['prediction'], perm=[0, 3, 1, 2])
# print(feature_map_trans.shape)
# print(feature_map_trans)
# print('----------------end down_block -----------')


discriminator = Discriminator()
discriminator_maps_real, _ = discriminator(driving_image, key_points=detach_keypoint(kp_driving))
print('----------------finish discriminator initial-----------')
print('----------------finish discriminator initial-----------')
print('----------------finish discriminator initial-----------')
print('----------------finish discriminator initial-----------')


# for weight_item in discriminator.weights:
#   print(weight_item.name, "\t", weight_item.shape)


checkpoint_path = '/home/jinghui/first-order-model/voc_models/vox-cpk.pth.tar'
# checkpoint_path = '/home/jinghui/First-Order-Motion-TensorFlow/ckpt_torch/149_5000imgs.pth.tar'
checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['discriminator']

# count = 0
# for k, v in checkpoint_dict.items():
#   if 'num_batches_tracked' in k:
#     continue
#   print('%03d: %s     ' % (count, k), v.shape)
#   # print(v.shape)
#   count += 1

def get_tensor_by_tfname_dis(weight_name):
  layer_torch_to_tf = {
    'discs.1.down_blocks.0.conv' : 'discriminator_woadv/down_block2d_woadv/conv2d_spectral_norm',
    'discs.1.down_blocks.1.conv' : 'discriminator_woadv/down_block2d_woadv_1/conv2d_spectral_norm_1',
    'discs.1.down_blocks.1.norm' : 'discriminator_woadv/down_block2d_woadv_1/instance_normalization_1',
    'discs.1.down_blocks.2.conv' : 'discriminator_woadv/down_block2d_woadv_2/conv2d_spectral_norm_2',
    'discs.1.down_blocks.2.norm' : 'discriminator_woadv/down_block2d_woadv_2/instance_normalization_2',
    'discs.1.down_blocks.3.conv' : 'discriminator_woadv/down_block2d_woadv_3/conv2d_spectral_norm_3',
    'discs.1.down_blocks.3.norm' : 'discriminator_woadv/down_block2d_woadv_3/instance_normalization_3',
    'discs.1.conv'               : 'discriminator_woadv/conv2d_spectral_norm_4',
 
  }
  layer_tf_to_torch = {v : k for k, v in layer_torch_to_tf.items()}
  tf_layer_name = '/'.join(weight_name.split('/')[:-1])
  torch_layer_name = layer_tf_to_torch[tf_layer_name]
  if '/conv2d' in weight_name:
    if 'kernel' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.weight_orig'].numpy().transpose((2,3,1,0))
    elif 'bias' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.bias'].numpy()
    elif 'u:0' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.weight_u'].numpy().reshape((1,-1))
    elif 'v:0' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.weight_v'].numpy().reshape((1,-1))


  elif '/instance_normalization' in weight_name:
    if 'gamma' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.weight'].numpy()
    elif 'beta' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.bias'].numpy()
    elif 'moving_mean' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.running_mean'].numpy()
    elif 'moving_variance' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.running_var'].numpy()
  # ask xuhui for AntiAliasInterpolation 2 3 0 1
  elif '/depthwise_conv2d' in weight_name:
    if 'depthwise_kernel' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.weight'].numpy().transpose((2,3,0,1))
  return value


def set_weights_for_conv_dis(conv_layer):
  weight_list = []
  for a_weight in conv_layer.weights:
    weight_name = a_weight.name
    value = get_tensor_by_tfname_dis(weight_name)
    a_weight.assign(value)
  #   weight_list.append(value)
  # conv_layer.set_weights(weight_list)

def set_weights_for_convblock_dis(conv_block):
  # for conv_block in network_module.layers:
  for a_layer in conv_block.layers:
    weight_list = []
    for a_weight in a_layer.weights:
      weight_name = a_weight.name
      value = get_tensor_by_tfname_dis(weight_name)
      a_weight.assign(value)
    #   weight_list.append(value)
    # a_layer.set_weights(weight_list)


set_weights_for_convblock_dis(discriminator.encoder_blocks)
set_weights_for_conv_dis(discriminator.conv)
discriminator.save_weights('./checkpoints/discriminator_woadv/discriminator')

# discriminator_maps_real, discriminator_pred_map = discriminator(driving_image, key_points=detach_keypoint(kp_driving), training = True)


# print('----------------after discriminator -----------')
# feature_map_trans = tf.transpose(discriminator_pred_map, perm=[0, 3, 1, 2])
# print(feature_map_trans.shape)
# print(feature_map_trans)
# print('----------------end discriminator -----------')
generator_full = FullGenerator(keypoint_detector, generator, discriminator)
discriminator_full = FullDiscriminator(discriminator)

with tf.GradientTape(persistent=True) as tape: 
    losses_generator, generated = generator_full(source_image, driving_image, tape, training = True)
    generator_loss = tf.math.reduce_sum(list(losses_generator.values()))

print(losses_generator)

with tf.GradientTape() as tape:
    # comment by yandong
    # losses_discriminator = discriminator_full(x)
    losses_discriminator = discriminator_full(driving_image, generated, training = True)
    print(losses_discriminator)

# for weight_item in discriminator.weights:
#   if 'discriminator_woadv/down_block2d_woadv/conv2d_spectral_norm' in weight_item.name:
#     print(weight_item.name, "\t", weight_item)

# count = 0
# for k, v in checkpoint_dict.items():
#   if 'num_batches_tracked' in k:
#     continue
#   if 'discs.1.down_blocks.0.conv' in k:
#     print(k)
#     if len(v.shape) == 4:
#       v = v.numpy().transpose((2,3,1,0))
#     print(v)

#   # print(v.shape)
#   count += 1















