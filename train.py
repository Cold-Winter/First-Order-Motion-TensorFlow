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
from networks.discriminator import Discriminator

###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
##########################for kp detector##################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

input_image = np.random.uniform(low=0.0, high=1.0, size = [1,256,256,3])
keypoint_detector = KeypointDetector()
output_kp = keypoint_detector(input_image, training=False)
print('----------------finish kp initial-----------')
print('----------------finish kp initial-----------')
print('----------------finish kp initial-----------')
print('----------------finish kp initial-----------')
print('----------------finish kp initial-----------')

# keypoint_detector.summary()
# print(keypoint_detector.predictor.encoder_blocks.layers)
# for weight_item in keypoint_detector.weights:
#   print(weight_item.name, "\t", weight_item.shape)


checkpoint_path = '/home/jinghui/first-order-model/voc_models/vox-adv-cpk.pth.tar'
checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['kp_detector']

# for layer_name in checkpoint_dict:
#   print(layer_name, checkpoint_dict[layer_name].numpy().shape)
def get_tensor_by_tfname_kpdetector(weight_name):
  layer_torch_to_tf = {
    'predictor.encoder.down_blocks.0.conv' : 'keypoint_detector/hourglass/down_block/conv2d',
    'predictor.encoder.down_blocks.0.norm' : 'keypoint_detector/hourglass/down_block/batch_normalization',
    'predictor.encoder.down_blocks.1.conv' : 'keypoint_detector/hourglass/down_block_1/conv2d_1',
    'predictor.encoder.down_blocks.1.norm' : 'keypoint_detector/hourglass/down_block_1/batch_normalization_1',
    'predictor.encoder.down_blocks.2.conv' : 'keypoint_detector/hourglass/down_block_2/conv2d_2',
    'predictor.encoder.down_blocks.2.norm' : 'keypoint_detector/hourglass/down_block_2/batch_normalization_2',
    'predictor.encoder.down_blocks.3.conv' : 'keypoint_detector/hourglass/down_block_3/conv2d_3',
    'predictor.encoder.down_blocks.3.norm' : 'keypoint_detector/hourglass/down_block_3/batch_normalization_3',
    'predictor.encoder.down_blocks.4.conv' : 'keypoint_detector/hourglass/down_block_4/conv2d_4',
    'predictor.encoder.down_blocks.4.norm' : 'keypoint_detector/hourglass/down_block_4/batch_normalization_4',
    'predictor.decoder.up_blocks.0.conv'   : 'keypoint_detector/hourglass/up_block/conv2d_5',
    'predictor.decoder.up_blocks.0.norm'   : 'keypoint_detector/hourglass/up_block/batch_normalization_5',
    'predictor.decoder.up_blocks.1.conv'   : 'keypoint_detector/hourglass/up_block_1/conv2d_6',
    'predictor.decoder.up_blocks.1.norm'   : 'keypoint_detector/hourglass/up_block_1/batch_normalization_6',
    'predictor.decoder.up_blocks.2.conv'   : 'keypoint_detector/hourglass/up_block_2/conv2d_7',
    'predictor.decoder.up_blocks.2.norm'   : 'keypoint_detector/hourglass/up_block_2/batch_normalization_7',
    'predictor.decoder.up_blocks.3.conv'   : 'keypoint_detector/hourglass/up_block_3/conv2d_8',
    'predictor.decoder.up_blocks.3.norm'   : 'keypoint_detector/hourglass/up_block_3/batch_normalization_8',
    'predictor.decoder.up_blocks.4.conv'   : 'keypoint_detector/hourglass/up_block_4/conv2d_9',
    'predictor.decoder.up_blocks.4.norm'   : 'keypoint_detector/hourglass/up_block_4/batch_normalization_9',
    'kp' : 'keypoint_detector/conv2d_10',
    'jacobian' : 'keypoint_detector/conv2d_11',
    'down' : 'keypoint_detector/anti_alias_interpolation/depthwise_conv2d',
  }
  layer_tf_to_torch = {v : k for k, v in layer_torch_to_tf.items()}
  tf_layer_name = '/'.join(weight_name.split('/')[:-1])
  torch_layer_name = layer_tf_to_torch[tf_layer_name]
  if '/conv2d' in weight_name:
    if 'kernel' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.weight'].numpy().transpose((2,3,1,0))
    elif 'bias' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.bias'].numpy()

  elif '/batch_normalization' in weight_name:
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

def set_weights_for_conv_kpdetector(conv_layer):
  weight_list = []
  for a_weight in conv_layer.weights:
    weight_name = a_weight.name
    value = get_tensor_by_tfname_kpdetector(weight_name)
    weight_list.append(value)
  conv_layer.set_weights(weight_list)


set_weights_for_conv_kpdetector(keypoint_detector.keypoints_map)
set_weights_for_conv_kpdetector(keypoint_detector.jacobian)
set_weights_for_conv_kpdetector(keypoint_detector.down)


for conv_block in keypoint_detector.predictor.layers:
  for a_layer in conv_block.layers:
    weight_list = []
    for a_weight in a_layer.weights:
      weight_name = a_weight.name
      value = get_tensor_by_tfname_kpdetector(weight_name)
      weight_list.append(value)
    a_layer.set_weights(weight_list)

# print(checkpoint_dict['predictor.encoder.down_blocks.0.conv.weight'])
# for a_weight in keypoint_detector.weights:
#   if a_weight.name == 'keypoint_detector/hourglass/down_block/conv2d/kernel:0':
#     print(tf.transpose(a_weight, perm=[2, 3, 0, 1]))

# print(checkpoint_dict['predictor.encoder.down_blocks.0.conv.bias'])
# for a_weight in keypoint_detector.weights:
#   if a_weight.name == 'keypoint_detector/hourglass/down_block/conv2d/bias:0':
#     print(a_weight)

# print(checkpoint_dict['predictor.encoder.down_blocks.0.norm.running_var'])
# for a_weight in keypoint_detector.weights:
#   if a_weight.name == 'keypoint_detector/hourglass/down_block/batch_normalization/moving_variance:0':
#     print(a_weight)

source_image = imageio.imread('/home/jinghui/first-order-model/yandong.jpeg')
source_image = resize(source_image, (256, 256))[..., :3]
source_image = source_image.reshape((1,256,256,3))

driving_image = imageio.imread('/home/jinghui/first-order-model/yuting.png')
driving_image = resize(driving_image, (256, 256))[..., :3]
driving_image = driving_image.reshape((1,256,256,3))

kp_source = keypoint_detector(source_image, training=False)
kp_driving = keypoint_detector(driving_image, training=False)

print(kp_source)
print(kp_driving)

###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
##########################for generator####################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
generator = Generator()

out_generator = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)
print('----------------finish generator initial-----------')
print('----------------finish generator initial-----------')
print('----------------finish generator initial-----------')
print('----------------finish generator initial-----------')

# for weight_item in generator.weights:
#   print(weight_item.name, "\t", weight_item.shape)

checkpoint_path = '/home/jinghui/first-order-model/voc_models/vox-adv-cpk.pth.tar'
checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['generator']


def get_tensor_by_tfname_generator(weight_name):
  layer_torch_to_tf = {
    'first.conv' : 'generator/same_block2d/conv2d_12',
    'first.norm' : 'generator/same_block2d/batch_normalization_10',
    'down_blocks.0.conv' : 'generator/down_block_5/conv2d_13',
    'down_blocks.0.norm' : 'generator/down_block_5/batch_normalization_11',
    'down_blocks.1.conv' : 'generator/down_block_6/conv2d_14',
    'down_blocks.1.norm' : 'generator/down_block_6/batch_normalization_12',
    'dense_motion_network.down' : 'generator/dense_motion_network/anti_alias_interpolation_1/depthwise_conv2d_1',
    'dense_motion_network.hourglass.encoder.down_blocks.0.conv' : 'generator/dense_motion_network/hourglass_1/down_block_7/conv2d_15',
    'dense_motion_network.hourglass.encoder.down_blocks.0.norm' : 'generator/dense_motion_network/hourglass_1/down_block_7/batch_normalization_13',
    'dense_motion_network.hourglass.encoder.down_blocks.1.conv' : 'generator/dense_motion_network/hourglass_1/down_block_8/conv2d_16',
    'dense_motion_network.hourglass.encoder.down_blocks.1.norm' : 'generator/dense_motion_network/hourglass_1/down_block_8/batch_normalization_14',
    'dense_motion_network.hourglass.encoder.down_blocks.2.conv' : 'generator/dense_motion_network/hourglass_1/down_block_9/conv2d_17',
    'dense_motion_network.hourglass.encoder.down_blocks.2.norm' : 'generator/dense_motion_network/hourglass_1/down_block_9/batch_normalization_15',
    'dense_motion_network.hourglass.encoder.down_blocks.3.conv' : 'generator/dense_motion_network/hourglass_1/down_block_10/conv2d_18',
    'dense_motion_network.hourglass.encoder.down_blocks.3.norm' : 'generator/dense_motion_network/hourglass_1/down_block_10/batch_normalization_16',
    'dense_motion_network.hourglass.encoder.down_blocks.4.conv' : 'generator/dense_motion_network/hourglass_1/down_block_11/conv2d_19',
    'dense_motion_network.hourglass.encoder.down_blocks.4.norm' : 'generator/dense_motion_network/hourglass_1/down_block_11/batch_normalization_17',
    'dense_motion_network.hourglass.decoder.up_blocks.0.conv'   : 'generator/dense_motion_network/hourglass_1/up_block_5/conv2d_20',
    'dense_motion_network.hourglass.decoder.up_blocks.0.norm'   : 'generator/dense_motion_network/hourglass_1/up_block_5/batch_normalization_18',
    'dense_motion_network.hourglass.decoder.up_blocks.1.conv'   : 'generator/dense_motion_network/hourglass_1/up_block_6/conv2d_21',
    'dense_motion_network.hourglass.decoder.up_blocks.1.norm'   : 'generator/dense_motion_network/hourglass_1/up_block_6/batch_normalization_19',
    'dense_motion_network.hourglass.decoder.up_blocks.2.conv'   : 'generator/dense_motion_network/hourglass_1/up_block_7/conv2d_22',
    'dense_motion_network.hourglass.decoder.up_blocks.2.norm'   : 'generator/dense_motion_network/hourglass_1/up_block_7/batch_normalization_20',
    'dense_motion_network.hourglass.decoder.up_blocks.3.conv'   : 'generator/dense_motion_network/hourglass_1/up_block_8/conv2d_23',
    'dense_motion_network.hourglass.decoder.up_blocks.3.norm'   : 'generator/dense_motion_network/hourglass_1/up_block_8/batch_normalization_21',
    'dense_motion_network.hourglass.decoder.up_blocks.4.conv'   : 'generator/dense_motion_network/hourglass_1/up_block_9/conv2d_24',
    'dense_motion_network.hourglass.decoder.up_blocks.4.norm'   : 'generator/dense_motion_network/hourglass_1/up_block_9/batch_normalization_22',
    'dense_motion_network.mask' : 'generator/dense_motion_network/conv2d_25',
    'dense_motion_network.occlusion' : 'generator/dense_motion_network/conv2d_26',
    'up_blocks.0.conv' : 'generator/up_block_10/conv2d_27',
    'up_blocks.0.norm' : 'generator/up_block_10/batch_normalization_23',
    'up_blocks.1.conv' : 'generator/up_block_11/conv2d_28',
    'up_blocks.1.norm' : 'generator/up_block_11/batch_normalization_24',
    'bottleneck.r0.conv1' : 'res_block2d/conv2d_29' ,
    'bottleneck.r0.conv2' : 'res_block2d/conv2d_30' ,
    'bottleneck.r0.norm1' : 'res_block2d/batch_normalization_25',
    'bottleneck.r0.norm2' : 'res_block2d/batch_normalization_26',
    'bottleneck.r1.conv1' : 'res_block2d_1/conv2d_31',
    'bottleneck.r1.conv2' : 'res_block2d_1/conv2d_32',
    'bottleneck.r1.norm1' : 'res_block2d_1/batch_normalization_27',
    'bottleneck.r1.norm2' : 'res_block2d_1/batch_normalization_28',
    'bottleneck.r2.conv1' : 'res_block2d_2/conv2d_33',
    'bottleneck.r2.conv2' : 'res_block2d_2/conv2d_34',
    'bottleneck.r2.norm1' : 'res_block2d_2/batch_normalization_29',
    'bottleneck.r2.norm2' : 'res_block2d_2/batch_normalization_30',
    'bottleneck.r3.conv1' : 'res_block2d_3/conv2d_35',
    'bottleneck.r3.conv2' : 'res_block2d_3/conv2d_36',
    'bottleneck.r3.norm1' : 'res_block2d_3/batch_normalization_31',
    'bottleneck.r3.norm2' : 'res_block2d_3/batch_normalization_32',
    'bottleneck.r4.conv1' : 'res_block2d_4/conv2d_37',
    'bottleneck.r4.conv2' : 'res_block2d_4/conv2d_38',
    'bottleneck.r4.norm1' : 'res_block2d_4/batch_normalization_33',
    'bottleneck.r4.norm2' : 'res_block2d_4/batch_normalization_34',
    'bottleneck.r5.conv1' : 'res_block2d_5/conv2d_39',
    'bottleneck.r5.conv2' : 'res_block2d_5/conv2d_40',
    'bottleneck.r5.norm1' : 'res_block2d_5/batch_normalization_35',
    'bottleneck.r5.norm2' : 'res_block2d_5/batch_normalization_36',
    'final' : 'generator/conv2d_41',
  }
  layer_tf_to_torch = {v : k for k, v in layer_torch_to_tf.items()}
  tf_layer_name = '/'.join(weight_name.split('/')[:-1])
  torch_layer_name = layer_tf_to_torch[tf_layer_name]
  if '/conv2d' in weight_name:
    if 'kernel' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.weight'].numpy().transpose((2,3,1,0))
    elif 'bias' in weight_name:
      value = checkpoint_dict[torch_layer_name+'.bias'].numpy()

  elif '/batch_normalization' in weight_name:
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

def set_weights_for_conv_generator(conv_layer):
  weight_list = []
  for a_weight in conv_layer.weights:
    weight_name = a_weight.name
    value = get_tensor_by_tfname_generator(weight_name)
    weight_list.append(value)
  conv_layer.set_weights(weight_list)

def set_weights_for_convblock_generator(conv_block):
  # for conv_block in network_module.layers:
  for a_layer in conv_block.layers:
    weight_list = []
    for a_weight in a_layer.weights:
      weight_name = a_weight.name
      value = get_tensor_by_tfname_generator(weight_name)
      weight_list.append(value)
    a_layer.set_weights(weight_list)

set_weights_for_convblock_generator(generator.first)
set_weights_for_conv_generator(generator.final)

set_weights_for_convblock_generator(generator.encoder_blocks)
set_weights_for_convblock_generator(generator.decoder_blocks)
set_weights_for_convblock_generator(generator.bottleneck)

for conv_block in generator.dense_motion_network.hourglass.layers:
  for a_layer in conv_block.layers:
    weight_list = []
    for a_weight in a_layer.weights:
      weight_name = a_weight.name
      value = get_tensor_by_tfname_generator(weight_name)
      weight_list.append(value)
    a_layer.set_weights(weight_list)

set_weights_for_conv_generator(generator.dense_motion_network.down)
set_weights_for_conv_generator(generator.dense_motion_network.mask)
set_weights_for_conv_generator(generator.dense_motion_network.occlusion)


out_generator = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)
keypoint_detector.save_weights('./checkpoints/kp_detector/kp_detector')
generator.save_weights('./checkpoints/generator/generator')








  # print(weight_item.shape)

# generator = Generator()
# discriminator = Discriminator()

# generator_full = FullGenerator(keypoint_detector, generator, discriminator)
# discriminator_full = FullDiscriminator(discriminator)

# @tf.function
# def train_step(source_images, driving_images):
#   with tf.GradientTape(persistent=True) as tape: 
#     losses_generator, generated = generator_full(source_images, driving_images, tape)
#     generator_loss = tf.math.reduce_sum(list(losses_generator.values()))

#   generator_gradients = tape.gradient(generator_loss, generator_full.trainable_variables)
#   keypoint_detector_gradients = tape.gradient(generator_loss, keypoint_detector.trainable_variables)

#   optimizer_generator.apply_gradients(zip(generator_gradients, generator_full.trainable_variables))
#   optimizer_keypoint_detector.apply_gradients(zip(keypoint_detector_gradients, keypoint_detector.trainable_variables))

#   with tf.GradientTape() as tape:
#     losses_discriminator = discriminator_full(x)
#     discriminator_loss = tf.math.reduce_sum(list(losses_discriminator.values()))
  
#   discriminator_gradients = tape.gradient(discriminator_loss, discriminator_full.trainable_variables)
#   optimizer_discriminator.apply_gradients(zip(discriminator_gradients, discriminator_full.trainable_variables))

#   return generator_loss + discriminator_loss

# def decay_lr(optimizer, epoch):
#   if epoch >= 60 and epoch <= 90:
#     current_lr = tf.keras.backend.get_value(optimizer.lr)
#     new_lr = current_lr * 0.1
#     tf.keras.backend.set_value(optimizer.lr, new_lr)

# loss_results = []

# def train(epochs, total_steps):
#   for epoch in range(epochs):
#     batch_time = time.time()
#     epoch_time = time.time()
#     step = 0

#     epoch_count = f"0{epoch + 1}/{epochs}" if epoch < 9 else f"{epoch + 1}/{epochs}"

#     for source_images, driving_images in zip(images_batches, labels_batches, masks_batches):
#       total_loss = train_step(source_images, driving_images)

#       loss = float(loss.numpy())
#       step += 1

#       print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
#             '| loss:', f"{loss:.5f}", "| Step time:", f"{time.time() - batch_time:.2f}", end='')    
      
#       batch_time = time.time()
#       total_steps += 1

#     loss_results.append(loss)
#     decay_lr(optimizer_generator, epoch)
#     decay_lr(optimizer_keypoint_detector, epoch)
#     decay_lr(optimizer_discriminator, epoch)

#     print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
#           '| loss:', "| Epoch time:", f"{time.time() - epoch_time:.2f}")