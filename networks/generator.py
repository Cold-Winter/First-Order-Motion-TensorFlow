import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from utils.util import interpolate_tensor, make_coordinate_grid
from utils.blocks import DownBlock, UpBlock, SameBlock2d, ResBlock2d
from networks.dense_motion_network import DenseMotionNetwork

class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    block_expansion = 64
    self.padding = [[0, 0], [3, 3], [3, 3], [0, 0]] # pad only height, width
    self.num_channels = 3
    self.num_keypoints = 10
    self.num_blocks = 2
    self.num_bottleneck_blocks = 6
    
    # 
    self.first = SameBlock2d(block_expansion)

    self.down_features_list = [128, 256]
    self.up_features_list = [128, 64] 

    encoder_blocks = []

    for i in range(self.num_blocks):
      num_features = self.down_features_list[i]
      encoder_blocks.append(DownBlock(num_features))
    
    self.encoder_blocks = encoder_blocks

    self.dense_motion_network = DenseMotionNetwork()

    decoder_blocks = []

    for i in range(self.num_blocks):
      num_features = self.up_features_list[i]
      decoder_blocks.append(UpBlock(num_features))
    
    self.decoder_blocks = decoder_blocks

    self.bottleneck = tf.keras.Sequential()

    for i in range(self.num_bottleneck_blocks):
      self.bottleneck.add(ResBlock2d(self.down_features_list[-1]))

    self.final = layers.Conv2D(self.num_channels, (7, 7), strides=1, padding="same")

  def deform_input(self, x, deformation):
    _, height_old, width_old, _ = deformation.shape
    _, height, width, _ = x.shape

    if height_old != height or width_old != width:
      # [Yandong To Do]interpolate_tensor is not equvilant with pytorch F.interpolate
      # deformation = interpolate_tensor(deformation, width)
      deformation = tf.image.resize(deformation, [width,width], method='bilinear')

    # new_max = width - 1
    # new_min = 0
    # deformation = (new_max - new_min) / (tf.keras.backend.max(deformation) - tf.keras.backend.min(deformation)) * (deformation - tf.keras.backend.max(deformation)) + new_max

    deformation = ((deformation + 1.0) * width - 1.0) * 0.5

    return tfa.image.resampler(x, deformation)
  
  def call(self, source_image, kp_driving, kp_source):
    out = self.first(source_image)

    for down_block in self.encoder_blocks:
      out = down_block(out)

    # print('----------------after down_block -----------')
    # feature_map_trans = tf.transpose(out, perm=[0, 3, 1, 2])
    # print(feature_map_trans.shape)
    # print(feature_map_trans)
    # print('----------------end down_block -----------')
    
    output_dict = {}

    dense_motion = self.dense_motion_network(source_image, kp_driving, kp_source)

    # Debug/print                                       
    output_dict['mask'] = dense_motion['mask']
    # Debug/print
    output_dict['warped_images'] = dense_motion['warped_images'] # sparse_deformed

    occlusion_map = dense_motion['occlusion_map']
    # shape batch x 256 x 256 x 1
    
    # Debug/print
    output_dict['occlusion_map'] = occlusion_map

    dense_optical_flow = dense_motion['dense_optical_flow'] # deformation
    # batch x 256 x 256 x 2 
    out = self.deform_input(out, dense_optical_flow)
    # batch x 256 x 256 x 2 

    if out.shape[1] != occlusion_map.shape[1] or out.shape[2] != occlusion_map.shape[2]:
      # occlusion_map = interpolate_tensor(occlusion_map, out[1])
      occlusion_map = tf.image.resize(occlusion_map, [out[1],out[2]], method='bilinear')
    
    out = out * occlusion_map

    # # Debug/print
    output_dict["aligned_features"] = self.deform_input(source_image, dense_optical_flow) # deformed
    # [Yandong To Do Fix the bug for interpolate]
    # print('----------------after down_block -----------')
    # feature_map_trans = tf.transpose(output_dict["aligned_features"], perm=[0, 3, 1, 2])
    # print(feature_map_trans.shape)
    # print(feature_map_trans)
    # print('----------------end down_block -----------')

    # Decoder part

    out = self.bottleneck(out)

    for up_block in self.decoder_blocks:
      out = up_block(out)

    out = self.final(out)
    out = tf.keras.activations.sigmoid(out)

    # print('----------------after down_block -----------')
    # feature_map_trans = tf.transpose(out, perm=[0, 3, 1, 2])
    # print(feature_map_trans.shape)
    # print(feature_map_trans)
    # print('----------------end down_block -----------')

    output_dict["prediction"] = out

    return output_dict



