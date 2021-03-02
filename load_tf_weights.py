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

input_image = np.random.uniform(low=0.0, high=1.0, size = [1,256,256,3])
keypoint_detector = KeypointDetector()
output_kp = keypoint_detector(input_image, training=False)

source_image = imageio.imread('/home/jinghui/first-order-model/yandong.jpeg')
source_image = resize(source_image, (256, 256))[..., :3]
source_image = source_image.reshape((1,256,256,3))

driving_image = imageio.imread('/home/jinghui/first-order-model/yuting.png')
driving_image = resize(driving_image, (256, 256))[..., :3]
driving_image = driving_image.reshape((1,256,256,3))

generator = Generator()

kp_source = keypoint_detector(source_image, training=False)
kp_driving = keypoint_detector(driving_image, training=False)
out_generator = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)

print('----------------finish generator initial-----------')
print('----------------finish generator initial-----------')
print('----------------finish generator initial-----------')
print('----------------finish generator initial-----------')

keypoint_detector.load_weights('./checkpoints/kp_detector/kp_detector')
generator.load_weights('./checkpoints/generator/generator')

kp_source = keypoint_detector(source_image, training=False)
kp_driving = keypoint_detector(driving_image, training=False)
out_generator = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)