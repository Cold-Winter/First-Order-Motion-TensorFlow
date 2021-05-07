import tensorflow as tf
from tensorflow.keras import layers
import time

from networks.full_generator import FullGenerator
from networks.full_discriminator import FullDiscriminator
from networks.keypoint_detector import KeypointDetector
from networks.generator import Generator
from networks.discriminator import Discriminator
from frames_dataset import FramesDataset
import frames_dataset
import numpy as np

lr = 2e-4


dataset_params = {
  'frame_shape': [256, 256, 3], 'id_sampling': False, 'pairs_list': 'data/vox256.csv', 
  'augmentation_params': {'flip_param': {'horizontal_flip': True, 'time_flip': True}, 
  'jitter_param': {'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.1}}}

train_dataset = FramesDataset(is_train=True, **dataset_params).get_dataset()

  # print(data[0])
  # print(data[1])
  # print(data[2])

optimizer_generator = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
optimizer_keypoint_detector = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
optimizer_discriminator = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)

batch_size = 20
epochs = 150
train_steps = 99 # change

keypoint_detector = KeypointDetector()
generator = Generator()
discriminator = Discriminator()

generator_full = FullGenerator(keypoint_detector, generator, discriminator)
discriminator_full = FullDiscriminator(discriminator)

@tf.function
def train_step(source_images, driving_images):
  with tf.GradientTape(persistent=True) as tape: 
    losses_generator, generated = generator_full(source_images, driving_images, tape)
    generator_loss = tf.math.reduce_sum(list(losses_generator.values()))

  generator_gradients = tape.gradient(generator_loss, generator_full.trainable_variables)
  keypoint_detector_gradients = tape.gradient(generator_loss, keypoint_detector.trainable_variables)

  optimizer_generator.apply_gradients(zip(generator_gradients, generator_full.trainable_variables))
  optimizer_keypoint_detector.apply_gradients(zip(keypoint_detector_gradients, keypoint_detector.trainable_variables))

  with tf.GradientTape() as tape:
    # comment by yandong
    # losses_discriminator = discriminator_full(x)
    losses_discriminator = discriminator_full(driving_images, generated)
    discriminator_loss = tf.math.reduce_sum(list(losses_discriminator.values()))
  
  discriminator_gradients = tape.gradient(discriminator_loss, discriminator_full.trainable_variables)
  optimizer_discriminator.apply_gradients(zip(discriminator_gradients, discriminator_full.trainable_variables))

  return generator_loss + discriminator_loss

def decay_lr(optimizer, epoch):
  if epoch >= 60 and epoch <= 90:
    current_lr = tf.keras.backend.get_value(optimizer.lr)
    new_lr = current_lr * 0.1
    tf.keras.backend.set_value(optimizer.lr, new_lr)

loss_results = []

def train(epochs, total_steps):
  for epoch in range(epochs):
    batch_time = time.time()
    epoch_time = time.time()
    step = 0

    epoch_count = f"0{epoch + 1}/{epochs}" if epoch < 9 else f"{epoch + 1}/{epochs}"

    # for source_images, driving_images in zip(images_batches, labels_batches, masks_batches):
    for data in train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE):
      file_names = data.numpy()
      driving_images = []
      source_images = []
      for file_name in file_names:
        file_name = file_name.decode("utf-8") 
        video_array = frames_dataset.read_video(file_name, [256, 256, 3])
        num_frames = len(video_array)
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) 
        video_array = video_array[frame_idx]
        source_images.append(video_array[0].reshape(1,256,256,3))
        driving_images.append(video_array[1].reshape(1,256,256,3))
      source_images = np.concatenate(source_images, axis=0)
      driving_images = np.concatenate(driving_images, axis=0)

      total_loss = train_step(source_images, driving_images)

      loss = float(total_loss.numpy())
      step += 1

      print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
            '| loss:', f"{loss:.5f}", "| Step time:", f"{time.time() - batch_time:.2f}", end='')    
      
      batch_time = time.time()
      total_steps += 1

    loss_results.append(loss)
    decay_lr(optimizer_generator, epoch)
    decay_lr(optimizer_keypoint_detector, epoch)
    decay_lr(optimizer_discriminator, epoch)

    print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
          '| loss:', "| Epoch time:", f"{time.time() - epoch_time:.2f}")
train(200, 100)