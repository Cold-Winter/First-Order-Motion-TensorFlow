import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
import pandas as pd
from augmentation import AllAugmentationTransform
import glob

import tensorflow as tf



_ROOT_DIR_ = '/data/jinghui/vox'


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if tf.io.gfile.isdir(name):
        frames = sorted(tf.io.gfile.listdir(name))
        num_frames = len(frames)
        print(num_frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

# @tf.function
def parse_fn_train(name):

    # image = tf.io.read_file(name)
    # # Decode the jpeg image to array [0, 255].
    # image = tf.image.decode_jpeg(image)
    # image = tf.image.resize(image, [256, 256])
    # print(image.shape)
    
    return name

class FramesDataset():
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):

        self.videos = os.listdir(_ROOT_DIR_)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(_ROOT_DIR_, 'train')):
            assert os.path.exists(os.path.join(_ROOT_DIR_, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(_ROOT_DIR_, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(_ROOT_DIR_, 'train'))
            train_videos = [os.path.join(os.path.join(_ROOT_DIR_, 'train'),video_name) for video_name in train_videos]
            test_videos = os.listdir(os.path.join(_ROOT_DIR_, 'test'))
            test_videos = [os.path.join(os.path.join(_ROOT_DIR_, 'test'),video_name) for video_name in test_videos]

        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

    


        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

 

    def get_dataset(self):

        ds = tf.data.Dataset.from_tensor_slices(self.videos)
        ds = ds.map(parse_fn_train, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(12).repeat()
        return ds




