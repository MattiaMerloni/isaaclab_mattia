from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

def resize(image):
    image = tf.image.resize_with_pad(image, target_width=224, target_height=224)
    image = tf.cast(image, tf.uint8)
    return image

def encode_image_as_png(image):
    """Encodes a TensorFlow image tensor as a PNG byte string."""
    return tf.image.encode_png(image).numpy()

class Fewreal(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'images': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Action consists of [3x robot delta displacements, '
                            '3x delta rotations and gripper closedness action].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'natural_language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'natural_language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # Load raw data
            data = np.load(episode_path, allow_pickle=True).item()

            # Extract data
            images = data["images"]
            actions = data["actions"]
            natural_language_instruction = "pick up the red cuboid"  # Fixed instruction for now

            # Verify data lengths
            if not (len(images) == len(actions)):
                raise ValueError(f"Mismatch in lengths of images and actions in `{episode_path}`")

            # Assemble episode
            episode = []
            for i in range(len(actions)):
                # Resize image
                resized_image = resize(images[i])

                # Encode image as PNG
                encoded_image = encode_image_as_png(resized_image)

                # Compute language embedding
                language_embedding = self._embed(["pick up the red cuboid"])[0].numpy()

                episode.append({
                    'observation': {
                        'images': encoded_image,  # Pass the encoded image
                    },
                    'action': actions[i],
                    'discount': 1.0,
                    'reward': float(i == (len(actions) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(actions) - 1),
                    'is_terminal': i == (len(actions) - 1),
                    'natural_language_instruction': natural_language_instruction,
                    'natural_language_embedding': language_embedding,
                })

            # Create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            return episode_path, sample

        # Create list of all examples
        episode_paths = glob.glob(path)

        # For smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)