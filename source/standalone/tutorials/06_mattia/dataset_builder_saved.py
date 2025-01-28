# FILE NON BUONO CHE USO SOLO COME SPUNTO 



from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class DataFranka1(tfds.core.GeneratorBasedBuilder):
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
                        'image_front': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'image_side': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Side camera RGB observation.',
                        ),
                        'image_hand': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Hand camera RGB observation.',
                        ),      
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [3x robot displacements, '
                                '3x rotations and gripper closedness].',
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
            'train': self._generate_examples(path='train_dataset/episode_*.npy'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""



        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding
                language_embedding = self._embed([step['natural_language_instruction']])[0].numpy()

                episode.append({
                    'observation': {
                        'image_front': step['image_front'].resize((224, 224)),    # CHIEDERE
                        'image_side': step['image_side'].resize((224, 224)),
                        'image_hand': step['image_hand'].resize((224, 224)),
                        'state': step['state'],
                    },
                    'action': step['action'],
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'natural_language_instruction': step['natural_language_instruction'],
                    'natural_language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

