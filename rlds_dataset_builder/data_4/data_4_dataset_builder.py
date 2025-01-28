# RICORDATI CHE IL NUOVO DATASET DI PROVA SI CHIAMA `data_1` E CHE IL NOME DEL FILE DEVE ESSERE `data_1_dataset_builder.py`


from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class Data5(tfds.core.GeneratorBasedBuilder):
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
                        dtype=np.bool_,                                # CAMBIARE DIMENSIONE (bool on int64)
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,                               # CAMBIARE DIMENSIONE (bool on int64)
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,                             # CAMBIARE DIMENSIONE (bool on int64)
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
            # 'val': self._generate_examples(path='data/val/episode_0.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""



        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True).item()     # this is a list of dicts in our case

            # # Verifica che le chiavi attese siano presenti
            # required_keys = ["images_front", "images_side", "images_hand", "states", "actions"]
            # for key in required_keys:
            #     if key not in data:
            #         raise KeyError(f"Chiave mancante nel file `{episode_path}`: {key}")

            # Estrai i dati
            images_front = data["images_front"]
            images_side = data["images_side"]
            images_hand = data["images_hand"]
            states = data["states"]
            actions = data["actions"]
            natural_language_instruction = data["natural_language"]

            # Verifica che le lunghezze dei dati corrispondano
            if not (len(images_front) == len(images_side) == len(images_hand) == len(states) == len(actions)):
                raise ValueError("Mismatch tra le lunghezze di immagini, stati e azioni in `{episode_path}`")

            # Assemble episode
            episode = []
            for i in range(len(states)):
                # Compute language embedding (usa un'istruzione fissa per ora)
                language_embedding = self._embed([natural_language_instruction[0]])[0].numpy()
                # language_embedding = self._embed(["pick up the red cube"])[0].numpy()


                episode.append({
                    'observation': {
                        'image_front': images_front[i],
                        'image_side': images_side[i],
                        'image_hand': images_hand[i],
                        'state': states[i],
                    },
                    'action': actions[i],
                    'discount': 1.0,
                    'reward': float(i == (len(states) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(states) - 1),
                    'is_terminal': i == (len(states) - 1),
                    'natural_language_instruction': natural_language_instruction[i],
                    'natural_language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

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

