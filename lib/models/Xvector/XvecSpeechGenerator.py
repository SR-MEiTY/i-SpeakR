
import numpy as np
import torch
import utils


class XvecSpeechGenerator():
    """Speech dataset."""

    def __init__(self, manifest, mode, win_length, n_fft):
        """
        Read the textfile and get the paths
        """
        self.mode = mode
        self.win_length = win_length
        self.n_fft = n_fft
        self.audio_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        self.labels = [int(line.rstrip('\n').split(' ')[1]) for line in open(manifest)]

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        # print(f'class_id={class_id}')
        win_length = self.win_length
        n_fft = self.n_fft
        # lang_label=lang_id[self.audio_links[idx].split('/')[-2]]
        spec = np.load(audio_link, allow_pickle=True)
        sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),
                  'labels': torch.from_numpy(np.ascontiguousarray(class_id)),
                  'path': audio_link
                  }
        return sample

