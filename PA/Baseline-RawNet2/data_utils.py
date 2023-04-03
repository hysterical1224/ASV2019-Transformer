import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
# from joblib import Parallel, delayed
import soundfile as sf
import collections

___author__ = "Shen Song"
__email__ = "1403826619@qq.com"

LOGICAL_DATA_ROOT = r'I:\2021\data\LA'
PHISYCAL_DATA_ROOT = r'I:\2021\data\PA'

sysid_dict= {
        '-': 0,  # bonafide speech
        'SS_1': 1,  # Wavenet vocoder
        'SS_2': 2,  # Conventional vocoder WORLD
        'SS_4': 3,  # Conventional vocoder MERLIN
        'US_1': 4,  # Unit selection system MaryTTS
        'VC_1': 5,  # Voice conversion using neural networks
        'VC_4': 6,  # transform function-based voice conversion
        # For PA:
        'AA': 7,
        'AB': 8,
        'AC': 9,
        'BA': 10,
        'BB': 11,
        'BC': 12,
        'CA': 13,
        'CB': 14,
        'CC': 15
    }

def genSpoof_list(is_logical, is_eval, is_train):
    if is_logical:
        data_root = LOGICAL_DATA_ROOT
        track = 'LA'
    else:
        data_root = PHISYCAL_DATA_ROOT
        track = 'PA'
    if is_eval:
        data_root = os.path.join('eval_data', data_root)
    track = track
    prefix = 'ASVspoof2019_{}'.format(track)
    protocols_fname = 'eval.trl' if is_eval else 'train.trn' if is_train else 'dev.trl'
    dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
    # print("dset_name:",dset_name)
    protocols_dir = os.path.join(data_root,
                                      '{}_cm_protocols/'.format(prefix))
    # print("protocols_fname:",protocols_fname)
    files_dir = os.path.join(data_root, '{}_{}'.format(
        prefix, dset_name), 'flac')


    protocols_fname = os.path.join(protocols_dir,
                                        'ASVspoof2019.{}.cm.{}.txt'.format(track, protocols_fname))


    def parse_line(line):
        tokens = line.strip().split(' ')
        return ASVFile(speaker_id=tokens[0],
                       file_name=tokens[1],
                       path=os.path.join(files_dir, tokens[1] + '.flac'),
                       sys_id=sysid_dict[tokens[3]],
                       key=int(tokens[4] == 'bonafide'))
    ASVFile = collections.namedtuple('ASVFile',
                                     ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])
    lines = open(protocols_fname).readlines()
    files_meta = list(map(parse_line, lines))
    # print("--------------------")
    # print("files_meta:", files_meta[1])
    # print("---------------------------")
    # files_meta = list(files_meta)



    return files_meta



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x




class Dataset_ASVspoof(Dataset):
    def __init__(self,files_meta):
        self.files_meta = files_meta
        self.data = list(map(self.read_file, self.files_meta))
        # print("data:",self.data[0])
        self.data_x, self.data_y, self.data_sysid = map(list, zip(*self.data))
        self.length = len(self.data_x)
        self.sysid_dict_inv =dict(zip(sysid_dict.values(),sysid_dict.keys()))


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        X_pad = pad(x, self.cut)
        x = Tensor(X_pad)
        y = self.data_y[idx]

        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        # data_x, sample_rate = sf.read(meta.path)
        data_x, sample_rate = librosa.load(meta.path, sr=16000)
        # print(meta.path)
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

from torch.utils.data import DataLoader
from transformer.otrans import encoder
if __name__ == '__main__':
    # is_logical=True, sample_size=None,is_train=True, is_eval=False, eval_part=0

    files_meta = genSpoof_list(is_logical=False, is_eval=False, is_train=True)
    train_set = Dataset_ASVspoof(files_meta)
    # train_loader = DataLoader(train_set, batch_size=12, shuffle=True, drop_last=True)
    # for batch_x, batch_y, batch_meta in train_loader:
    #     # batch_x : [12, 64600]
    #     model = encoder.SincConv(device='cuda',out_channels=20, kernel_size=1024, in_channels=1)
    #     transformer_outputs = model(batch_x)
    #     print(transformer_outputs.shape)
    # sysid_dict_inv =dict(zip(sysid_dict.values(),sysid_dict.keys()))
    # print(sysid_dict_inv)

                
                












