# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os

import fire
import kaldiio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wespeaker.dataset.dataset import Dataset
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug
from wespeaker.frontend import *
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import parse_config_or_kwargs, validate_path


def extract(config='conf/config.yaml', **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs['model_path']
    embed_ark = configs['embed_ark']
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 1)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    test_conf = copy.deepcopy(configs['dataset_args'])
    # model: frontend (optional) => speaker model
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    frontend_type = test_conf.get('frontend', 'fbank')
    if frontend_type != 'fbank':
        frontend_args = frontend_type + "_args"
        print('Initializing frontend model (this could take some time) ...')
        frontend = frontend_class_dict[frontend_type](
            **test_conf[frontend_args], sample_rate=test_conf['resample_rate'])
        model.add_module("frontend", frontend)
    print('Loading checkpoint ...')
    load_checkpoint(model, model_path)
    print('Finished !!! Start extracting ...')
    device = torch.device("cuda")
    model.to(device).eval()

    # test_configs
    # test_conf = copy.deepcopy(configs['dataset_args'])
    test_conf['speed_perturb'] = False
    if 'fbank_args' in test_conf:
        test_conf['fbank_args']['dither'] = 0.0
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['aug_prob'] = configs.get('aug_prob', 0.0)
    test_conf['filter'] = False

    dataset = Dataset(configs['data_type'],
                      configs['data_list'],
                      test_conf,
                      spk2id_dict={},
                      whole_utt=(batch_size == 1),
                      reverb_lmdb_file=configs.get('reverb_data', None),
                      noise_lmdb_file=configs.get('noise_data', None),
                      repeat_dataset=False)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)

    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"
    
    variance_ark = embed_ark[:-4]+'_variance.ark'
    validate_path(variance_ark)
    variance_scp = variance_ark[:-3] + "scp"

    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," + embed_scp) as embed_writer, \
            kaldiio.WriteHelper('ark,scp:' + variance_ark + "," + variance_scp) as var_writer:

            for _, batch in tqdm(enumerate(dataloader)):
                utts = batch['key']

                if frontend_type == 'fbank':
                    features = batch['feat'].float().to(device)  # (B,T,F)
                else:
                    wavs = batch['wav'].squeeze(1).float().to(device)
                    wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                        wavs.shape[0]).to(device)
                    features, _ = model.frontend(wavs, wavs_len)

                # CMVN
                if test_conf.get('cmvn', True):
                    features = apply_cmvn(features,
                                        **test_conf.get('cmvn_args', {}))

                # SpecAug (normally False for extract)
                if test_conf.get('spec_aug', False):
                    features = spec_aug(features, **test_conf['spec_aug_args'])

                B, T, F = features.shape
                chunk_size = 10000
                num_chunks = (T + chunk_size - 1) // chunk_size

                embed_list = []
                var_list = []
                chunk_weights = []

                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, T)

                    chunk = features[:, start:end, :]

                    outputs = model(chunk)

                    assert len(outputs) == 3, "Model output must be _, var, embed"

                    if isinstance(outputs, tuple):
                        var, embed = outputs[1], outputs[-1]
                    else:
                        raise RuntimeError("Model must output (variance, embedding) tuple")

                    embed_list.append(embed)   # (B,D)
                    var_list.append(var)       # (B,D)
                    chunk_weights.append((end - start) / T)

                # ---------- Embedding 加权平均 ----------
                chunk_weights = torch.tensor(chunk_weights, device=device)  # (N,)

                embed_stack = torch.stack(embed_list, dim=0)  # (N,B,D)
                embeds = (chunk_weights[:, None, None] * embed_stack).sum(dim=0)  # (B,D)

                # ---------- Variance 精度融合 ----------
                var_stack = torch.stack(var_list, dim=0)  # (N,B,D)

                eps = 1e-6
                var_stack = torch.clamp(var_stack, min=eps)

                precision = chunk_weights[:, None, None] / var_stack
                precision_sum = precision.sum(dim=0)  # (B,D)

                variance = 1.0 / precision_sum

                embeds = embeds.cpu().numpy()
                variance = variance.cpu().numpy()

                for i, utt in enumerate(utts):
                    embed_writer(utt, embeds[i])
                    var_writer(utt, variance[i])



if __name__ == '__main__':
    fire.Fire(extract)
