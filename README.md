
# $\mathcal{U}^3$-xi 

This is the official implementation of our paper "$\mathcal{U}^3$-xi: Pushing the Boundaries of Speaker Recognition via Incorporating Uncertainty."

## Experiments: 
Our experiments are located in the [`examples/voxceleb/v2`](examples/voxceleb/v2) directory.

- configs: 
    - [ecapa_tdnn_u_cube.yaml](examples/voxceleb/v2/conf/ecapa_tdnn_u_cube.yaml)
    - [resnet_u_cube.yaml](examples/voxceleb/v2/conf/resnet_u_cube.yaml)
    - [redimnet_u_cube.yaml](examples/voxceleb/v2/conf/redimnet_u_cube.yaml)
- models:
    - [ECAPA-TDNN model](examples/voxceleb/v2/wespeaker/models/ecapa_tdnn.py#L230-L244)
    - [ResNet](wespeaker/models/resnet.py#195)
    - [RedimNet](wespeaker/models/redimnet.py#L863)
- pooling_layer: [U_Cube_XI](examples/voxceleb/v2/wespeaker/models/pooling_layers.py#L422)
    - Multi view selfattention: [MVA](examples/voxceleb/v2/wespeaker/models/MHA.py)
- projection: [ArcMarginProduct_uncertainty](examples/voxceleb/v2/wespeaker/models/projections.py#L37)
- changes in executor: [executor.py](examples/voxceleb/v2/wespeaker/utils/executor.py)
- changes in train.py: comment out code 'jit', since it is not compatablile with MVA [train.py](examples/voxceleb/v2/wespeaker/bin/train.py#L153)

## Results: 
| Model | Params | Flops | LM | AS-Norm | QMF | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------|:------:|:------|:--:|:-------:|:---:|:------------:|:------------:|:------------:|
| ECAPA_TDNN_GLOB_c512-ASTP-emb192  | 6.19M | 1.04G | × | × | × | 1.069 | 1.209 | 2.310 |
|                                   |       |       | × | √ | × | 0.957 | 1.128 | 2.105 |
|                                   |       |       | √ | × | × | 0.878 | 1.072 | 2.007 |
|                                   |       |       | √ | √ | × | 0.782 | 1.005 | 1.824 |
| ECAPA_TDNN_GLOB_c1024-ASTP-emb192 | 14.65M | 2.65G | × | × | × | 0.856 | 1.072 | 2.059 |
|                                   |        |       | × | √ | × | 0.808 | 0.990 | 1.874 |
|                                   |        |       | √ | × | × | 0.798 | 0.993 | 1.883 |
|                                   |        |       | √ | √ | × | 0.728 | 0.929 | 1.721 |
|                                   |        |       | √ | √ | √ | 0.707 | 0.894 | 1.615 |
| ResNet34-TSTP-emb256 | 6.63M | 4.55G | × | × | × | 0.867 | 1.049 | 1.959 |
|                      |       |       | × | √ | × | 0.787 | 0.964 | 1.726 |
|                      |       |       | × | √ | √ | 0.718 | 0.911 | 1.606 |
|                      |       |       | √ | × | × | 0.797 | 0.937 | 1.695 |
|                      |       |       | √ | √ | × | 0.723 | 0.867 | 1.532 |
|                      |       |       | √ | √ | √ | 0.659 | 0.821 | 1.437 |
| XI_VEC_ECAPA_TDNN_c512       | 5.9M | 1.04G      | x | x | × | 0.995 | 1.130 | 2.169 |
|                  |       |       | × | √ | × | 0.883 | 1.056 | 1.976 |
|                  |       |       | √ | × | × | 0.909 | 1.000 | 1.855 |
|                  |       |       | √ | √ | × | 0.787 | 0.930 | 1.693 |
| **ECAPA_TDNN_c512_u_cube_xi(ours)**| 6.7M | 1.20G      | x | x | x | 0.782 | 1.016 | 1.888 | 
| **ResNet34_u_cube_xi(ours)**       | 7.9M |            | x | x | x | 0.867 | 0.868 | 1.641 |
| **ReDimNet-B2(ours)**              | 5.5M |            | x | x | x | 0.606 | 0.779 | 1.494 |
|                                    |      |            | √ | x | x | 0.489 | 0.698 | 1.311 |
|                                    |      |            | √ | √ | x | 0.399 | 0.638 | 1.170 |





# WeSpeaker

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.8%7C3.9-brightgreen)](https://github.com/wenet-e2e/wespeaker)

[**Roadmap**](ROADMAP.md)
| [**Docs**](http://wenet.org.cn/wespeaker)
| [**Paper**](https://arxiv.org/abs/2210.17016)
| [**Runtime**](https://github.com/wenet-e2e/wespeaker/tree/master/runtime)
| [**Pretrained Models**](docs/pretrained.md)
| [**Huggingface Demo**](https://huggingface.co/spaces/wenet/wespeaker_demo)
| [**Modelscope Demo**](https://www.modelscope.cn/studios/wenet/Speaker_Verification_in_WeSpeaker/summary)


WeSpeaker mainly focuses on [**speaker embedding learning**](https://wsstriving.github.io/talk/ncmmsc_slides_shuai.pdf), with application to the speaker verification task. We support
online feature extraction or loading pre-extracted features in kaldi-format.

## Installation

### Install python package
``` sh
pip install git+https://github.com/wenet-e2e/wespeaker.git
```
**Command-line usage** (use `-h` for parameters):

``` sh
$ wespeaker --task embedding --audio_file audio.wav --output_file embedding.txt
$ wespeaker --task embedding_kaldi --wav_scp wav.scp --output_file /path/to/embedding
$ wespeaker --task similarity --audio_file audio.wav --audio_file2 audio2.wav
$ wespeaker --task diarization --audio_file audio.wav
```

**Python programming usage**:

``` python
import wespeaker

model = wespeaker.load_model('chinese')
embedding = model.extract_embedding('audio.wav')
utt_names, embeddings = model.extract_embedding_list('wav.scp')
similarity = model.compute_similarity('audio1.wav', 'audio2.wav')
diar_result = model.diarize('audio.wav')
```

You can set the environment variable `WESPEAKER_HOME` to specify the path of downloaded pre-trained models. By default it will be `$HOME/.wespeaker`.

Please refer to [python usage](docs/python_package.md) for more command line and python programming usage.

### Install for development & deployment
* Clone this repo
``` sh
git clone https://github.com/wenet-e2e/wespeaker.git
```

* Create conda env: pytorch version >= 1.12.1 is recommended !!!
``` sh
conda create -n wespeaker python=3.9
conda activate wespeaker
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
pre-commit install  # for clean and tidy code
```

## 🔥 News
* 2025.12.05: Add support for the [w2v-bert2 model](https://www.arxiv.org/pdf/2510.04213), see [#439](https://github.com/wenet-e2e/wespeaker/pull/439) and [#441](https://github.com/wenet-e2e/wespeaker/pull/441).
* 2025.02.23: Add support for the Xi-vector, see [#404](https://github.com/wenet-e2e/wespeaker/pull/404).
* 2024.09.03: Support the SimAM_ResNet and the model pretrained on VoxBlink2, check [Pretrained Models](docs/pretrained.md) for the pretrained model, [VoxCeleb Recipe](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2) for the super performance, and [python usage](docs/python_package.md) for the command line usage!
* 2024.08.30: We support whisper_encoder based frontend and propose the [Whisper-PMFA](https://arxiv.org/pdf/2408.15585) framework, check [#356](https://github.com/wenet-e2e/wespeaker/pull/356).
* 2024.08.20: Update diarization recipe for VoxConverse dataset by leveraging umap dimensionality reduction and hdbscan clustering, see [#347](https://github.com/wenet-e2e/wespeaker/pull/347) and [#352](https://github.com/wenet-e2e/wespeaker/pull/352).
* 2024.08.18: Support using ssl pre-trained models as the frontend. The [WavLM recipe](https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/run_wavlm.sh) is also provided, see [#344](https://github.com/wenet-e2e/wespeaker/pull/344).
* 2024.05.15: Add support for [quality-aware score calibration](https://arxiv.org/pdf/2211.00815), see [#320](https://github.com/wenet-e2e/wespeaker/pull/320).
* 2024.04.25: Add support for the gemini-dfresnet model, see [#291](https://github.com/wenet-e2e/wespeaker/pull/291).
* 2024.04.23: Support MNN inference engine in runtime, see [#310](https://github.com/wenet-e2e/wespeaker/pull/310).
* 2024.04.02: Release [Wespeaker document](http://wenet.org.cn/wespeaker) with detailed model-training tutorials, introduction of various runtime platforms, etc.
* 2024.03.04: Support the [eres2net-cn-common-200k](https://www.modelscope.cn/models/iic/speech_eres2net_sv_zh-cn_16k-common/summary) and [campplus-cn-common-200k](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) of damo [#281](https://github.com/wenet-e2e/wespeaker/pull/281), check [python usage](https://github.com/wenet-e2e/wespeaker/blob/master/docs/python_package.md) for details.
* 2024.02.05: Support the ERes2Net [#272](https://github.com/wenet-e2e/wespeaker/pull/272) and Res2Net [#273](https://github.com/wenet-e2e/wespeaker/pull/273) models.
* 2023.11.13: Support CLI usage of wespeaker, check [python usage](https://github.com/wenet-e2e/wespeaker/blob/master/docs/python_package.md) for details.
* 2023.07.18: Support the kaldi-compatible PLDA and unsupervised adaptation, see [#186](https://github.com/wenet-e2e/wespeaker/pull/186).
* 2023.07.14: Support the [NIST SRE16 recipe](https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation-2016), see [#177](https://github.com/wenet-e2e/wespeaker/pull/177).

## Recipes

* [VoxCeleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb): Speaker Verification recipe on the [VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
    * 🔥 UPDATE 2024.05.15: We support score calibration for Voxceleb and achieve better performance!
    * 🔥 UPDATE 2023.07.10: We support self-supervised learning recipe on Voxceleb! Achieving **2.627%** (ECAPA_TDNN_GLOB_c1024) EER on vox1-O-clean test set without any labels.
    * 🔥 UPDATE 2022.10.31: We support deep r-vector up to the 293-layer version! Achieving **0.447%/0.043** EER/mindcf on vox1-O-clean test set
    * 🔥 UPDATE 2022.07.19: We apply the same setups as the CNCeleb recipe, and obtain SOTA performance considering the open-source systems
      - EER/minDCF on vox1-O-clean test set are **0.723%/0.069** (ResNet34) and **0.728%/0.099** (ECAPA_TDNN_GLOB_c1024), after LM fine-tuning and AS-Norm
* [CNCeleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/cnceleb/v2): Speaker Verification recipe on the [CnCeleb dataset](http://cnceleb.org/)
    * 🔥 UPDATE 2024.05.16: We support score calibration for Cnceleb and achieve better EER.
    * 🔥 UPDATE 2022.10.31: 221-layer ResNet achieves **5.655%/0.330**  EER/minDCF
    * 🔥 UPDATE 2022.07.12: We migrate the winner system of CNSRC 2022 [report](https://aishell-cnsrc.oss-cn-hangzhou.aliyuncs.com/T082.pdf) [slides](https://aishell-cnsrc.oss-cn-hangzhou.aliyuncs.com/T082-ZhengyangChen.pdf)
      - EER/minDCF reduction from 8.426%/0.487 to **6.492%/0.354** after large margin fine-tuning and AS-Norm
* [NIST SRE16](https://github.com/wenet-e2e/wespeaker/tree/master/examples/sre/v2): Speaker Verification recipe for the [2016 NIST Speaker Recognition Evaluation Plan](https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation-2016). Similar recipe can be found in [Kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/sre16).
   * 🔥 UPDATE 2023.07.14: We support NIST SRE16 recipe. After PLDA adaptation, we achieved 6.608%, 10.01%, and 2.974% EER on trial Pooled, Tagalog, and Cantonese, respectively.
* [VoxConverse](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxconverse): Diarization recipe on the [VoxConverse dataset](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)

## Discussion

For Chinese users, you can scan the QR code on the left to follow our offical account of `WeNet Community`.
We also created a WeChat group for better discussion and quicker response. Please scan the QR code on the right to join the chat group.
| <img src="https://github.com/wenet-e2e/wenet-contributors/blob/main/wenet_official.jpeg" width="250px"> | <img src="https://github.com/wenet-e2e/wenet-contributors/blob/main/wespeaker/wangshuai.jpg" width="250px"> |
| ---- | ---- |

## Citations
If you find wespeaker useful, please cite it as
```bibtex
@article{wang2024advancing,
  title={Advancing speaker embedding learning: Wespeaker toolkit for research and production},
  author={Wang, Shuai and Chen, Zhengyang and Han, Bing and Wang, Hongji and Liang, Chengdong and Zhang, Binbin and Xiang, Xu and Ding, Wen and Rohdin, Johan and Silnova, Anna and others},
  journal={Speech Communication},
  volume={162},
  pages={103104},
  year={2024},
  publisher={Elsevier}
}

@inproceedings{wang2023wespeaker,
  title={Wespeaker: A research and production oriented speaker embedding learning toolkit},
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
## Looking for contributors

If you are interested to contribute, feel free to contact @wsstriving or @robin1001
