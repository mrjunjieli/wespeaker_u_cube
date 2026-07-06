
💡This repository contains the implementations of three related papers on uncertainty-aware speaker recognition.

# 1. $\mathcal{U}^3$-xi  Pushing the Boundaries of Speaker Recognition via Incorporating Uncertainty


<p align="center">
  <b>Junjie Li</b>, Kong Aik Lee <br>
  <i>The Hong Kong Polytechnic University</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.15719">
    <img src="https://img.shields.io/badge/arXiv-2308.08143-b31b1b.svg"/>
  </a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=mrjunjieli.wespeaker_u_cube"/>
  <img src="https://img.shields.io/github/stars/mrjunjieli/wespeaker_u_cube?style=social"/>
</p>
<p align="center">
  Email: <a href="mailto:junjie98.li@connect.polyu.hk">junjie98.li@connect.polyu.hk</a>
</p>


## ✨Key Highlights: 
This is the official implementation of our paper "$\mathcal{U}^3$-xi: Pushing the Boundaries of Speaker Recognition via Incorporating Uncertainty."

We introduce three main modifications:
- Uncertainty Estimation Module: multi-view self-attention (MVA) 
- Global Uncertainty Supervision by incoporating uncertainty into scale 
- Uncertainty-aware cosine scoring 

## 🚀Experiments: 
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

## 📊Results: 
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

## 💥Pretrained Models:
There models are re-trained, hence results infered from these are slightly different from our paper.  

- [Voxceleb2_ECAPA-TDNN-512_u_cube](https://drive.google.com/file/d/1bMqisoKLM4eKkCAW7k7Ya-M3tNERQVh0/view?usp=drive_link)
- [Voxceleb2_ResNet34_u_cube](https://drive.google.com/file/d/1Lf2CnB6ReIMyq7w06ucORmMF41GzdH76/view?usp=drive_link)
- [Voxceleb2_ReDimNet-B2_u_cube](https://drive.google.com/file/d/1S2pWyI3HIRGgdTh686J08oAWK3UmqAT-/view?usp=drive_link)
- [Voxceleb2_Xi-ECAPA-TDNN-512](https://drive.google.com/file/d/1Aw1qPe_oQkjDnFjYunCq5-KXIga3J1r6/view?usp=sharing)



# 2. Uncertainty Score Normalization and  Calibration for Speaker Verification

<p align="center">
  <a href="https://arxiv.org/abs/2601.15719">
    <img src="https://img.shields.io/badge/arXiv-2308.08143-b31b1b.svg"/>
  </a>
</p>

Below are introductions about 'Uncertainty Score Normalization and  Calibration for Speaker Verification'

We introduce three main modifications:
- Uncertainty-aware cosine score 
- UAS-Norm: Uncertainty-Aware AS-Norm
- UQMFs: Uncertainty-Aware Quality Measure Functions

## 🚀Experiments: 
Our experiments are located in the [`examples/voxceleb/v2`](examples/voxceleb/v2) directory.

- extract uncertainty
  - [local/extract_vox_uncertainty.sh](examples/voxceleb/v2/local/extract_vox_uncertainty.sh)
  - [tools/extract_embedding_uncertainty.sh](examples/voxceleb/v2/tools/extract_embedding_uncertainty.sh)
  - [wespeaker/bin/extract_uncertainty.py](wespeaker/bin/extract_uncertainty.py)

- uncertainty-aware cosine score
  - [local/score_uncertainty.sh](examples/voxceleb/v2/local/score_uncertainty.sh)
  - [wespeaker/bin/score_uncertainty.py](wespeaker/bin/score_uncertainty.py)

- uncertainty-aware AS-norm (uncertainty-aware cosine score + uncertainty-aware AS-norm)
  - [local/score_norm_uncertainty.sh](examples/voxceleb/v2/local/score_norm_uncertainty.sh)
  - [variance_mean.py](examples/voxceleb/v2/tools/variance_mean.py)
  - [wespeaker/bin/score_norm_uncertainty.py](wespeaker/bin/score_norm_uncertainty.py)

- uncertianty-aware QMFs (uncertainty-aware cosine score + uncertainty-aware AS-norm + uncertianty-aware QMFs)
  - [local/score_calibration_uncertainty.sh](local/score_calibration_uncertainty.sh)
  - [wespeaker/bin/score_norm_forvox2.py](wespeaker/bin/score_norm_forvox2.py)
  - [wespeaker/bin/score_calibration_uncertainty.py](wespeaker/bin/score_calibration_uncertain.py)

  
# 3. Towards Robust Uncertainty-Aware Speaker Modeling

In this paper, we propose two new methods:
- Inter- and Intra-Speaker-Aware Uncertainty Softmax
- Uncertainty-Calibrated Domain Adaptation (UCDA)

This repro contains only the code of first method. 

## 🚀Experiments: 
- Inter- and Intra-Speaker-Aware Uncertainty Softmax (UAAM Softmax Inter-intra):[wespeaker/models/projections.py:ArcMarginProduct_uncertainty_inter_intra](wespeaker/models/projections.py#L500) 
- Inter- and Intra-Speaker-Aware Uncertainty Softmax (USphereFace2 Inter-intra):[wespeaker/models/projections.py:SphereFace2_uncertainty_Arcguide](wespeaker/models/projections.py#L193) 
- Inter- and Intra-Speaker-Aware Uncertainty Softmax (UAM Softmax Inter-intra):[wespeaker/models/projections.py:AddMarginProduct_uncertainty_inter_intra](wespeaker/models/projections.py#L829) 


## Results: 

<table style="border-collapse: collapse; border-top: 3px solid #000; border-bottom: 3px solid #000;">
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2"># Param.</th>
      <th rowspan="2">Loss</th>
      <th rowspan="2">Uncertainty-aware cosine score</th>
      <th colspan="7">In-domain</th>
      <th colspan="3">Cross-domain</th>
    </tr>
    <tr>
      <th colspan="2">Vox1-O</th>
      <th colspan="2">Vox1-E</th>
      <th colspan="2">Vox1-H</th>
      <th rowspan="2">RI (%)</th>
      <th colspan="2">CNCeleb</th>
      <th rowspan="2">RI (%)</th>
    </tr>
    <tr>
      <th colspan="4"></th>
      <th>EER</th>
      <th>minDCF</th>
      <th>EER</th>
      <th>minDCF</th>
      <th>EER</th>
      <th>minDCF</th>
      <th>EER</th>
      <th>minDCF</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-top: 3px solid #000;">
      <td>ECAPA512</td>
      <td>6.19 M</td>
      <td>AAM-Softmax</td>
      <td>No</td>
      <td>1.069</td>
      <td>0.122</td>
      <td>1.209</td>
      <td>0.136</td>
      <td>2.310</td>
      <td>0.226</td>
      <td>Benchmark</td>
      <td>15.314</td>
      <td>0.633</td>
      <td>Benchmark</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td rowspan="2">ECAPA512+$\mathcal{U}^3$-xi</td>
      <td rowspan="2">6.69 M</td>
      <td rowspan="2">UAAM-Softmax</td>
      <td>No</td>
      <td>0.856</td>
      <td>0.109</td>
      <td>1.064</td>
      <td>0.121</td>
      <td>1.982</td>
      <td>0.195</td>
      <td>13.57</td>
      <td>13.706</td>
      <td>0.608</td>
      <td>7.23</td>
    </tr>
    <tr>
      <td>Yes</td>
      <td>0.782</td>
      <td>0.100</td>
      <td>1.016</td>
      <td>0.115</td>
      <td>1.888</td>
      <td>0.187</td>
      <td>18.64</td>
      <td>10.271</td>
      <td>1.000</td>
      <td>-12.52</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td rowspan="2">ECAPA512+$\mathcal{U}^3$-xi</td>
      <td rowspan="2">6.69 M</td>
      <td rowspan="2">UAAM-Softmax inter-intra</td>
      <td>No</td>
      <td>0.936</td>
      <td>0.102</td>
      <td>1.050</td>
      <td>0.122</td>
      <td>1.978</td>
      <td>0.195</td>
      <td>13.40</td>
      <td>13.974</td>
      <td>0.581</td>
      <td>8.48</td>
    </tr>
    <tr>
      <td>Yes</td>
      <td>0.840</td>
      <td>0.086</td>
      <td><strong>0.965</strong></td>
      <td>0.110</td>
      <td>1.833</td>
      <td>0.189</td>
      <td>21.22</td>
      <td>10.781</td>
      <td>0.835</td>
      <td>-1.16</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td>ECAPA512</td>
      <td>6.19 M</td>
      <td>AM-Softmax</td>
      <td>No</td>
      <td>1.005</td>
      <td>0.107</td>
      <td>1.206</td>
      <td>0.133</td>
      <td>2.254</td>
      <td>0.221</td>
      <td>Benchmark</td>
      <td>14.162</td>
      <td>0.611</td>
      <td>Benchmark</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td rowspan="2">ECAPA512+$\mathcal{U}^3$-xi</td>
      <td rowspan="2">6.69 M</td>
      <td rowspan="2">UAM-Softmax inter-intra</td>
      <td>No</td>
      <td>0.888</td>
      <td>0.099</td>
      <td>1.076</td>
      <td>0.119</td>
      <td>1.973</td>
      <td>0.186</td>
      <td>11.46</td>
      <td>12.436</td>
      <td>0.553</td>
      <td>10.84</td>
    </tr>
    <tr>
      <td>Yes</td>
      <td>0.808</td>
      <td><strong>0.084</strong></td>
      <td>0.991</td>
      <td>0.109</td>
      <td>1.794</td>
      <td><strong>0.178</strong></td>
      <td>19.46</td>
      <td><strong>9.411</strong></td>
      <td>1.000</td>
      <td>-15.03</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td>ECAPA512</td>
      <td>6.19 M</td>
      <td>SphereFace2</td>
      <td>No</td>
      <td>0.963</td>
      <td>0.108</td>
      <td>1.121</td>
      <td>0.125</td>
      <td>1.967</td>
      <td>0.199</td>
      <td>Benchmark</td>
      <td>12.582</td>
      <td>0.573</td>
      <td>Benchmark</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td rowspan="2">ECAPA512+$\mathcal{U}^3$-xi</td>
      <td rowspan="2">6.69 M</td>
      <td rowspan="2">USphereFace2 inter-intra</td>
      <td>No</td>
      <td>0.856</td>
      <td>0.104</td>
      <td>1.035</td>
      <td>0.119</td>
      <td>1.918</td>
      <td>0.196</td>
      <td>5.21</td>
      <td>12.265</td>
      <td><strong>0.550</strong></td>
      <td>3.27</td>
    </tr>
    <tr>
      <td>Yes</td>
      <td><strong>0.739</strong></td>
      <td>0.102</td>
      <td><strong>0.965</strong></td>
      <td><strong>0.108</strong></td>
      <td><strong>1.771</strong></td>
      <td><strong>0.178</strong></td>
      <td>12.81</td>
      <td>10.560</td>
      <td>0.624</td>
      <td>3.59</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td>ResNet34</td>
      <td>6.63 M</td>
      <td>AAM-Softmax</td>
      <td>No</td>
      <td><strong>0.867</strong></td>
      <td>0.091</td>
      <td>1.049</td>
      <td>0.121</td>
      <td>1.960</td>
      <td>0.192</td>
      <td>Benchmark</td>
      <td>11.090</td>
      <td><strong>0.488</strong></td>
      <td>Benchmark</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td rowspan="2">ResNet34+$\mathcal{U}^3$-xi</td>
      <td rowspan="2">7.92 M</td>
      <td rowspan="2">UAAM-Softmax</td>
      <td>No</td>
      <td>0.888</td>
      <td>0.085</td>
      <td>0.900</td>
      <td>0.099</td>
      <td>1.712</td>
      <td>0.175</td>
      <td>9.68</td>
      <td>11.732</td>
      <td>0.513</td>
      <td>-5.46</td>
    </tr>
    <tr>
      <td>Yes</td>
      <td><strong>0.867</strong></td>
      <td><strong>0.078</strong></td>
      <td><strong>0.868</strong></td>
      <td><strong>0.095</strong></td>
      <td><strong>1.641</strong></td>
      <td><strong>0.172</strong></td>
      <td>13.29</td>
      <td><strong>10.082</strong></td>
      <td>0.541</td>
      <td>-0.89</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td rowspan="2">ResNet34+$\mathcal{U}^3$-xi</td>
      <td rowspan="2">7.92 M</td>
      <td rowspan="2">USphereFace2 inter-intra</td>
      <td>No</td>
      <td>1.483</td>
      <td>0.148</td>
      <td>1.451</td>
      <td>0.156</td>
      <td>2.112</td>
      <td>0.206</td>
      <td>-36.00</td>
      <td>11.441</td>
      <td>0.512</td>
      <td>-4.04</td>
    </tr>
    <tr>
      <td>Yes</td>
      <td>1.340</td>
      <td>0.156</td>
      <td>1.357</td>
      <td>0.150</td>
      <td>1.986</td>
      <td>0.193</td>
      <td>-30.19</td>
      <td>10.949</td>
      <td>0.499</td>
      <td>-0.49</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td>ReDimNet-B2</td>
      <td>5.46 M</td>
      <td>AAM-Softmax</td>
      <td>No</td>
      <td>0.782</td>
      <td>0.064</td>
      <td>0.907</td>
      <td>0.097</td>
      <td>1.667</td>
      <td>0.162</td>
      <td>Benchmark</td>
      <td>12.385</td>
      <td>0.552</td>
      <td>Benchmark</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td rowspan="2">ReDimNet-B2+$\mathcal{U}^3$-xi</td>
      <td rowspan="2">5.46 M</td>
      <td rowspan="2">UAAM-Softmax</td>
      <td>No</td>
      <td>0.649</td>
      <td>0.073</td>
      <td>0.801</td>
      <td>0.089</td>
      <td>1.532</td>
      <td>0.153</td>
      <td>6.09</td>
      <td>13.464</td>
      <td>0.552</td>
      <td>-4.36</td>
    </tr>
    <tr>
      <td>Yes</td>
      <td><strong>0.606</strong></td>
      <td>0.065</td>
      <td>0.779</td>
      <td>0.091</td>
      <td>1.494</td>
      <td>0.157</td>
      <td>9.12</td>
      <td><strong>9.479</strong></td>
      <td>1.000</td>
      <td>-28.85</td>
    </tr>
    <tr style="border-top: 3px solid #000;">
      <td rowspan="2">ReDimNet-B2+$\mathcal{U}^3$-xi</td>
      <td rowspan="2">4.89 M</td>
      <td rowspan="2">USphereFace2 inter-intra</td>
      <td>No</td>
      <td>0.622</td>
      <td>0.052</td>
      <td>0.776</td>
      <td>0.085</td>
      <td>1.440</td>
      <td>0.146</td>
      <td>14.92</td>
      <td>12.081</td>
      <td>0.515</td>
      <td>4.58</td>
    </tr>
    <tr style="border-bottom: 3px solid #000;">
      <td>Yes</td>
      <td>0.622</td>
      <td><strong>0.051</strong></td>
      <td><strong>0.774</strong></td>
      <td><strong>0.084</strong></td>
      <td><strong>1.433</strong></td>
      <td><strong>0.145</strong></td>
      <td>15.56</td>
      <td>11.899</td>
      <td><strong>0.506</strong></td>
      <td>6.13</td>
    </tr>
  </tbody>
</table>



## Pre-trained Models:
- [Voxceleb2_ECAPA-TDNN-512_AM](https://drive.google.com/file/d/1m4rp2WOJbMitRLNYJK41D4r5HJczPmBJ/view?usp=sharing)
- [Voxceleb2_ECAPA-TDNN-512_u_cube_AM+Inter-Intra](https://drive.google.com/file/d/1YJlAsVtFavhsv12BwpvVRVHRUsRYGNx8/view?usp=sharing)
- [Voxceleb2_ECAPA-TDNN-512_u_cube_AAM+Inter-Intra](https://drive.google.com/file/d/17vrgCwdH9mROTV2_tePEdXjhlw3qZNjq/view?usp=sharing)

--- 
---
---
Below are original wespeaker's readme

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
