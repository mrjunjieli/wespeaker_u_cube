#!/bin/bash

# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

score_norm_method="asnorm"  # asnorm/snorm
cohort_set=vox2_dev
top_n=100
batch_size=512
exp_dir=
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
data=data

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "compute mean xvector"
  python tools/vector_mean.py \
    --spk2utt ${data}/${cohort_set}/spk2utt \
    --xvector_scp $exp_dir/embeddings/${cohort_set}/xvector.scp \
    --spk_xvector_ark $exp_dir/embeddings/${cohort_set}/spk_xvector.ark
  
  echo 'compute mean variance'
  python tools/variance_mean.py \
    --spk2utt ${data}/${cohort_set}/spk2utt \
    --xvector_scp $exp_dir/embeddings/${cohort_set}/xvector_variance.scp \
    --spk_xvector_ark $exp_dir/embeddings/${cohort_set}/spk_xvector_variance.ark

fi

output_name=${cohort_set}_${score_norm_method}
[ "${score_norm_method}" == "asnorm" ] && output_name=${output_name}${top_n}
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "compute norm score"
  for x in $trials; do
    python wespeaker/bin/score_norm_uncertainty.py \
      --score_norm_method $score_norm_method \
      --top_n $top_n \
      --trial_score_file $exp_dir/scores_uncertainty/${x}.score \
      --score_norm_file $exp_dir/scores_uncertainty/${output_name}_${x}.score \
      --cohort_emb_scp ${exp_dir}/embeddings/${cohort_set}/spk_xvector.scp \
      --cohort_var_scp ${exp_dir}/embeddings/${cohort_set}/spk_xvector_variance.scp \
      --eval_emb_scp ${exp_dir}/embeddings/vox1/xvector.scp \
      --eval_uncertainty_scp ${exp_dir}/embeddings/vox1/xvector_variance.scp \
      --mean_vec_path ${exp_dir}/embeddings/vox2_dev/mean_vec.npy \
      --batch_size $batch_size
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "compute metrics"
  for x in ${trials}; do
    scores_dir=${exp_dir}/scores_uncertainty
    python wespeaker/bin/compute_metrics.py \
      --p_target 0.01 \
      --c_fa 1 \
      --c_miss 1 \
      ${scores_dir}/${output_name}_${x}.score \
      2>&1 | tee -a ${scores_dir}/vox1_${score_norm_method}${top_n}_result

    python wespeaker/bin/compute_det.py \
      ${scores_dir}/${output_name}_${x}.score
  done
fi
