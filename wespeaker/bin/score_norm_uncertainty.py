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

import logging
import os

import fire
import kaldiio
import numpy as np
from tqdm import tqdm

from wespeaker.utils.file_utils import read_table


def get_mean_std_uncertainty(
    emb,
    cohort,
    cohort_var,
    top_n,
    eps=1e-6
):
    """
    emb:        [B, D]
    cohort:     [C, D]
    cohort_var: [C, D]   (diagonal variance)
    """
    # ---- L2 normalize ----
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + eps)
    cohort = cohort / (np.linalg.norm(cohort, axis=1, keepdims=True) + eps)

    # ---- cosine similarity ----
    scores = emb @ cohort.T                 # [B, C]

    # ---- top-N selection ----
    idx = np.argpartition(scores, -top_n, axis=1)[:, -top_n:]
    scores_topn = np.take_along_axis(scores, idx, axis=1)   # [B, N]

    # ---- variance of score (ONLY top-N) ----
    emb_sq = emb ** 2                        # [B, D]
    cohort_var_topn = cohort_var[idx]        # [B, N, D]

    # Var(e^T c) = sum_d e_d^2 * Var(c_d)
    score_var_topn = np.sum(
        cohort_var_topn * emb_sq[:, None, :],
        axis=-1
    )                                        # [B, N]

    # weighted mean
    precision = 1.0 / (score_var_topn + eps)
    mean = np.sum(precision * scores_topn, axis=1) / np.sum(precision, axis=1)

    #  weighted variance
    var = np.sum(
        precision * (scores_topn - mean[:, None]) ** 2,
        axis=1
    ) / np.sum(precision, axis=1)
    std = np.sqrt(var + eps)


    return mean, std


  


def split_embedding(utt_list, emb_scp, mean_vec):
    embs = []
    utt2idx = {}
    utt2emb = {}
    for utt, emb in kaldiio.load_scp_sequential(emb_scp):
        emb = emb - mean_vec
        utt2emb[utt] = emb

    for utt in utt_list:
        embs.append(utt2emb[utt])
        utt2idx[utt] = len(embs) - 1

    return np.array(embs), utt2idx


def split_var(utt_list, var_scp):
    vars = []
    utt2idx = {}
    utt2var = {}
    for utt, var in kaldiio.load_scp_sequential(var_scp):
        utt2var[utt] = var

    for utt in utt_list:
        vars.append(utt2var[utt])
        utt2idx[utt] = len(vars) - 1

    return np.array(vars), utt2idx


def main(score_norm_method,
         top_n,
         trial_score_file,
         score_norm_file,
         cohort_emb_scp,
         cohort_var_scp,
         eval_emb_scp,
         eval_uncertainty_scp=None,
         mean_vec_path=None):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    # get embedding
    if not mean_vec_path:
        print("Do not do mean normalization for evaluation embeddings.")
        mean_vec = 0.0
    else:
        assert os.path.exists(
            mean_vec_path), "mean_vec file ({}) does not exist !!!".format(
                mean_vec_path)
        mean_vec = np.load(mean_vec_path)

    # get embedding
    logging.info('get embedding ...')
    
    enroll_list, test_list, _, _ = zip(*read_table(trial_score_file))
    enroll_list = sorted(list(set(enroll_list)))  # remove overlap and sort
    test_list = sorted(list(set(test_list)))
    enroll_emb, enroll_utt2idx = split_embedding(enroll_list, eval_emb_scp,
                                                 mean_vec)
    test_emb, test_utt2idx = split_embedding(test_list, eval_emb_scp, mean_vec)

    enroll_var, _ = split_var(enroll_list, eval_uncertainty_scp)
    test_var, _ = split_var(test_list, eval_uncertainty_scp)

    cohort_list, _ = zip(*read_table(cohort_emb_scp))
    cohort_emb, cohort_utt2idx_emb = split_embedding(cohort_list, cohort_emb_scp, mean_vec)


    cohort_var = None 
    cohort_list, _ = zip(*read_table(cohort_var_scp))
    cohort_var, cohort_utt2idx_var = split_var(cohort_list, cohort_var_scp)
    
    assert cohort_utt2idx_emb == cohort_utt2idx_var, "cohort var and emb do not match"

    logging.info("computing normed score ...")
    if score_norm_method == "asnorm":
        top_n = top_n
    elif score_norm_method == "snorm":
        top_n = cohort_emb.shape[0]
    else:
        raise ValueError(score_norm_method)
   

    enroll_mean, enroll_std = get_mean_std_uncertainty(enroll_emb, cohort_emb, cohort_var, top_n)
    test_mean, test_std = get_mean_std_uncertainty(test_emb, cohort_emb, cohort_var, top_n)

    # score norm
    with open(trial_score_file, 'r', encoding='utf-8') as fin:
        with open(score_norm_file, 'w', encoding='utf-8') as fout:
            lines = fin.readlines()
            for line in tqdm(lines):
                line = line.strip().split()
                enroll_idx = enroll_utt2idx[line[0]]
                test_idx = test_utt2idx[line[1]]
                score = float(line[2])
                enroll_var_c = enroll_var[enroll_idx]
                test_var_c = test_var[test_idx]

                
                alpha_1 = np.dot(enroll_emb[enroll_idx], enroll_emb[enroll_idx])/np.sum(enroll_emb[enroll_idx] / (1+enroll_var_c) *enroll_emb[enroll_idx])
                alpha_1 = alpha_1**0.5

                alpha_2 = np.dot(test_emb[test_idx], test_emb[test_idx])/np.sum(test_emb[test_idx] / (1+test_var_c) *test_emb[test_idx])
                alpha_2 = alpha_2**0.5
                normed_score = alpha_1*(score - enroll_mean[enroll_idx]) / enroll_std[enroll_idx] \
                    + alpha_2*(score - test_mean[test_idx]) / test_std[test_idx]
                
                # compute mag mean for score calibration
                enroll_mag = np.linalg.norm(enroll_emb[enroll_idx])
                test_mag = np.linalg.norm(test_emb[test_idx])



                enroll_mag_u = np.sqrt(
                    np.sum(enroll_emb[enroll_idx]**2 / (enroll_var_c )))
                test_mag_u = np.sqrt(
                    np.sum(test_emb[test_idx]**2 / (test_var_c)))
                
                # enroll_mag_u2 
                # enroll_mag_u = np.sqrt(
                #     np.sum(enroll_emb[enroll_idx]**2 / (enroll_var_c + 1)))
                # test_mag_u = np.sqrt(
                #     np.sum(test_emb[test_idx]**2 / (test_var_c + 1)))   

                fout.write(
                    '{} {} {:.5f} {} {:.4f} {:.4f} {:.4f} {:.4f}   \n'.format(
                        line[0], line[1], normed_score, line[3],  enroll_mag_u,test_mag_u,  enroll_mean[enroll_idx],
                        test_mean[test_idx]))


if __name__ == "__main__":
    fire.Fire(main)
