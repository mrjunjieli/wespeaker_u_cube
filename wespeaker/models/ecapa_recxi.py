# Copyright (c) 2026 Junjie LI (mrjunjieli@gmail.com)
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

"""ECAPA-TDNN with RecXi three-layer Bayesian disentanglement."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import wespeaker.models.MHA as MHA
from wespeaker.models.ecapa_tdnn import Conv1dReluBn, SE_Res2Block


class PrecisionEstimatorAttn(nn.Module):
    """Auxiliary network: estimates per-frame log-precision using MHA attention."""

    def __init__(self, input_dim, hidden_size=256):
        super().__init__()
        self.lin1_relu_bn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size))
        self.encoder_layer = MHA.MultiViewTransformerEncoderLayer(
            embed_dim=hidden_size,
            num_heads=8,
            ff_hidden=hidden_size * 4,
            dropout=0.2)
        self.lin2 = nn.Conv1d(hidden_size, input_dim,
                               kernel_size=1, stride=1, bias=True)
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, x):
        """x: (B, D, T) -> logprec: (B, D, T)"""
        temp = self.lin1_relu_bn(x)
        temp = self.encoder_layer(temp.permute(0, 2, 1)).permute(0, 2, 1)
        logprec = self.softplus(self.lin2(temp))
        logprec = 2.0 * torch.log(logprec)
        logprec = logprec.clamp(min=-15.0, max=15.0)
        return logprec


class RecXiParallel(nn.Module):
    """Parallel three-layer Bayesian decomposition with frame-level G (v4).

    Key difference from v3: no utt-level G. Frame-level G is used for both
    Layer 2 observation transform AND Layer 3 residual. Precision is adjusted
    at frame level for the G transformation.
    """

    def __init__(self, input_dim, hidden_size=128, n_rec_xi_set=16):
        super().__init__()
        self.input_dim = input_dim

        # Priors for three layers
        self.prior_mean_Xi_f = nn.Parameter(
            torch.zeros(1, input_dim, 1), requires_grad=True)
        self.log_prior_prec_Xi_f = nn.Parameter(
            torch.zeros(1, input_dim, 1), requires_grad=True)
        self.prior_mean_Rec = nn.Parameter(
            torch.zeros(1, input_dim, 1), requires_grad=True)
        self.log_prior_prec_Rec = nn.Parameter(
            torch.zeros(1, input_dim, 1), requires_grad=True)
        self.prior_mean_Xi_b = nn.Parameter(
            torch.zeros(1, input_dim, 1), requires_grad=True)
        self.log_prior_prec_Xi_b = nn.Parameter(
            torch.zeros(1, input_dim, 1), requires_grad=True)

        # G mechanism: mixture of basis functions (frame-level capable)
        self.G_container_0 = nn.Parameter(
            torch.ones(1, 1, input_dim, 1), requires_grad=False)
        self.G_container_ = nn.Parameter(
            torch.randn(1, n_rec_xi_set - 1, input_dim, 1), requires_grad=True)
        self.G_weight = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, n_rec_xi_set, kernel_size=1, stride=1),
            nn.Softmax(dim=1))

        self.softmax = nn.Softmax(dim=2)
        self.scale2 = 1e-10
        self.scale_up = 1e12

    def _compute_G(self, features):
        """Compute G mechanism. Works for (B,D,T) and (B,D,1)."""
        G_container = torch.abs(
            torch.cat((self.G_container_0, self.G_container_), dim=1))
        weight_G = self.G_weight(features).unsqueeze(2)
        G = torch.sum(G_container * weight_G, dim=1) + self.scale2
        G = torch.nan_to_num(G, nan=1.0, posinf=1.0, neginf=1.0)
        G = torch.clamp(G, max=self.scale_up)
        return G

    def _bayesian_pool(self, observations, logprec_obs, prior_mu, prior_logprec):
        """Parallel Bayesian pooling over all frames."""
        all_logprec = torch.cat((logprec_obs, prior_logprec), dim=2)
        weights = self.softmax(all_logprec)
        posterior_mean = torch.sum(
            torch.cat((observations, prior_mu), dim=2) * weights, dim=2)
        log_post_prec = torch.logsumexp(all_logprec, dim=2)
        posterior_var = 1.0 / torch.clamp(
            torch.exp(log_post_prec), min=1e-12)
        return posterior_mean, posterior_var, log_post_prec

    def forward(self, inputs, logprec):
        """
        Args:
            inputs: (B, D, T) frame-level features
            logprec: (B, D, T) per-frame log-precision from auxiliary network
        Returns:
            student, teacher, var_Xi_f, var_Rec, var_Xi_b
        """
        batch_size = inputs.shape[0]

        # ---- Expand priors ----
        prior_mu_Xi_f = self.prior_mean_Xi_f.expand(batch_size, -1, -1)
        prior_logprec_Xi_f = self.log_prior_prec_Xi_f.expand(batch_size, -1, -1)
        prior_mu_Rec = self.prior_mean_Rec.expand(batch_size, -1, -1)
        prior_logprec_Rec = self.log_prior_prec_Rec.expand(batch_size, -1, -1)
        prior_mu_Xi_b = self.prior_mean_Xi_b.expand(batch_size, -1, -1)
        prior_logprec_Xi_b = self.log_prior_prec_Xi_b.expand(batch_size, -1, -1)

        # ---- Layer 1: Xi_f (static speaker) ----
        #   obs=inputs, logprec=logprec
        mu_Xi_f, var_Xi_f, logprec_post_Xi_f = self._bayesian_pool(
            inputs, logprec, prior_mu_Xi_f, prior_logprec_Xi_f)


        # ---- Layer 2: Rec (dynamic content) ----
        #   obs=G_content, logprec=logprec_G_content

        # ---- Content residual & G mechanism ----
        content_residual = inputs - mu_Xi_f.unsqueeze(2)     # (B, D, T)
        # Precision of content residual = prec(inputs - Xi_f)
        logprec_post_Xi_f_exp = logprec_post_Xi_f.unsqueeze(2)  # (B, D, 1)
        # Variance addition: var_content = 1/prec_obs + 1/prec_Xi_f, then prec = 1/var_content
        logprec_content = -torch.logaddexp(-logprec, -logprec_post_Xi_f_exp)

        G = self._compute_G(content_residual)                # (B, D, T)
        G_content = G * content_residual                     # (B, D, T) G-scaled content
        # G transform: prec(G * residual) = prec(residual) / G^2
        logprec_G_content = logprec_content - 2 * torch.log(G)  # (B, D, T)        
        mu_Rec, var_Rec, _ = self._bayesian_pool(
            G_content, logprec_G_content, prior_mu_Rec, prior_logprec_Rec)

        # ---- Layer 3: Xi_b (refined speaker) ----
        #   obs=inputs - G_content, logprec derived from obs precision
        speaker_obs = inputs - G_content                     # (B, D, T)
        logprec_speaker_obs = -torch.logaddexp(-logprec, -logprec_G_content)

        mu_Xi_b, var_Xi_b, _ = self._bayesian_pool(
            speaker_obs, logprec_speaker_obs, prior_mu_Xi_b, prior_logprec_Xi_b)

        student = mu_Xi_f - mu_Rec
        teacher = mu_Xi_b

        return student, teacher, var_Xi_f, var_Rec, var_Xi_b


class ECAPA_TDNN_RecXi(nn.Module):
    """ECAPA-TDNN with RecXi three-layer Bayesian disentanglement."""

    def __init__(self,
                 channels=512,
                 feat_dim=80,
                 embed_dim=192,
                 hidden_size=256,
                 n_rec_xi_set=16,
                 emb_bn=False,
                 rec_xi_hidden=128):
        super().__init__()

        # ---- ECAPA-TDNN backbone ----
        self.layer1 = Conv1dReluBn(feat_dim, channels,
                                    kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1,
                                    padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1,
                                    padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1,
                                    padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        out_channels = cat_channels
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)

        self.prec_estimator = PrecisionEstimatorAttn(out_channels, hidden_size)
        self.rec_xi = RecXiParallel(out_channels,
                                     hidden_size=rec_xi_hidden,
                                     n_rec_xi_set=n_rec_xi_set)

        combined_dim = out_channels * 2
        self.bn = nn.BatchNorm1d(combined_dim)
        self.linear = nn.Linear(combined_dim, embed_dim)
        self.emb_bn = emb_bn
        if emb_bn:
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = nn.Identity()

    def _frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = self.conv(out)
        return out, out4

    def forward(self, x):
        out, out4 = self._frame_level_feat(x)
        out = F.relu(out)

        logprec = self.prec_estimator(out)
        student, teacher, var_Xi_f, var_Rec, var_Xi_b = self.rec_xi(out, logprec)

        combined = torch.cat((student, teacher), dim=1)

        student_var = var_Xi_f + var_Rec
        combined_var = torch.cat((student_var, var_Xi_b), dim=1)

        combined = self.bn(combined)
        combined_var = combined_var / (self.bn.running_var + self.bn.eps)
        combined_var = self.bn.weight ** 2 * combined_var

        embedding = self.linear(combined)
        var_diag = combined_var @ (self.linear.weight ** 2).T

        if self.emb_bn:
            embedding = self.bn2(embedding)
            var_diag = var_diag / (self.bn2.running_var + self.bn2.eps)
            var_diag = self.bn2.weight ** 2 * var_diag

        if self.training:
            return out4, var_diag, embedding, teacher, student
        else:
            return out4, var_diag, embedding


def ECAPA_TDNN_RecXi_c512(feat_dim, embed_dim, hidden_size=256,
                          n_rec_xi_set=16, emb_bn=False,
                          rec_xi_hidden=128):
    return ECAPA_TDNN_RecXi(channels=512,
                            feat_dim=feat_dim,
                            embed_dim=embed_dim,
                            hidden_size=hidden_size,
                            n_rec_xi_set=n_rec_xi_set,
                            emb_bn=emb_bn,
                            rec_xi_hidden=rec_xi_hidden)


if __name__ == '__main__':
    x = torch.zeros(2, 200, 80)
    model = ECAPA_TDNN_RecXi_c512(feat_dim=80, embed_dim=192)
    model.train()
    out = model(x)
    print(f"Training output: {len(out)} tensors")
    print(f"  out4:      {out[0].shape}")
    print(f"  var_diag:  {out[1].shape}")
    print(f"  embedding: {out[2].shape}")
    print(f"  teacher:   {out[3].shape}")
    print(f"  student:   {out[4].shape}")

    model.eval()
    out = model(x)
    print(f"\nInference output: {len(out)} tensors")
    print(f"  out4:      {out[0].shape}")
    print(f"  var_diag:  {out[1].shape}")
    print(f"  embedding: {out[2].shape}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {num_params / 1e6:.2f} M")
