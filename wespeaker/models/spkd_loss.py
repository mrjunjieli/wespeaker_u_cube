# Copyright (c) 2024 Zhengyang Chen (chenzhengyang117@gmail.com)
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

"""Similarity-Preserving Knowledge Distillation Loss.

Reference:
    "Similarity-Preserving Knowledge Distillation", ICCV 2019.
    Ported from RecXi project.
"""

import torch
import torch.nn as nn


class SPKDLoss(nn.Module):
    """Similarity-Preserving Knowledge Distillation Loss.

    Encourages the student to preserve the pairwise similarity structure
    of the teacher's representations.
    """

    def __init__(self, reduction="batchmean"):
        super().__init__()
        self.reduction = reduction

    def _matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return torch.nn.functional.normalize(torch.matmul(z, z.t()), 1)

    def forward(self, teacher_outputs, student_outputs):
        """
        Args:
            teacher_outputs: (B, D) teacher representation
            student_outputs: (B, D) student representation
        Returns:
            SPKD loss scalar
        """
        batch_size = teacher_outputs.shape[0]
        g_t = self._matmul_and_normalize(teacher_outputs)
        g_s = self._matmul_and_normalize(student_outputs)
        spkd_loss = torch.norm(g_t - g_s) ** 2
        if self.reduction == 'batchmean':
            return spkd_loss / (batch_size ** 2)
        return spkd_loss
