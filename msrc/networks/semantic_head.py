import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


# ====================================================================
# 1. Semantic Conditioned Plane Guidance Module (PGM)
# ====================================================================
class PlaneGuidanceModule(nn.Module):
    """
    Lightweight PGM:
    输入:
        feat: NDDepth backbone features, shape B×D×H×W
        sem_emb: semantic embedding (projected), shape B×D×H×W
    输出:
        geo_feat: geometry-aware + semantic-aware feature
    """

    def __init__(self, feat_dim=256, K=8):
        super().__init__()
        self.K = K

        # 使用一个 1×1 卷积把语义 embedding 整合
        self.sem_fuse = nn.Conv2d(feat_dim, feat_dim, 1)

        # Query 生成器（从全局语义 embedding）
        self.query_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, K * feat_dim)
        )

        # 输出融合
        self.fuse = nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1)

    def forward(self, feat, sem_emb):
        """
        feat: B × D × H × W
        sem_emb: B × D × H × W
        """
        B, D, H, W = feat.shape

        # -----------------------------
        # 1. semantic feature 融入
        # -----------------------------
        sem_emb = self.sem_fuse(sem_emb)

        # -----------------------------
        # 2. 全局平均池化生成 Queries
        # -----------------------------
        g = F.adaptive_avg_pool2d(sem_emb, 1).view(B, D)  # B×D
        Q = self.query_mlp(g)                            # B×(K*D)
        Q = Q.view(B, self.K, D)                         # B×K×D

        # -----------------------------
        # 3. 计算简化版 Attention（dot-product）
        # -----------------------------
        feat_flat = feat.view(B, D, -1)                  # B×D×(HW)
        A = torch.einsum('bkd,bdv->bkv', Q, feat_flat)    # B×K×(HW)
        A = F.softmax(A, dim=1)
        A = A.view(B, self.K, H, W)

        # -----------------------------
        # 4. 计算 plane-guided feature
        # -----------------------------
        Q_expanded = Q.unsqueeze(-1).unsqueeze(-1)  # B×K×D×1×1
        G = (A.unsqueeze(2) * Q_expanded).sum(1)    # B×D×H×W

        # -----------------------------
        # 5. 融合 semantic + geometry
        # -----------------------------
        F_fused = torch.cat([feat, G], dim=1)       # B×2D×H×W
        out = self.fuse(F_fused)                   # B×D×H×W

        return out, A  # A 可用于可视化 (plane attention)
    
