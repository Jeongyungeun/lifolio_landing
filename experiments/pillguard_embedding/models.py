"""쿼리 측 어댑터들. 백본(DINOv2)과 갤러리 임베딩은 전부 동결.

어댑터는 (query_transform, gallery_transform) 쌍을 제공한다.
갤러리 변환이 identity가 아닌 경우는 5-(a) 제로 패딩 비교뿐이고,
그조차 갤러리 '값'을 바꾸지 않고 차원만 맞춘다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Identity(nn.Module):
    """1단계 기준선: 어댑터 없음."""
    out_dim = 768

    def forward(self, x):
        return x

    def gallery(self, g):
        return g


class ResidualAdapter(nn.Module):
    """3·4단계: 768 -> hidden -> 768 residual MLP.

    hidden=768  -> 3단계 (768->768 어댑터)
    hidden=2048 -> 4단계 (768->2048->768 어댑터)
    residual 구조라 학습 초기에 기준선과 동일한 출력에서 출발한다.
    """
    out_dim = 768

    def __init__(self, dim: int = 768, hidden: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        nn.init.zeros_(self.net[-1].weight)   # 초기 출력 = 기준선
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)

    def gallery(self, g):
        return g


class StructuralExpand(nn.Module):
    """5단계: 사용자 가설의 구조적 구현.

    e(x) = [ x (백본 출력 그대로, 동결·복사) ; g(x) (학습되는 extra_dim 차원) ]

    readout:
      'full'   : 2048 전체 사용. 갤러리는 768 -> 2048 제로 패딩. (5-a)
      'trunc'  : 앞 768차원만 잘라서 사용. (5-b)
      'reduce' : 학습된 2048 -> 768 선형 축소. (5-c)
    """

    def __init__(self, dim: int = 768, extra_dim: int = 1280, hidden: int = 2048,
                 readout: str = "reduce"):
        super().__init__()
        assert readout in {"full", "trunc", "reduce"}
        self.dim, self.extra_dim, self.readout = dim, extra_dim, readout
        self.expand = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, extra_dim)
        )
        self.reduce = nn.Linear(dim + extra_dim, dim, bias=False) if readout == "reduce" else None
        if self.reduce is not None:
            # 초기값을 [I | 0] 으로 두어 학습 시작점을 기준선과 일치시킨다.
            with torch.no_grad():
                self.reduce.weight.zero_()
                self.reduce.weight[:, :dim] = torch.eye(dim)
        self.out_dim = {"full": dim + extra_dim, "trunc": dim, "reduce": dim}[readout]

    def expanded(self, x):
        """앞 768은 x를 그대로 붙인다 (detach 불필요: x는 이미 상수 입력)."""
        return torch.cat([x, self.expand(x)], dim=1)

    def forward(self, x):
        e = self.expanded(x)
        if self.readout == "full":
            return e
        if self.readout == "trunc":
            return e[:, : self.dim]
        return self.reduce(e)

    def gallery(self, g):
        if self.readout == "full":                    # 차원만 맞춤. 값은 불변.
            pad = torch.zeros(g.shape[0], self.extra_dim, dtype=g.dtype, device=g.device)
            return torch.cat([g, pad], dim=1)
        return g


def build(name: str) -> nn.Module:
    """실험 이름 -> 모델. run_experiments.py 에서 사용."""
    return {
        "baseline":            lambda: Identity(),
        "adapter_768":         lambda: ResidualAdapter(hidden=768),
        "adapter_2048":        lambda: ResidualAdapter(hidden=2048),
        "structural_full":     lambda: StructuralExpand(readout="full"),
        "structural_trunc":    lambda: StructuralExpand(readout="trunc"),
        "structural_reduce":   lambda: StructuralExpand(readout="reduce"),
    }[name]()


@torch.no_grad()
def apply_model(model: nn.Module, q: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """numpy 임베딩에 어댑터를 적용해 (쿼리, 갤러리) 출력을 돌려준다."""
    model.eval()
    qt = torch.as_tensor(np.asarray(q, dtype=np.float32))
    gt = torch.as_tensor(np.asarray(g, dtype=np.float32))
    return model(qt).numpy().astype(np.float64), model.gallery(gt).numpy().astype(np.float64)
