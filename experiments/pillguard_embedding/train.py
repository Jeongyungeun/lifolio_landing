"""어댑터 학습: 동결된 갤러리 프로토타입에 쿼리를 정렬하는 대조학습(InfoNCE).

- 갤러리(AI Hub) 임베딩은 gradient를 받지 않는다.
- negative는 기본적으로 seen 알약만 사용한다. unseen 알약을 negative로 쓰면
  unseen 일반화 평가가 낙관적으로 오염되므로 기본값에서 제외한다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common import l2norm


def pill_prototypes(g: np.ndarray, g_pid: np.ndarray, pills: np.ndarray) -> torch.Tensor:
    """알약별 갤러리 벡터 평균 -> 정규화. (P, 768) 상수 텐서."""
    protos = []
    for p in pills:
        v = g[g_pid == p].mean(axis=0)
        protos.append(v)
    return torch.as_tensor(l2norm(np.stack(protos)), dtype=torch.float32)


def train_adapter(
    model,
    q_train: np.ndarray,
    q_train_pid: np.ndarray,
    g: np.ndarray,
    g_pid: np.ndarray,
    seen_pills: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-4,
    temperature: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 0,
    verbose: bool = False,
):
    """쿼리 어댑터를 학습. 반환: 학습된 model (in-place 갱신)."""
    if sum(p.numel() for p in model.parameters()) == 0:
        return model                                   # Identity: 학습할 것 없음

    torch.manual_seed(seed)
    protos = pill_prototypes(g, g_pid, seen_pills)     # 동결 앵커
    protos = model.gallery(protos)                     # 'full' 모드면 제로 패딩
    protos = F.normalize(protos, dim=1)

    pid_to_idx = {p: i for i, p in enumerate(seen_pills)}
    keep = np.array([p in pid_to_idx for p in q_train_pid])
    X = torch.as_tensor(np.asarray(q_train[keep], dtype=np.float32))
    y = torch.as_tensor(np.array([pid_to_idx[p] for p in q_train_pid[keep]]), dtype=torch.long)
    if len(X) == 0:
        return model

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=gen)
        total = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i : i + batch_size]
            out = F.normalize(model(X[idx]), dim=1)
            logits = out @ protos.T / temperature       # protos는 gradient 없음
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if verbose and (ep + 1) % 20 == 0:
            print(f"    epoch {ep+1:3d}  loss {total/len(X):.4f}")
    return model
