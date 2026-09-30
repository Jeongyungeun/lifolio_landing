"""공통 유틸: 정규화, 알약 단위 분할, 검색 지표.

설계 원칙
---------
- 갤러리(AI Hub) 임베딩과 DINOv2 가중치는 절대 변형하지 않는다.
- 어댑터는 '쿼리 측'에만 적용한다. 갤러리는 등록된 그대로 쓴다.
- 분할은 알약 종류(pill_id) 기준. 같은 알약 이미지가 train/test에 섞이지 않는다.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def l2norm(x: np.ndarray) -> np.ndarray:
    """행 단위 L2 정규화. 코사인 유사도를 내적으로 계산하기 위한 전처리."""
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, EPS)


def split_pills(pill_ids: np.ndarray, seen_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """알약 종류를 seen/unseen으로 분할. 이미지가 아니라 '종류'를 나눈다."""
    uniq = np.unique(pill_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_seen = int(round(len(uniq) * seen_frac))
    return np.sort(perm[:n_seen]), np.sort(perm[n_seen:])


def per_pill_scores(
    q: np.ndarray, g: np.ndarray, g_pid: np.ndarray, pill_order: np.ndarray
) -> np.ndarray:
    """쿼리 x 알약 점수 행렬. 한 알약에 여러 등록 벡터가 있으면 최대 유사도를 쓴다.

    q: (Nq, D) 정규화된 쿼리, g: (Ng, D) 정규화된 갤러리
    반환: (Nq, P) — pill_order 순서
    """
    sims = q @ g.T                                    # (Nq, Ng) 코사인
    out = np.full((q.shape[0], len(pill_order)), -np.inf)
    for j, pid in enumerate(pill_order):
        cols = np.flatnonzero(g_pid == pid)
        if cols.size:
            out[:, j] = sims[:, cols].max(axis=1)     # 다중 벡터 등록 = max 풀링
    return out


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUROC. 같은 알약 점수가 다른 알약 점수보다 높을 확률."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0
    # 동점 보정
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def evaluate(
    q: np.ndarray, q_pid: np.ndarray, g: np.ndarray, g_pid: np.ndarray
) -> dict:
    """검색 성능 + 유사도 분포를 한 번에 측정.

    주의: q, g는 이미 정규화되어 있어야 한다.
    """
    pill_order = np.unique(g_pid)
    scores = per_pill_scores(q, g, g_pid, pill_order)
    pid_to_col = {p: i for i, p in enumerate(pill_order)}

    # 정답 알약이 갤러리에 등록되어 있는 쿼리만 평가 대상
    keep = np.array([p in pid_to_col for p in q_pid])
    scores, q_pid = scores[keep], q_pid[keep]
    if scores.shape[0] == 0:
        return {"n_query": 0}
    true_col = np.array([pid_to_col[p] for p in q_pid])

    order = np.argsort(-scores, axis=1)               # 내림차순 랭킹
    rank_of_true = np.argmax(order == true_col[:, None], axis=1)

    rows = np.arange(scores.shape[0])
    same = scores[rows, true_col]                     # 같은 알약 유사도
    mask = np.ones_like(scores, dtype=bool)
    mask[rows, true_col] = False
    diff = scores[mask]                               # 다른 알약 유사도
    hardest = scores.copy()
    hardest[rows, true_col] = -np.inf
    hardest_neg = hardest.max(axis=1)                 # 가장 헷갈리는 오답

    s_same, s_diff = same.std(), diff.std()
    dprime = float((same.mean() - diff.mean()) / np.sqrt((s_same**2 + s_diff**2) / 2 + EPS))

    return {
        "n_query": int(scores.shape[0]),
        "n_pill_gallery": int(len(pill_order)),
        "top1": float((rank_of_true == 0).mean()),
        "top5": float((rank_of_true < 5).mean()),
        "same_mean": float(same.mean()),
        "same_std": float(s_same),
        "diff_mean": float(diff.mean()),
        "diff_std": float(s_diff),
        "d_prime": dprime,
        "auroc": auroc(same, diff),
        # 외형이 비슷한 알약과의 오인식: 최근접 오답이 정답을 이긴 비율
        "hard_neg_mean": float(hardest_neg.mean()),
        "margin_mean": float((same - hardest_neg).mean()),
        "hard_confusion_rate": float((hardest_neg >= same).mean()),
        "_same": same,
        "_diff": diff,
    }


def confusion_pairs(
    q: np.ndarray, q_pid: np.ndarray, g: np.ndarray, g_pid: np.ndarray, top: int = 10
) -> list[tuple]:
    """가장 빈번한 (정답, 오인식) 알약 쌍. 외형 유사 알약 진단용."""
    pill_order = np.unique(g_pid)
    scores = per_pill_scores(q, g, g_pid, pill_order)
    pid_to_col = {p: i for i, p in enumerate(pill_order)}
    keep = np.array([p in pid_to_col for p in q_pid])
    scores, q_pid = scores[keep], q_pid[keep]
    pred = pill_order[scores.argmax(axis=1)]
    wrong = pred != q_pid
    from collections import Counter
    c = Counter(zip(q_pid[wrong].tolist(), pred[wrong].tolist()))
    return c.most_common(top)
