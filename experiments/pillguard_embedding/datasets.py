"""데이터 로딩 + 알약 단위 누출 없는 분할.

실데이터 계약 (CSV는 pandas 없이 읽는다)
----------------------------------------
gallery_emb.npy  : (Ng, 768) float — AI Hub 임베딩. **절대 수정하지 않는다.**
gallery_meta.csv : 헤더에 pill_id 열 필수. 행 순서 = gallery_emb 행 순서.
real_emb.npy     : (Nr, 768) float — 실환경 임베딩.
real_meta.csv    : 헤더에 pill_id 열 필수. angle 열이 있으면 각도 분포를 보고한다.

임베딩이 아직 없으면 embed_images.py 로 먼저 생성한다.
"""
from __future__ import annotations

import csv
import numpy as np


def _read_meta(path: str) -> dict[str, list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: 행이 없습니다")
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    if "pill_id" not in cols:
        raise ValueError(f"{path}: 'pill_id' 열이 필요합니다. 발견된 열: {list(cols)}")
    return cols


def load_real(gallery_emb, gallery_meta, real_emb, real_meta) -> dict:
    ge = np.load(gallery_emb)
    re_ = np.load(real_emb)
    gm, rm = _read_meta(gallery_meta), _read_meta(real_meta)
    if len(gm["pill_id"]) != len(ge):
        raise ValueError(f"갤러리 행 수 불일치: emb {len(ge)} vs meta {len(gm['pill_id'])}")
    if len(rm["pill_id"]) != len(re_):
        raise ValueError(f"실환경 행 수 불일치: emb {len(re_)} vs meta {len(rm['pill_id'])}")
    return {
        "gallery": ge.astype(np.float64),
        "gallery_pid": np.array(gm["pill_id"]),
        "real": re_.astype(np.float64),
        "real_pid": np.array(rm["pill_id"]),
        "real_angle": np.array(rm.get("angle", ["unknown"] * len(re_))),
        "source": "real",
    }


def make_synthetic(
    seed: int = 0,
    n_pill: int = 120,
    dim: int = 768,
    n_gallery_view: int = 4,
    n_real: int = 12,
    view_noise: float = 0.35,
    domain_shift: float = 1.1,
    real_noise: float = 0.8,
    lookalike_groups: int = 60,
    lookalike_sim: float = 0.99,
    n_factor: int = 24,
) -> dict:
    """합성 데이터. **파이프라인 점검 전용이며 실제 알약 성능과 무관하다.**

    실제 문제 구조를 모사한다:
      - AI Hub 뷰: 정면 계열 작은 변형 (angle 0/90/180/270)
      - 실환경: 공통 도메인 이동(카메라·조명) + 큰 변형, angle 에 diagonal 포함
      - lookalike 그룹: 외형이 거의 같은 알약 쌍 (오인식률 측정용)
    """
    rng = np.random.default_rng(seed)
    sc = 1.0 / np.sqrt(dim)   # 노이즈 노름을 sigma 수준으로 맞추는 스케일
    # 실제 알약은 서로 비슷하다(흰색 원형 정제 등). 무작위 768차원 벡터는
    # 서로 거의 직교해서 비현실적으로 쉬우므로, 저차원 잠재 요인에서 생성한다.
    A = rng.normal(size=(dim, n_factor)) / np.sqrt(dim)
    base = rng.normal(size=(n_pill, n_factor)) @ A.T
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    # 외형 유사 알약 쌍 생성
    for gi in range(min(lookalike_groups, n_pill // 2)):
        a, b = 2 * gi, 2 * gi + 1
        base[b] = lookalike_sim * base[a] + np.sqrt(1 - lookalike_sim**2) * base[b]
        base[b] /= np.linalg.norm(base[b])

    shift = rng.normal(size=dim)                  # 모든 알약에 공통인 도메인 이동
    shift /= np.linalg.norm(shift)

    g_angles = ["0", "90", "180", "270"]
    g, g_pid = [], []
    for p in range(n_pill):
        for v in range(n_gallery_view):
            g.append(base[p] + view_noise * sc * rng.normal(size=dim))
            g_pid.append(f"pill_{p:04d}")

    r, r_pid, r_ang = [], [], []
    for p in range(n_pill):
        for v in range(n_real):
            ang = "diagonal" if v % 3 == 0 else g_angles[v % len(g_angles)]
            extra = domain_shift * (1.4 if ang == "diagonal" else 1.0)
            r.append(base[p] + extra * shift + real_noise * sc * rng.normal(size=dim))
            r_pid.append(f"pill_{p:04d}")
            r_ang.append(ang)

    return {
        "gallery": np.stack(g), "gallery_pid": np.array(g_pid),
        "real": np.stack(r), "real_pid": np.array(r_pid),
        "real_angle": np.array(r_ang), "source": "synthetic",
    }


def split_real_images(
    real_pid: np.ndarray, seen_pills: np.ndarray, register_k: int, seed: int
) -> dict[str, np.ndarray]:
    """알약별로 실환경 이미지를 register / train / test 로 나눈다.

    - register: 2단계 다중 벡터 등록에 쓰는 이미지 (모든 알약)
    - train   : 어댑터 학습용 (seen 알약만)
    - test    : 쿼리 (모든 알약). train/register 와 절대 겹치지 않는다.
    """
    rng = np.random.default_rng(seed)
    seen = set(seen_pills.tolist())
    reg, tr, te = [], [], []
    for pid in np.unique(real_pid):
        idx = np.flatnonzero(real_pid == pid)
        rng.shuffle(idx)
        k = min(register_k, max(0, len(idx) - 2))      # 쿼리를 최소 2장 남긴다
        reg.extend(idx[:k])
        rest = idx[k:]
        if pid in seen:
            n_tr = len(rest) // 2
            tr.extend(rest[:n_tr])
            te.extend(rest[n_tr:])
        else:
            te.extend(rest)
    return {
        "register": np.array(sorted(reg), dtype=int),
        "train": np.array(sorted(tr), dtype=int),
        "test": np.array(sorted(te), dtype=int),
    }
