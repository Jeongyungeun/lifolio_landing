#!/usr/bin/env python3
"""PillGuard 임베딩 검색 실험: 0~5단계를 한 번에 실행하고 기준선 대비 표를 출력.

사용법 (실데이터):
  python run_experiments.py \
      --gallery-emb  /path/aihub_emb.npy  --gallery-meta /path/aihub_meta.csv \
      --real-emb     /path/real_emb.npy   --real-meta    /path/real_meta.csv \
      --out ./results --seeds 0 1 2

사용법 (파이프라인 점검용 합성 데이터 — 실제 성능 아님):
  python run_experiments.py --synthetic --out ./results_smoke --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from common import l2norm, split_pills, evaluate, confusion_pairs
from datasets import load_real, make_synthetic, split_real_images
from models import build, apply_model
from train import train_adapter

METHODS = [
    ("baseline",          "1. 기준선 (어댑터 없음, 768)"),
    ("multivector",       "2. 다중 벡터 등록 (재등록)"),
    ("adapter_768",       "3. 768->768 residual 어댑터"),
    ("adapter_2048",      "4. 768->2048->768 어댑터"),
    ("structural_full",   "5a. 구조적 확장, 2048 전체"),
    ("structural_trunc",  "5b. 구조적 확장, 768 절단"),
    ("structural_reduce", "5c. 구조적 확장, 학습된 축소"),
]


def step0_summary(d: dict) -> dict:
    pills = np.unique(d["real_pid"])
    per = Counter(d["real_pid"].tolist())
    counts = np.array([per[p] for p in pills])
    g_per = Counter(d["gallery_pid"].tolist())
    only_real = sorted(set(d["real_pid"]) - set(d["gallery_pid"]))
    return {
        "source": d["source"],
        "dim": int(d["gallery"].shape[1]),
        "n_gallery_vec": int(len(d["gallery"])),
        "n_gallery_pill": int(len(g_per)),
        "n_real_img": int(len(d["real"])),
        "n_real_pill": int(len(pills)),
        "real_per_pill_min": int(counts.min()),
        "real_per_pill_median": float(np.median(counts)),
        "real_per_pill_max": int(counts.max()),
        "angle_dist": dict(Counter(d["real_angle"].tolist())),
        "pills_missing_from_gallery": only_real[:20],
        "n_pills_missing_from_gallery": len(only_real),
    }


def run_seed(d: dict, seed: int, seen_frac: float, register_k: int, epochs: int,
             verbose: bool) -> dict:
    seen, unseen = split_pills(d["real_pid"], seen_frac, seed)
    sp = split_real_images(d["real_pid"], seen, register_k, seed)

    G, Gp = d["gallery"], d["gallery_pid"]
    R, Rp = d["real"], d["real_pid"]

    is_seen_test = np.isin(Rp[sp["test"]], seen)
    q_seen, qp_seen = R[sp["test"]][is_seen_test], Rp[sp["test"]][is_seen_test]
    q_uns, qp_uns = R[sp["test"]][~is_seen_test], Rp[sp["test"]][~is_seen_test]

    out: dict = {"n_seen_pill": len(seen), "n_unseen_pill": len(unseen),
                 "n_test_seen": len(q_seen), "n_test_unseen": len(q_uns)}

    for name, _label in METHODS:
        if name == "multivector":
            # 갤러리에 실환경 등록 이미지를 추가. 어댑터 없음.
            g2 = np.concatenate([G, R[sp["register"]]])
            gp2 = np.concatenate([Gp, Rp[sp["register"]]])
            res_s = evaluate(l2norm(q_seen), qp_seen, l2norm(g2), gp2)
            res_u = evaluate(l2norm(q_uns), qp_uns, l2norm(g2), gp2)
        else:
            model = build(name)
            if name != "baseline":
                train_adapter(model, R[sp["train"]], Rp[sp["train"]], G, Gp, seen,
                              epochs=epochs, seed=seed, verbose=verbose)
            qs, gs = apply_model(model, q_seen, G)
            qu, gu = apply_model(model, q_uns, G)
            res_s = evaluate(l2norm(qs), qp_seen, l2norm(gs), Gp)
            res_u = evaluate(l2norm(qu), qp_uns, l2norm(gu), Gp)
        out[name] = {"seen": res_s, "unseen": res_u}

    # 5b 절단이 기준선과 수치적으로 동일한지 직접 검증
    base_s, tr_s = out["baseline"]["seen"], out["structural_trunc"]["seen"]
    out["_trunc_equals_baseline"] = {
        "top1_abs_diff": abs(base_s["top1"] - tr_s["top1"]),
        "same_mean_abs_diff": abs(base_s["same_mean"] - tr_s["same_mean"]),
        "max_same_elemwise_diff": float(np.max(np.abs(base_s["_same"] - tr_s["_same"]))),
    }
    # 5a 전체 사용이 기준선과 랭킹 동일한지 검증 (유사도 값은 축소됨)
    fu_s = out["structural_full"]["seen"]
    out["_full_vs_baseline"] = {
        "top1_abs_diff": abs(base_s["top1"] - fu_s["top1"]),
        "top5_abs_diff": abs(base_s["top5"] - fu_s["top5"]),
        "same_mean_baseline": base_s["same_mean"],
        "same_mean_full": fu_s["same_mean"],
    }
    out["_confusion_baseline_seen"] = confusion_pairs(
        l2norm(q_seen), qp_seen, l2norm(G), Gp, top=8)
    return out


def agg(runs: list[dict], method: str, split: str, key: str) -> tuple[float, float]:
    v = [r[method][split][key] for r in runs if r[method][split].get("n_query")]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


def table(runs: list[dict], split: str) -> str:
    base_t1 = agg(runs, "baseline", split, "top1")[0]
    base_t5 = agg(runs, "baseline", split, "top5")[0]
    lines = [
        f"### {split} 알약 (학습에 {'사용' if split=='seen' else '미사용'})",
        "",
        "| 방법 | top-1 | top-5 | Δtop-1 | 같은알약 평균 | 다른알약 평균 | d' | AUROC | 최근접오답 역전율 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, label in METHODS:
        t1, t1s = agg(runs, name, split, "top1")
        t5, t5s = agg(runs, name, split, "top5")
        sm = agg(runs, name, split, "same_mean")[0]
        dm = agg(runs, name, split, "diff_mean")[0]
        dp = agg(runs, name, split, "d_prime")[0]
        au = agg(runs, name, split, "auroc")[0]
        hc = agg(runs, name, split, "hard_confusion_rate")[0]
        d1 = t1 - base_t1
        lines.append(
            f"| {label} | {t1*100:.2f} ±{t1s*100:.2f} | {t5*100:.2f} ±{t5s*100:.2f} | "
            f"{d1*100:+.2f}p | {sm:.4f} | {dm:.4f} | {dp:.3f} | {au:.4f} | {hc*100:.2f}% |"
        )
    lines += ["", f"(기준선 top-1 {base_t1*100:.2f}%, top-5 {base_t5*100:.2f}%. "
                  f"±는 시드 {len(runs)}회 표준편차)"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="합성 데이터로 파이프라인만 점검 (실제 성능 아님)")
    ap.add_argument("--gallery-emb"); ap.add_argument("--gallery-meta")
    ap.add_argument("--real-emb");    ap.add_argument("--real-meta")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seen-frac", type=float, default=0.6)
    ap.add_argument("--register-k", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.synthetic:
        d = make_synthetic(seed=1234)
    else:
        missing = [k for k in ("gallery_emb", "gallery_meta", "real_emb", "real_meta")
                   if getattr(a, k) is None]
        if missing:
            ap.error(f"실데이터 경로 누락: {missing} (또는 --synthetic 사용)")
        d = load_real(a.gallery_emb, a.gallery_meta, a.real_emb, a.real_meta)

    s0 = step0_summary(d)
    print("=" * 78)
    print("0. 데이터 요약")
    print("=" * 78)
    for k, v in s0.items():
        print(f"  {k:32s}: {v}")
    if d["source"] == "synthetic":
        print("\n  *** 합성 데이터입니다. 아래 수치는 파이프라인 점검용이며 "
              "실제 알약 성능이 아닙니다. ***")
    if s0["n_pills_missing_from_gallery"]:
        print(f"\n  경고: 실환경에만 있고 갤러리에 없는 알약 "
              f"{s0['n_pills_missing_from_gallery']}종은 평가에서 제외됩니다.")

    runs = []
    for s in a.seeds:
        print(f"\n[seed {s}] 실행 중 ...")
        runs.append(run_seed(d, s, a.seen_frac, a.register_k, a.epochs, a.verbose))

    print("\n" + "=" * 78)
    print("1~5. 단계별 결과 (기준선 대비)")
    print("=" * 78)
    for split in ("seen", "unseen"):
        print("\n" + table(runs, split))

    print("\n" + "=" * 78)
    print("구조적 검증 (가설 5a/5b)")
    print("=" * 78)
    te = [r["_trunc_equals_baseline"] for r in runs]
    fu = [r["_full_vs_baseline"] for r in runs]
    print(f"  5b 절단 vs 기준선  top-1 최대 차이     : "
          f"{max(x['top1_abs_diff'] for x in te):.3e}")
    print(f"  5b 절단 vs 기준선  유사도 원소별 최대 차이: "
          f"{max(x['max_same_elemwise_diff'] for x in te):.3e}")
    print(f"  5a 전체 vs 기준선  top-1 최대 차이     : "
          f"{max(x['top1_abs_diff'] for x in fu):.3e}")
    print(f"  5a 전체 vs 기준선  같은알약 평균 유사도  : "
          f"{fu[0]['same_mean_baseline']:.4f} -> {fu[0]['same_mean_full']:.4f} (값은 축소, 순위 불변)")

    print("\n  기준선 상위 오인식 쌍 (seen, seed 0):")
    for (t, p), c in runs[0]["_confusion_baseline_seen"]:
        print(f"    정답 {t} -> 오인식 {p} : {c}회")

    # 산출물 저장 (기존 파일 덮어쓰지 않도록 out 디렉터리에만)
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if not k.startswith("_same")
                    and not k.startswith("_diff")}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, list):
            return [strip(x) for x in o]
        return o

    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"data": s0, "config": vars(a),
                   "runs": [strip(r) for r in runs]}, f, ensure_ascii=False, indent=2)

    hist = {}
    for name, _ in METHODS:
        for split in ("seen", "unseen"):
            r = runs[0][name][split]
            if r.get("n_query"):
                hist[f"{name}__{split}__same"] = r["_same"]
                hist[f"{name}__{split}__diff"] = r["_diff"]
    np.savez_compressed(os.path.join(a.out, "similarity_distributions.npz"), **hist)

    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("# PillGuard 임베딩 실험 결과\n\n## 0. 데이터 요약\n\n")
        for k, v in s0.items():
            f.write(f"- `{k}`: {v}\n")
        if d["source"] == "synthetic":
            f.write("\n> **합성 데이터**입니다. 파이프라인 점검용이며 실제 성능이 아닙니다.\n")
        f.write("\n## 2. 단계별 결과\n\n")
        for split in ("seen", "unseen"):
            f.write(table(runs, split) + "\n\n")
    print(f"\n산출물 저장: {a.out}/ (summary.json, report.md, similarity_distributions.npz)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (name, label) in zip(axes.ravel(),
                                     [("baseline", "기준선"), ("multivector", "다중벡터"),
                                      ("adapter_2048", "768-2048-768"),
                                      ("structural_reduce", "구조적+학습축소")]):
            r = runs[0][name]["unseen"]
            if not r.get("n_query"):
                continue
            ax.hist(r["_diff"], bins=80, alpha=.55, density=True, label="different pill")
            ax.hist(r["_same"], bins=80, alpha=.55, density=True, label="same pill")
            ax.set_title(f"{name} (unseen)"); ax.set_xlabel("cosine"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "similarity_hist.png"), dpi=130)
        print(f"히스토그램 저장: {a.out}/similarity_hist.png")
    except ImportError:
        print("matplotlib 미설치 — 히스토그램은 similarity_distributions.npz 로 저장됨")


if __name__ == "__main__":
    main()
