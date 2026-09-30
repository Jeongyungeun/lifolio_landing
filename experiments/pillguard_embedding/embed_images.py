#!/usr/bin/env python3
"""실환경 이미지 -> DINOv2 임베딩 (.npy + meta.csv) 생성.

*** 중요 ***
AI Hub 백본 임베딩을 만들 때 쓴 설정과 **정확히 동일**해야 한다. 하나라도 다르면
기준선 비교가 무의미해진다. 반드시 대조할 항목:
  - 모델 변종 (dinov2_vits14 / vitb14 / vitl14 ... 768차원이면 vitb14 계열)
  - CLS 토큰 vs patch 평균 풀링 vs 둘의 concat
  - 입력 해상도, 리사이즈/크롭 방식, 보간법
  - 정규화 mean/std
  - L2 정규화를 임베딩 저장 시점에 했는지 여부

검증 방법: AI Hub 이미지 몇 장을 이 스크립트로 다시 임베딩해서 기존 갤러리
벡터와 코사인 유사도가 1.0에 매우 가까운지 확인하라 (--verify-against 옵션).

사용법:
  python embed_images.py --image-root /path/real_images --out-prefix ./real \
      --labels labels.csv --model dinov2_vitb14 --size 224
  # labels.csv: image_path,pill_id[,angle]  (image_path는 --image-root 기준 상대경로)
  # 미지정 시 디렉터리 이름을 pill_id 로 사용: <root>/<pill_id>/<img>.jpg
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch


def collect(image_root: str, labels: str | None) -> list[tuple[str, str, str]]:
    """(상대경로, pill_id, angle) 목록."""
    if labels:
        with open(labels, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows or "image_path" not in rows[0] or "pill_id" not in rows[0]:
            raise ValueError("labels.csv 에 image_path, pill_id 열이 필요합니다")
        return [(r["image_path"], r["pill_id"], r.get("angle", "unknown")) for r in rows]
    out = []
    for pid in sorted(os.listdir(image_root)):
        sub = os.path.join(image_root, pid)
        if not os.path.isdir(sub):
            continue
        for fn in sorted(os.listdir(sub)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                out.append((os.path.join(pid, fn), pid, "unknown"))
    if not out:
        raise ValueError(f"{image_root} 에서 이미지를 찾지 못했습니다")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-prefix", required=True, help="예: ./real -> real_emb.npy, real_meta.csv")
    ap.add_argument("--labels")
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pool", choices=["cls", "mean", "cls_mean"], default="cls")
    ap.add_argument("--l2", action="store_true", help="저장 전 L2 정규화 (갤러리와 동일하게 맞춰라)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                    else ("cuda" if torch.cuda.is_available() else "cpu"))
    a = ap.parse_args()

    from PIL import Image
    import torchvision.transforms as T

    if os.path.exists(a.out_prefix + "_emb.npy"):
        raise SystemExit(f"{a.out_prefix}_emb.npy 가 이미 있습니다. 덮어쓰지 않습니다.")

    items = collect(a.image_root, a.labels)
    print(f"이미지 {len(items)}장, 알약 {len(set(i[1] for i in items))}종, device={a.device}")

    model = torch.hub.load("facebookresearch/dinov2", a.model).to(a.device).eval()
    tf = T.Compose([
        T.Resize(a.size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(a.size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    embs = []
    with torch.no_grad():
        for i in range(0, len(items), a.batch):
            chunk = items[i : i + a.batch]
            batch = torch.stack([
                tf(Image.open(os.path.join(a.image_root, rel)).convert("RGB"))
                for rel, _, _ in chunk
            ]).to(a.device)
            out = model.forward_features(batch)
            cls, patch = out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(dim=1)
            v = {"cls": cls, "mean": patch,
                 "cls_mean": torch.cat([cls, patch], dim=1)}[a.pool]
            embs.append(v.float().cpu().numpy())
            print(f"  {min(i+a.batch, len(items))}/{len(items)}", end="\r")

    E = np.concatenate(embs)
    if a.l2:
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    np.save(a.out_prefix + "_emb.npy", E)
    with open(a.out_prefix + "_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "pill_id", "angle"])
        w.writerows(items)
    print(f"\n저장: {a.out_prefix}_emb.npy {E.shape}, {a.out_prefix}_meta.csv")
    print("갤러리와 설정이 같은지 반드시 교차 검증하라 (모듈 docstring 참고).")


if __name__ == "__main__":
    main()
