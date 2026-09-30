# PillGuard 임베딩 검색 실험 하네스

알약 이미지 임베딩 검색에서 **"AI Hub 백본 임베딩을 재등록 없이 유지하면서 실환경
각도 변화에 강건해질 수 있는가"** 를 검증하는 실험 코드.

로컬(Mac)에서 실행하도록 작성되었다. 클라우드 세션에는 데이터가 없어 실행하지 못했고,
**합성 데이터로 파이프라인 동작만 검증한 상태**다. 실제 수치는 아직 없다.

## 실행 전 확인: 가설 5-(a)/(b)는 이미 결론이 정해져 있다

쿼리 임베딩을 `e(x) = [b(x); g(x)]` (앞 768 = DINOv2 출력 그대로 동결) 로 두면:

- **(b) 768 절단**: `trunc(e(x)) = b(x)` — 기준선과 **비트 단위로 동일**. 개선폭 0 확정.
- **(a) 2048 전체**: 갤러리는 768이므로 제로 패딩이 필요하고, 그러면
  `cos = (b_q·b_g) / (‖[b_q;g_q]‖·‖b_g‖)` 가 되어 분모의 쿼리 노름만 커진다.
  쿼리 하나 안에서는 모든 알약 점수가 **같은 비율로 축소**되므로 **랭킹 불변**.
  top-k 개선폭 0. 단 유사도 절대값이 낮아져 임계값 기반 판정은 더 보수적으로 바뀐다.

`run_experiments.py` 는 이 두 주장을 매 실행마다 수치로 재확인해 출력한다
(합성 데이터 검증 결과: 절단 유사도 원소별 최대 차이 `0.000e+00`, top-1 차이 `0.000e+00`).

따라서 실제로 측정 가치가 있는 것은 **(c) 학습된 축소 매핑**이며, 이는 정의상
`768→768` 어댑터(은닉 2048)와 동일한 함수족이다. 즉 3·4단계와 같은 것을 재확인한다.

## 데이터 계약

| 파일 | 내용 |
|---|---|
| `gallery_emb.npy` | `(Ng, 768)` AI Hub 임베딩. **읽기 전용, 절대 수정 금지** |
| `gallery_meta.csv` | 헤더에 `pill_id` 필수. 행 순서 = `gallery_emb` 행 순서 |
| `real_emb.npy` | `(Nr, 768)` 실환경 임베딩 |
| `real_meta.csv` | `pill_id` 필수, `angle` 있으면 각도 분포를 보고 |

임베딩이 없으면 먼저 생성한다. **AI Hub 갤러리를 만든 설정과 정확히 일치시켜야 한다**
(모델 변종·풀링·해상도·정규화). `embed_images.py` docstring의 교차검증 절차를 따를 것.

```bash
python embed_images.py --image-root /path/real_images --out-prefix ./real \
    --model dinov2_vitb14 --size 224 --pool cls
```

## 실행

```bash
pip install numpy torch matplotlib pillow torchvision

# 실데이터
python run_experiments.py \
  --gallery-emb gallery_emb.npy --gallery-meta gallery_meta.csv \
  --real-emb real_emb.npy       --real-meta real_meta.csv \
  --out ./results --seeds 0 1 2

# 파이프라인 점검 (합성 데이터, 실제 성능 아님)
python run_experiments.py --synthetic --out ./results_smoke --seeds 0 1 2
```

산출물은 `--out` 디렉터리에만 쓴다 (`summary.json`, `report.md`,
`similarity_distributions.npz`, `similarity_hist.png`). 기존 데이터는 건드리지 않는다.

## 평가 설계 (누출 방지)

- **알약 종류 기준 분할**: `seen`(어댑터 학습에 사용) / `unseen`(전혀 미사용)을 분리 보고.
  같은 알약의 이미지가 train/test에 섞이지 않는다.
- 알약별 실환경 이미지는 `register` / `train` / `test` 로 3분할. 겹치지 않는다.
  - `register`: 2단계 다중 벡터 등록용 (모든 알약)
  - `train`: 어댑터 학습용 (**seen 알약만**)
  - `test`: 쿼리 (모든 알약)
- 어댑터 학습 시 negative는 **seen 알약만** 사용한다. unseen을 negative로 쓰면
  unseen 일반화 평가가 낙관적으로 오염된다.
- 갤러리(AI Hub)와 DINOv2 가중치는 전 단계에서 동결. 어댑터는 쿼리 측에만 적용.

## 지표

`top-1` / `top-5`, 같은·다른 알약 코사인 분포(평균·표준편차), 분리도 `d'`, `AUROC`,
그리고 **최근접 오답 역전율**(`hard_confusion_rate`, 가장 헷갈리는 다른 알약이 정답을
이긴 비율) — 외형 유사 알약 오인식을 직접 본다. 같은 알약 유사도만 오르고 다른 알약도
같이 오르는 경우를 걸러내기 위한 지표다. 시드 3회 평균 ± 표준편차로 보고한다.

## 비교 대상

| 이름 | 단계 |
|---|---|
| `baseline` | 1. 어댑터 없음, 원본 768 |
| `multivector` | 2. 알약당 여러 뷰 등록 후 max 유사도 (= 재등록 기준선) |
| `adapter_768` | 3. 768→768 residual 어댑터 |
| `adapter_2048` | 4. 768→2048→768 어댑터 (차원 확장 이득 확인용) |
| `structural_full` | 5a. 앞 768 동결 + 1280 확장, 2048 전체 사용 |
| `structural_trunc` | 5b. 같은 구조, 768 절단 |
| `structural_reduce` | 5c. 같은 구조, 학습된 2048→768 축소 |

residual 어댑터와 `structural_reduce` 는 **초기값이 기준선과 동일**하도록 마지막 레이어를
0 / `[I|0]` 으로 초기화했다. 학습이 기준선에서 출발하므로 개선폭 해석이 명확하다.

## 선행 연구

이 문제 설정은 이미 정립된 분야다. 특히 **FCT**가 사용자 가설과 거의 같은 목표
(갤러리 재계산 없는 모델 업그레이드)를 학습된 변환으로 달성한다.

- *Forward Compatible Training for Large-Scale Embedding Retrieval Systems*, Ramanujan et al., CVPR 2022
- *Towards Backward-Compatible Representation Learning*, Shen·Xiong et al., CVPR 2020
- *Boundary-Aware Backward-Compatible Representation via Adversarial Learning in Image Retrieval*, Pan et al., CVPR 2023
- *Matryoshka Representation Learning* (절단이 동작하려면 앞 차원을 그 목적으로 학습해야 함)
- *Search-Adaptor: Embedding Customization for Information Retrieval*, ACL 2024
- *DER: Dynamically Expandable Representation for Class Incremental Learning*, Yan et al., CVPR 2021

## 합성 데이터 검증 결과 (파이프라인 점검 전용)

실제 알약 성능이 **아니다**. 하네스가 개선·무영향·악화를 모두 정상 탐지하는지만 확인한 것.

- 개선 탐지: `adapter_768` seen +11.11p, unseen +6.64p
- 구조적 영입 탐지: `structural_full` / `structural_trunc` 모두 **정확히 +0.00p**
- 악화 탐지: `multivector` unseen −13.43p (등록 벡터가 다른 알약 유사도까지 올림)
- 과적합 격차 탐지: seen(+11.11p) > unseen(+6.64p)
- 오인식 쌍 추적: 합성 데이터에 심은 lookalike 쌍을 정확히 회수
