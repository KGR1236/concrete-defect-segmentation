# Concrete Segmentation — 콘크리트 결함(균열/박리) 분할

콘크리트 표면 이미지에서 **균열(Crack)** 과 **박리(Spalling)** 를 픽셀 단위로 분할한다.
U-Net, DeepLabV3+, SegFormer 세 모델에 대해 단순 증강과 손실함수 변경의 효과를 비교한다.

## 클래스

| 원본 마스크 픽셀값 | 의미 | 학습 클래스 |
|---|---|---|
| 0 | 배경 | 0 Background |
| 1 | 균열 | 1 Crack |
| 2 | 박리 | 2 Spalling |
| 3, 4, 5 | 철근노출 | 2 Spalling (박리로 병합) |

## 실험 설정

- **분할**: 학습:검증:테스트 = 7:1:2, 박리 포함 여부로 층화 (seed 42)
- **입력**: 512×512, ImageNet 정규화
- **손실**: 가중 CE (배경 0.1, 균열 3, 박리 1) 또는 0.5·CE + 0.5·Dice
- **단순 증강(8배)**: 원본, 좌우/상하 반전, 90° 좌/우 회전, 밝기 0.7/1.3배, 가우시안 노이즈(σ=15)
- **최적화**: Adam lr 1e-4 (SegFormer: AdamW lr 1e-5), ReduceLROnPlateau(patience 5, ×0.5)
- **학습**: 최대 100 epoch, val_loss 기준 조기 종료(patience 15)
- **평가**: 테스트 셋 클래스별 IoU / Precision / Recall, mIoU(배경 제외)

| 모델 | 구성 |
|---|---|
| `unet` | smp U-Net, ResNet34 (ImageNet) |
| `deeplabv3plus` | smp DeepLabV3+, ResNet34 (ImageNet) |
| `segformer` | HuggingFace SegFormer `nvidia/mit-b2` |

## 사용법

1. **데이터 점검** (크기 매칭, 픽셀값, 군집 수, 분할별 픽셀 수)
   ```bash
   python -m scripts.check_data --image-dir <이미지_폴더> --mask-dir <마스크_폴더> [--json-dir <JSON_폴더>]
   ```
2. **학습 + 테스트 평가**
   ```bash
   # 기본 (CE)
   python -m scripts.train --model unet --image-dir <이미지_폴더> --mask-dir <마스크_폴더>
   # 손실함수 변경 (CE + Dice)
   python -m scripts.train --model unet --image-dir ... --mask-dir ... --loss ce_dice
   # 단순 증강 + 손실함수 변경
   python -m scripts.train --model segformer --image-dir ... --mask-dir ... --augment --loss ce_dice
   ```
   best 가중치는 `checkpoints/best_<모델>_<aug|noaug>_<손실>.pth` 로 저장된다.

> 모든 명령은 `concrete_segmentation/` 폴더에서 실행하세요.
> 이미지와 마스크는 파일명 정렬 순서로 짝지어지므로 두 폴더의 파일명 순서가 같아야 합니다.

## 구조

```
concrete_segmentation/
├── scripts/
│   ├── check_data.py      # 데이터 점검
│   └── train.py           # 학습 + 테스트 평가
└── src/
    ├── data/
    │   ├── split.py       # 층화 7:1:2 분할
    │   ├── augment.py     # 단순 증강 8종
    │   └── dataset.py     # ConcreteDataset
    ├── models/factory.py  # U-Net / DeepLabV3+ / SegFormer
    └── training/
        ├── losses.py      # CE, CE+Dice
        ├── metrics.py     # IoU / Precision / Recall
        └── trainer.py     # 학습 루프, 조기 종료
```

## 설치

```bash
pip install -r requirements.txt
```

⚠️ torch는 CUDA 12.1 환경 기준입니다. 다른 CUDA 버전을 쓰는 경우
`requirements.txt`의 torch/torchvision 버전과 `--extra-index-url`을 맞춰 수정하세요.
