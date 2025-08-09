# commento_ai_week3

사용한 데이터셋 : https://docs.ultralytics.com/ko/datasets/segment/carparts-seg/




# 자동차 부품 세그멘테이션 (YOLO11n-seg)

이 프로젝트는 **Ultralytics YOLO11n-seg** 모델을 이용해 자동차 부품 이미지를 객체 탐지 및 세그멘테이션하는 예제입니다.  
사용자는 제공된 데이터셋(`carparts-seg`)을 학습시키고, 테스트 이미지를 통해 모델 성능을 평가할 수 있습니다.

---

## 📌 주요 기능
- YOLO11n-seg 사전 학습 모델 로드  
- 데이터 증강(Augmentation)을 적용한 모델 학습  
- 학습된 가중치 기반 객체 탐지 및 분할(Segmentation)  
- 모델 성능 검증 (mAP, Precision, Recall)  
- 학습 결과(`results.csv`)를 이용한 시각화  

---

## 📂 데이터셋 구조 예시
```bash
datasets/n
└── carparts-seg/
├── images/
│ ├── train/
│ ├── val/
│ └── test/
└── labels/
├── train/
├── val/
└── test/
```

- `images/` : 원본 이미지  
- `labels/` : YOLO 형식 라벨 파일 (`.txt`), 세그멘테이션 폴리곤 좌표 포함  
- `carparts-seg.yaml` : 데이터 경로 및 클래스 설정  

---

## 🛠 설치 방법
```
pip install torch torchvision opencv-python matplotlib ultralytics
```


## 🚀 학습 방법
```
from ultralytics import YOLO
```


### 모델 로드
```
model = YOLO("yolo11n-seg.pt")

# 학습 실행
results = model.train(
    data="carparts-seg.yaml",  # 데이터셋 설정 파일
    epochs=20,                 # 학습 반복 횟수
    imgsz=640,                  # 입력 이미지 크기
    augment=True                # 데이터 증강 활성화
)
```

## 🔍 최신 best.pt 불러오기
```
import glob
from ultralytics import YOLO

latest_model = sorted(glob.glob("runs/segment/*/weights/best.pt"))[-1]
model = YOLO(latest_model)
```


## 🖼 테스트 이미지 불러오기
```
import cv2

image_path = "datasets/carparts-seg/images/test/example.jpg"
image = cv2.imread(image_path)
```


## 📌 예측
```
results = model.predict(
    source=image,
    save=True,
    imgsz=640,
    conf=0.25
)
save=True 옵션을 사용하면 runs/segment/predict/ 폴더에 결과 이미지가 저장됩니다.
```


## 📊 모델 검증
```
metrics = model.val()
print(metrics)
```
출력 예시:
```
makefile
mAP50: 0.85
mAP50-95: 0.72
Precision: 0.88
Recall: 0.81
```


## 📈 예: mAP 변화 시각화
```
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("runs/segment/train2/results.csv")
df = pd.read_csv(csv_path)

plt.plot(df["epoch"], df["metrics/mAP50"])
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5")
plt.title("mAP 변화 추이")
plt.show()
```


## 📁 디렉토리 구조 예시
```
.
├── datasets/
│   └── carparts-seg/
├── runs/
│   └── segment/
│       ├── train*/        # 학습 결과 폴더
│       └── predict*/      # 예측 결과 폴더
├── carparts-seg.yaml
└── model.ipynb
```



