import cv2
import json
import numpy as np
import os
from ultralytics import YOLO

# === 설정 ===
TEST_DIR = "./testdata"                      # 테스트 이미지가 들어있는 폴더
ROI_JSON = "roi_points.json"               # ROI 좌표 JSON
MODEL_PATH = "best.pt"                     # 학습된 YOLOv8 분류 모델
CROP_SIZE = (200, 300)                     # ROI 보정 크기 (width, height)

# === 함수: ROI 점들을 시계 방향으로 정렬 ===
def sort_points_clockwise(pts):
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    pts = sorted(pts, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    return np.array(pts, dtype=np.float32)

# === ROI 좌표 불러오기 ===
with open(ROI_JSON, "r") as f:
    roi_data = json.load(f)

# === YOLO 분류 모델 불러오기 ===
model = YOLO(MODEL_PATH)

# === 테스트 이미지들 순회 ===
image_files = [f for f in sorted(os.listdir(TEST_DIR)) if f.lower().endswith(".jpg")]

for idx, filename in enumerate(image_files):
    image_path = os.path.join(TEST_DIR, filename)
    img = cv2.imread(image_path)
    img_draw = img.copy()

    for roi in roi_data:
        name = roi["name"]
        src_pts = sort_points_clockwise(roi["points"])

        dst_pts = np.float32([
            [0, 0],
            [CROP_SIZE[0], 0],
            [CROP_SIZE[0], CROP_SIZE[1]],
            [0, CROP_SIZE[1]]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, CROP_SIZE)

        # === 분류 예측 ===
        result = model(warped, verbose=False)
        label_idx = result[0].probs.top1
        label_name = model.names[label_idx]
        confidence = result[0].probs.top1conf

        # 색상: 초록 or 빨강
        color = (0, 255, 0) if label_name == "empty" else (0, 0, 255)

        # ROI 표시
        cv2.polylines(img_draw, [src_pts.astype(int)], isClosed=True, color=color, thickness=3)

        # 텍스트 표시
        text = f"{name}: {label_name} ({confidence:.2f})"
        text_pos = tuple(np.mean(src_pts, axis=0).astype(int))
        cv2.putText(img_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === 이미지 저장 ===
    output_filename = f"output_visualized{idx}.jpg"
    cv2.imwrite(output_filename, img_draw)
    print(f"✅ 저장됨: {output_filename}")
