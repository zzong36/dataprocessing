import cv2
import numpy as np
import os
import json

# === 설정 ===
INPUT_DIR = "rawdata"        # 원본 이미지 폴더
OUTPUT_DIR = "cropped"          # 저장 폴더
CROP_SIZE = (200, 300)          # 출력 이미지 크기 (width, height)

# === 꼭짓점 시계 방향 정렬 함수 ===
def sort_points_clockwise(pts):
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    pts = sorted(pts, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    return np.array(pts, dtype=np.float32)

# === ROI 좌표 불러오기 ===
with open("roi_points.json", "r") as f:
    roi_data = json.load(f)

# === 저장 폴더 준비 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 이미지 반복 처리 ===
for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)
    base_name = os.path.splitext(filename)[0]

    for roi in roi_data:
        name = roi["name"]
        points = sort_points_clockwise(roi["points"])  # 꼭짓점 정렬

        dst_pts = np.float32([
            [0, 0],
            [CROP_SIZE[0], 0],
            [CROP_SIZE[0], CROP_SIZE[1]],
            [0, CROP_SIZE[1]]
        ])

        M = cv2.getPerspectiveTransform(points, dst_pts)
        warped = cv2.warpPerspective(img, M, CROP_SIZE)

        # 저장: ex) cropped/frame1_A1.jpg
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{name}.jpg")
        cv2.imwrite(out_path, warped)

print("✅ 모든 이미지에 대해 ROI crop 완료!")
