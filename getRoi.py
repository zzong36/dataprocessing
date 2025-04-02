import cv2
import json

IMG_PATH = "./rawdata/full1.jpg"  # ROI를 잡을 이미지 경로
roi_points_list = []               # ROI 포인트들을 담을 리스트
current_points = []                # 현재 ROI 4점
roi_counter = 1                    # ROI 이름 자동 증가 (A1부터 시작)

img = cv2.imread(IMG_PATH)
display_img = img.copy()

def mouse_callback(event, x, y, flags, param):
    global current_points, display_img, roi_counter

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("image", display_img)

        if len(current_points) == 4:
            name = f"A{roi_counter}"  # 자동 이름 생성: A1, A2, A3...
            roi_counter += 1

            roi_points_list.append({
                "name": name,
                "points": current_points
            })

            print(f"✅ {name} ROI 저장됨: {current_points}")

            current_points = []               # 다음 ROI를 위한 초기화
            display_img = img.copy()          # 원본 이미지로 리셋
            cv2.imshow("image", display_img)  # 다시 그리기

# 실행
cv2.imshow("image", display_img)
cv2.setMouseCallback("image", mouse_callback)
print("🖱️ ROI 꼭짓점 4개를 클릭하세요. 자동으로 A1, A2 이름이 붙습니다. ESC 누르면 종료")

while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC 키 누르면 종료
        break

cv2.destroyAllWindows()

# ROI 정보 출력
print("\n✅ 전체 ROI 좌표 목록:")
for roi in roi_points_list:
    print(f'"{roi["name"]}": {roi["points"]},')

# JSON 파일 저장
with open("roi_points.json", "w") as f:
    json.dump(roi_points_list, f, indent=2)
    print("💾 roi_points.json 파일로 저장됨!")
