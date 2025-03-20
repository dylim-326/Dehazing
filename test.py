import os
import cv2
import torch
import numpy as np
from skimage.exposure import match_histograms
from torchvision import transforms
from PIL import Image
from model import HybridDehazeNet
from ultralytics import YOLO

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridDehazeNet()
model.load_state_dict(torch.load("dehazing_model.pth"))
model.to(device).eval()

# YOLO 모델 로드
try:
    yolo_model = YOLO("yolov8n.pt")  # 사전 학습된 YOLO 모델
    print("YOLO 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"YOLO 모델 로드 중 에러 발생: {e}")

def detect_objects_with_yolo(frame):
    try:
        results = yolo_model(frame)
        annotated_frame = results[0].plot()  # 감지된 객체 표시
        return annotated_frame
    except Exception as e:
        print(f"YOLO 추론 중 에러 발생: {e}")
        return frame  # 에러 발생 시 원본 프레임 반환

# Gamma 보정 함수
def adjust_gamma(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# 전처리 함수 (모델 입력 전)
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0  # [0, 1]로 정규화
    frame_tensor = torch.tensor(np.transpose(frame, (2, 0, 1))).unsqueeze(0).float().to(device)
    return frame_tensor

# 후처리 함수 (모델 출력 후)
def postprocess_frame(output, original_frame):
    dehazed_frame = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    dehazed_frame = (dehazed_frame * 255).astype(np.uint8)  # [0, 255] 범위로 변환
    dehazed_frame = adjust_gamma(dehazed_frame, gamma=1.5)  # Gamma 보정
    matched_frame = match_histograms(dehazed_frame, original_frame, channel_axis=-1)  # 색감 조정
    return matched_frame

# 디헤이징 처리 함수
def dehaze_with_model(frame):
    frame_tensor = preprocess_frame(frame)
    with torch.no_grad():
        output = model(frame_tensor)  # 모델 적용
    dehazed_frame = postprocess_frame(output, frame)
    return dehazed_frame

# 이미지 처리
def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.ToTensor()
    for image_file in os.listdir(input_dir):
        if image_file.endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, image_file)

            # 이미지 로드
            original_image = Image.open(input_path).convert("RGB")
            frame = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

            # Step 1: 디헤이징
            dehazed_frame = dehaze_with_model(frame)

            # Step 2: YOLO 감지
            detected_frame = detect_objects_with_yolo(dehazed_frame)

            # 저장
            cv2.imwrite(output_path, detected_frame)
            print(f"Processed image saved: {output_path}")

# 동영상 처리
def process_videos(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for video_file in os.listdir(input_dir):
        if video_file.endswith((".mp4", ".avi", ".mkv")):
            input_path = os.path.join(input_dir, video_file)
            output_path = os.path.join(output_dir, video_file)

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video file {video_file}")
                continue

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Step 1: 디헤이징
                dehazed_frame = dehaze_with_model(frame)

                # Step 2: YOLO 감지
                detected_frame = detect_objects_with_yolo(dehazed_frame)

                out.write(detected_frame)

            cap.release()
            out.release()
            print(f"Processed video saved: {output_path}")

# 실행 부분
if __name__ == "__main__":
    # 입력 및 출력 경로
    image_hazy_dir = r"C:\dev\bme\test\hazy_images"
    image_output_dir = r"C:\dev\bme\test\output_images"

    video_hazy_dir = r"C:\dev\bme\test\hazy_videos"
    video_output_dir = r"C:\dev\bme\test\output_videos"

    print("Starting image dehazing...")
    process_images(image_hazy_dir, image_output_dir)
    print("Image dehazing completed.")

    print("Starting video dehazing...")
    process_videos(video_hazy_dir, video_output_dir)
    print("Video dehazing completed.")
