import os
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile
import streamlit as st
from model import HybridDehazeNet
from ultralytics import YOLO
from skimage.exposure import match_histograms

# GPU 또는 CPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 디헤이징 모델 로드
dehaze_model = HybridDehazeNet()
dehaze_model.load_state_dict(torch.load("dehazing_model.pth", map_location=device))
dehaze_model.to(device).eval()

# YOLO 모델 로드
try:
    yolo_model = YOLO("yolov8n.pt").to(device)
    print("YOLO 모델 로드 성공")
except Exception as e:
    print(f"YOLO 모델 로드 실패: {e}")

# Gamma 보정 함수
def adjust_gamma(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# 디헤이징 처리 함수
def dehaze_with_model(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    frame_tensor = torch.tensor(np.transpose(frame, (2, 0, 1))).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = dehaze_model(frame_tensor)

    dehazed_frame = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    dehazed_frame = (dehazed_frame * 255).astype(np.uint8)  # [0, 255]로 변환 및 dtype 변경
    frame = (frame * 255).astype(np.uint8)  # 원본 frame도 uint8로 변환

    matched_frame = match_histograms(dehazed_frame, frame, channel_axis=-1)  # 색감 조정
    return cv2.cvtColor(matched_frame, cv2.COLOR_RGB2BGR)

# YOLO 객체 감지 함수
def detect_objects_with_yolo(frame):
    try:
        results = yolo_model.predict(source=frame, save=False)
        return results[0].plot()
    except Exception as e:
        print(f"YOLO 추론 중 에러 발생: {e}")
        return frame

# Streamlit 애플리케이션
st.title("Dehazing and Object Detection")
st.markdown("### 업로드한 이미지 또는 동영상과 처리 결과를 비교하세요.")

# 임시 디렉토리 설정
if "temp_dir" not in st.session_state:
    st.session_state["temp_dir"] = tempfile.mkdtemp()

uploaded_file = st.file_uploader("이미지 또는 동영상을 업로드하세요", type=["jpg", "jpeg", "png", "mp4", "avi", "mkv"])

if uploaded_file:
    file_name = uploaded_file.name
    input_type = "image" if uploaded_file.type.startswith("image/") else "video"

    # 업로드 파일 저장
    input_path = os.path.join(st.session_state["temp_dir"], file_name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    if input_type == "image":
        # 이미지 처리
        image = Image.open(input_path)
        st.image(image, caption="업로드한 이미지", use_column_width=True)

        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dehazed_frame = dehaze_with_model(frame)
        detected_frame = detect_objects_with_yolo(dehazed_frame)

        st.image(detected_frame, caption="처리된 이미지", use_column_width=True)

    elif input_type == "video":
        # 동영상 처리
        st.video(input_path, format="video/mp4")

        output_path = os.path.join(st.session_state["temp_dir"], "processed_" + file_name)
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            dehazed_frame = dehaze_with_model(frame)
            detected_frame = detect_objects_with_yolo(dehazed_frame)
            out.write(detected_frame)

        cap.release()
        out.release()

        st.video(output_path, format="video/mp4")
