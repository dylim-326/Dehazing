import os
import shutil

# 원본 이미지 경로
source_root = r"C:\Users\LDY\Desktop\leftImg8bit_trainvaltest\leftImg8bit\train"

# 타겟 경로
destination_dir = r"C:\dev\bme\train\clean"
os.makedirs(destination_dir, exist_ok=True)  # 타겟 경로가 없으면 생성

# 파일 복사
for root, _, files in os.walk(source_root):  # 모든 하위 폴더 탐색
    for file_name in files:
        if file_name.endswith(".png"):  # PNG 파일만 선택
            source_path = os.path.join(root, file_name)
            dest_path = os.path.join(destination_dir, file_name)
            try:
                shutil.copy(source_path, dest_path)  # 파일 복사
                print(f"Copied: {source_path} -> {dest_path}")
            except Exception as e:
                print(f"Failed to copy {source_path}: {e}")

print("All files have been copied.")
