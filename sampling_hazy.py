import os
import shutil

# 원본 이미지 경로 (foggy 데이터가 저장된 루트 디렉토리)
foggy_root = r"C:\Users\LDY\Desktop\leftImg8bit_trainvaltest_foggy\leftImg8bit_foggy\train"

# 타겟 경로 (필터링된 파일을 저장할 위치)
output_dir = r"C:\dev\bme\train\hazy"
os.makedirs(output_dir, exist_ok=True)

# 선택 기준: 안개 밀도가 0.02인 파일
target_beta = "_foggy_beta_0.02"

# 모든 하위 폴더에서 파일 탐색
for root, _, files in os.walk(foggy_root):  # os.walk를 사용해 모든 하위 폴더 탐색
    for file_name in files:
        if target_beta in file_name and file_name.endswith(".png"):  # 조건에 맞는 파일 필터링
            source_path = os.path.join(root, file_name)
            dest_path = os.path.join(output_dir, file_name)
            shutil.copy(source_path, dest_path)  # 파일 복사
            print(f"Copied: {source_path} -> {dest_path}")

print("File selection and copy complete.")
