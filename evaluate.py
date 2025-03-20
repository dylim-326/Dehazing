import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import histogram
from PIL import Image
import cv2

def evaluate_images(clean_dir, output_dir):
    """
    PSNR 및 SSIM을 사용하여 이미지를 평가.

    Args:
        clean_dir (str): Clean 이미지 폴더 경로.
        output_dir (str): Dehazing된 이미지 폴더 경로.

    Returns:
        list: 이미지별 PSNR 및 SSIM 결과.
    """
    results = []
    clean_files = sorted(os.listdir(clean_dir))
    output_files = sorted(os.listdir(output_dir))

    for clean_file, output_file in zip(clean_files, output_files):
        clean_path = os.path.join(clean_dir, clean_file)
        output_path = os.path.join(output_dir, output_file)

        if not (os.path.isfile(clean_path) and os.path.isfile(output_path)):
            continue

        # 이미지 로드
        clean_image = np.array(Image.open(clean_path).convert("RGB"))
        output_image = np.array(Image.open(output_path).convert("RGB"))

        # PSNR 및 SSIM 계산
        psnr_value = psnr(clean_image, output_image, data_range=255)
        ssim_value = ssim(clean_image, output_image, channel_axis=-1)

        results.append({
            "Image": clean_file,
            "PSNR": psnr_value,
            "SSIM": ssim_value
        })

    return results

def evaluate_videos(hazy_dir, output_dir):
    """
    히스토그램 비교를 사용하여 동영상 품질을 평가.

    Args:
        hazy_dir (str): Hazy 동영상 폴더 경로.
        output_dir (str): Dehazing된 동영상 폴더 경로.

    Returns:
        list: 동영상별 히스토그램 비교 결과.
    """
    results = []
    hazy_files = sorted(os.listdir(hazy_dir))
    output_files = sorted(os.listdir(output_dir))

    for hazy_file, output_file in zip(hazy_files, output_files):
        hazy_path = os.path.join(hazy_dir, hazy_file)
        output_path = os.path.join(output_dir, output_file)

        if not (os.path.isfile(hazy_path) and os.path.isfile(output_path)):
            continue

        # 동영상 로드
        hazy_cap = cv2.VideoCapture(hazy_path)
        output_cap = cv2.VideoCapture(output_path)

        hazy_hist = None
        output_hist = None

        while True:
            hazy_ret, hazy_frame = hazy_cap.read()
            output_ret, output_frame = output_cap.read()

            if not hazy_ret or not output_ret:
                break

            # 히스토그램 계산
            hazy_hist = cv2.calcHist([hazy_frame], [0], None, [256], [0, 256])
            output_hist = cv2.calcHist([output_frame], [0], None, [256], [0, 256])

        hazy_cap.release()
        output_cap.release()

        # 히스토그램 비교
        if hazy_hist is not None and output_hist is not None:
            hist_diff = cv2.compareHist(hazy_hist, output_hist, cv2.HISTCMP_CORREL)
            results.append({
                "Video": hazy_file,
                "Histogram Correlation": hist_diff
            })

    return results

if __name__ == "__main__":
    # 이미지 경로 설정
    clean_image_dir = r"C:\dev\bme\test\clean_images"
    output_image_dir = r"C:\dev\bme\test\output_images"

    # 동영상 경로 설정
    hazy_video_dir = r"C:\dev\bme\test\hazy_videos"
    output_video_dir = r"C:\dev\bme\test\output_videos"

    # 이미지 평가
    print("Evaluating images...")
    image_results = evaluate_images(clean_image_dir, output_image_dir)
    for res in image_results:
        print(f"Image: {res['Image']}, PSNR: {res['PSNR']:.2f}, SSIM: {res['SSIM']:.4f}")

    # 동영상 평가
    print("Evaluating videos...")
    video_results = evaluate_videos(hazy_video_dir, output_video_dir)
    for res in video_results:
        print(f"Video: {res['Video']}, Histogram Correlation: {res['Histogram Correlation']:.4f}")
