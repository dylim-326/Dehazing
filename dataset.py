import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CleanHazyDataset(Dataset):
    def __init__(self, clean_dir, hazy_dir, transform=None):
        """
        Args:
            clean_dir (str): Clean 이미지가 저장된 디렉토리 경로.
            hazy_dir (str): Hazy 이미지가 저장된 디렉토리 경로.
            transform (callable, optional): 이미지를 처리할 transform 함수.
        """
        self.clean_dir = clean_dir
        self.hazy_dir = hazy_dir
        self.transform = transform

        # clean과 hazy 이미지를 매칭
        self.image_pairs = self.match_images()

    def match_images(self):
        """clean 폴더와 hazy 폴더의 파일을 매칭"""
        clean_files = os.listdir(self.clean_dir)
        hazy_files = os.listdir(self.hazy_dir)

        image_pairs = []
        for clean_file in clean_files:
            # clean 이미지에서 공통 키 추출
            key = clean_file.replace("_leftImg8bit.png", "")
            # hazy 이미지 중 매칭되는 파일 검색
            matching_hazy = [f for f in hazy_files if key in f and "_foggy_beta_0.02.png" in f]
            if matching_hazy:
                image_pairs.append((os.path.join(self.clean_dir, clean_file),
                                    os.path.join(self.hazy_dir, matching_hazy[0])))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        clean_path, hazy_path = self.image_pairs[idx]

        clean_image = Image.open(clean_path).convert("RGB")
        hazy_image = Image.open(hazy_path).convert("RGB")

        if self.transform:
            clean_image = self.transform(clean_image)
            hazy_image = self.transform(hazy_image)

        return hazy_image, clean_image
