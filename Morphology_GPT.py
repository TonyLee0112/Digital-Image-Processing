import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# 1. 로컬 파일 → 흑백 이미지
def path_to_gray(path: str | Path) -> np.ndarray:
    """파일 경로를 받아 흑백(np.uint8) 이미지 반환"""
    path = Path(path).expanduser().resolve()
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'파일을 찾을 수 없습니다: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# 2. 수리형태학 연산
def morph(img: np.ndarray,
          op: str,
          ksize: int = 3,
          shape: int = cv2.MORPH_RECT,
          n_iter: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(shape, (ksize, ksize))
    if op == 'dilate':
        return cv2.dilate(img, kernel, iterations=n_iter)
    if op == 'erode':
        return cv2.erode(img, kernel, iterations=n_iter)
    if op == 'open':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=n_iter)
    if op == 'close':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=n_iter)
    raise ValueError('op 인자는 dilate/erode/open/close 중 하나여야 합니다.')


# 3. 시각화
def show_pair(src: np.ndarray, dst: np.ndarray, title: str):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1), plt.imshow(src, cmap='gray'), plt.axis('off'), plt.title('Grayscale')
    plt.subplot(1, 2, 2), plt.imshow(dst, cmap='gray'), plt.axis('off'), plt.title(title)
    plt.tight_layout()
    plt.show()


# 4. CLI 진입점
img_path = r"C:\Users\leesooho\Desktop\Digital_image_processing\Sample Image\titan.png"
gray = path_to_gray(img_path.strip())

print('\n연산 선택')
print('1: Dilation  |  2: Erosion  |  3: Opening  |  4: Closing')
op = {'1': 'dilate', '2': 'erode', '3': 'open', '4': 'close'}.get(input('번호: ').strip())
if op is None:
    print('잘못된 번호')

result = morph(gray, op)
show_pair(gray, result, op.capitalize())

