import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 1. 컬러 이미지 전처리 함수: 이미지 불러와 RGB로 변환 후 정규화
def preprocess_image_color(filepath: str) -> np.ndarray:
    img = Image.open(filepath).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    return img_np


# 2. 각 채널별 FFT 수행 함수
def perform_fft_color(image: np.ndarray) -> np.ndarray:
    fft_channels = []
    for ch in range(3):
        fft_data = np.fft.fft2(image[..., ch])
        fft_data = np.fft.fftshift(fft_data)
        fft_channels.append(fft_data)
    return np.stack(fft_channels, axis=-1)


# 3. 각 채널별 역 FFT 수행 함수
def inverse_fft_color(fft_data: np.ndarray) -> np.ndarray:
    channels = []
    for ch in range(3):
        fft_unshifted = np.fft.ifftshift(fft_data[..., ch])
        channel_recon = np.fft.ifft2(fft_unshifted)
        channels.append(np.abs(channel_recon))
    return np.stack(channels, axis=-1)


# 4. Display 함수: 원본, FFT Magnitude (채널별 노말라이즈 후 의사 컬러) 및 재구성 이미지 표시
def display_color_images(original: np.ndarray, fft_data: np.ndarray, reconstructed: np.ndarray) -> None:
    fft_magnitude = np.log1p(np.abs(fft_data))

    # 각 채널 정규화 함수
    def normalize(channel):
        channel = channel - np.min(channel)
        return channel / (np.ptp(channel) + 1e-8)

    fft_magnitude_norm = np.stack(
        [normalize(fft_magnitude[..., ch]) for ch in range(3)],
        axis=-1
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original Color Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(fft_magnitude_norm)
    plt.title('FFT Magnitude (Pseudo Color)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(reconstructed, 0, 1))
    plt.title('Reconstructed Color Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 실행 예시
if __name__ == '__main__':
    filepath = r"C:\Users\shb06\Desktop\lulu2.jpg"
    original_img = preprocess_image_color(filepath)
    fft_img = perform_fft_color(original_img)
    reconstructed_img = inverse_fft_color(fft_img)
    display_color_images(original_img, fft_img, reconstructed_img)
