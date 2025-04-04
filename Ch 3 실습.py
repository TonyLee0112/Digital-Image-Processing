import cv2
import numpy as np
import matplotlib.pyplot as plt

def bit_plane_slicing(img_gray):
    """
    입력 그레이스케일 이미지의 각 비트 평면(0~7)을 분리하여 리스트로 반환.
    각 비트 평면은 해당 비트가 1이면 1, 0이면 0으로 표현됩니다.
    """
    bit_planes = []
    for i in range(8):
        # 오른쪽 쉬프트 후 &1을 통해 i번째 비트 평면 추출
        bit_plane = (img_gray >> i) & 1
        bit_planes.append(bit_plane)
    return bit_planes

def resize_image(img_gray, scale):
    """
    cv2.resize를 이용해 이미지 크기를 scale 배율로 조절.
    """
    return cv2.resize(img_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def quantize_image(img_gray, levels):
    """
    그레이스케일 이미지를 levels 단계로 양자화합니다.
    예를 들어 levels=16이면 256단계를 16단계로 줄입니다.
    """
    # 각 양자화 단계의 크기 (정수 연산을 위해 나눗셈)
    Q = 255 // (levels - 1)
    quantized = (img_gray // Q) * Q
    return quantized

def dither_image(img_gray, m=4):
    """
    m단계(예: m=4)로 양자화된 이미지를 Dithering 기법으로 변환합니다.
    - Q : 양자화 단계 간격 (예, m=4이면 Q=85)
    - dither_matrix : 2x2 행렬(예제: [[0,56],[84,28]])를 이미지 전체에 타일링하여 적용
    각 픽셀의 나머지(remainder)를 dither_matrix의 해당 위치의 값과 비교하여
    조건에 따라 양자화 레벨을 조정합니다.
    """
    Q = int(255 / (m - 1))  # 예: m=4이면 Q=85
    # 강의자료 예시에서 사용한 2x2 Dither Matrix (값들은 0~Q 범위 내)
    dither_matrix = np.array([[0, 56],
                              [84, 28]], dtype=np.uint8)
    h, w = img_gray.shape
    # dither_matrix를 이미지 크기에 맞게 타일링
    tiled_matrix = np.tile(dither_matrix, (h // 2 + 1, w // 2 + 1))
    tiled_matrix = tiled_matrix[:h, :w]
    
    dithered = np.copy(img_gray).astype(np.uint8)
    # 각 픽셀마다 양자화 및 dithering 적용
    for i in range(h):
        for j in range(w):
            quant = img_gray[i, j] // Q
            remainder = img_gray[i, j] % Q
            # dither_matrix의 임계값과 비교하여 양자화 레벨 조정 (최대 m-1)
            if remainder > tiled_matrix[i, j]:
                quant = min(quant + 1, m - 1)
            dithered[i, j] = quant * Q
    return dithered

def error_diffusion(img_gray):
    """
    Floyd-Steinberg error diffusion 알고리즘을 사용해 
    이진화(0 또는 255) 이미지를 생성합니다.
    각 픽셀에서 발생한 오차를 주변 픽셀로 분산시켜 보다 자연스러운 결과를 만듭니다.
    """
    # float32 타입으로 변환하여 계산의 정확도를 높임
    img = img_gray.astype(np.float32)
    h, w = img.shape
    output = np.copy(img)
    
    for i in range(h):
        for j in range(w):
            old_pixel = output[i, j]
            # 임계값 128 기준 이진화
            new_pixel = 255 if old_pixel >= 128 else 0
            output[i, j] = new_pixel
            error = old_pixel - new_pixel
            # Floyd-Steinberg 가중치 분산
            if j+1 < w:
                output[i, j+1] += error * 7/16
            if i+1 < h and j > 0:
                output[i+1, j-1] += error * 3/16
            if i+1 < h:
                output[i+1, j] += error * 5/16
            if i+1 < h and j+1 < w:
                output[i+1, j+1] += error * 1/16
    # 값의 범위를 0~255로 제한
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

def main():
    # 1. 이미지 불러오기 (그레이스케일)
    img_path = 'test_image.jpg'
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        # 이미지 파일이 없으면 예제용 그레이디언트 이미지를 생성
        print("이미지를 찾을 수 없으므로, 그레이디언트 이미지를 생성합니다.")
        img_gray = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    
    # -------------------------------
    # (1) 기본 이미지 표시
    # -------------------------------
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 4, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # -------------------------------
    # (2) 비트 평면 분해
    # -------------------------------
    bit_planes = bit_plane_slicing(img_gray)
    # 예시로 하위 4비트 평면(0~3)을 표시합니다.
    for i in range(4):
        plt.subplot(2, 4, i+2)
        plt.imshow(bit_planes[i], cmap='gray')
        plt.title(f"Bit Plane {i}")
        plt.axis('off')
    
    # -------------------------------
    # (3) 공간 해상도 변경
    # -------------------------------
    # 이미지를 0.5배 축소한 후 다시 2배 확대하여 해상도 손실 확인
    img_down = resize_image(img_gray, 0.5)
    img_up = resize_image(img_down, 2.0)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_down, cmap='gray')
    plt.title("Downsampled (0.5x)")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_up, cmap='gray')
    plt.title("Upsampled (2x)")
    plt.axis('off')
    
    # -------------------------------
    # (4) 명암 단계(Gray Level) Quantization
    # -------------------------------
    # 16, 8, 4, 2 단계로 양자화
    levels = [16, 8, 4, 2]
    plt.figure(figsize=(12, 8))
    for idx, lev in enumerate(levels):
        quantized = quantize_image(img_gray, lev)
        plt.subplot(2, 2, idx+1)
        plt.imshow(quantized, cmap='gray')
        plt.title(f"Quantization: {lev} levels")
        plt.axis('off')
    
    # -------------------------------
    # (5) Dithering (특히 중요)
    # -------------------------------
    # m=4 (4단계)로 양자화한 후, Dithering 적용
    dithered_img = dither_image(img_gray, m=4)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    # 단순 uniform quantization 결과와 비교
    plt.imshow(quantize_image(img_gray, 4), cmap='gray')
    plt.title("Uniform Quantization (m=4)")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(dithered_img, cmap='gray')
    plt.title("Dithered Image (m=4)")
    plt.axis('off')
    
    # -------------------------------
    # (6) Error Diffusion (Floyd-Steinberg)
    # -------------------------------
    error_diffused_img = error_diffusion(img_gray)
    plt.figure(figsize=(6, 6))
    plt.imshow(error_diffused_img, cmap='gray')
    plt.title("Error Diffusion (Floyd-Steinberg)")
    plt.axis('off')
    
    # 모든 결과 창 표시
    plt.show()

if __name__ == "__main__":
    main()
