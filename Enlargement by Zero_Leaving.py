import cv2
import numpy as np
import matplotlib.pyplot as plt


def zero_interleaving(img, scale):
    # scale은 정수여야 함 (예: 2, 3, …)
    scale = int(scale)
    H, W = img.shape[:2]
    new_H, new_W = H * scale, W * scale
    # 채널 여부에 따라 결과 배열 생성
    if img.ndim == 3:
        result = np.zeros((new_H, new_W, img.shape[2]), dtype=img.dtype)
    else:
        result = np.zeros((new_H, new_W), dtype=img.dtype)
    # 원본 픽셀은 scale 간격으로 대입, 그 사이는 0으로 남음
    result[::scale, ::scale] = img
    return result


def main():
    file_path = r"C:\Users\shb06\Desktop\lulu5.jpg"  # 이미지 파일 경로
    print("예: scale=2 → 원본의 2배 확대 (면적은 4배)")
    scale = float(input("확대 배수 입력 : "))
    img = cv2.imread(file_path)
    if img is None:
        print("이미지 파일을 찾을 수 없습니다.")
        return
    # BGR -> RGB 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_zero = zero_interleaving(img, scale)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(res_zero)
    plt.title("Zero-interleaving Enlargement")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
