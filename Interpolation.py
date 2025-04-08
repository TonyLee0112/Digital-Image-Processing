import cv2
import numpy as np
import matplotlib.pyplot as plt

# 단일 scale 값이 가로와 세로에 동일하게 적용되어 종횡비가 유지됨
def nearest_neighbor_interpolation(img, scale):
    H, W = img.shape[:2]
    new_H, new_W = int(H * scale), int(W * scale)  # X, Y에 동일한 배수를 적용
    if img.ndim == 3:
        result = np.zeros((new_H, new_W, 3), dtype=np.uint8)
    else:
        result = np.zeros((new_H, new_W), dtype=np.uint8)

    for i in range(new_H):
        for j in range(new_W):
            src_i = min(round(i / scale), H - 1)
            src_j = min(round(j / scale), W - 1)
            result[i, j] = img[src_i, src_j]
    return result

def bilinear_interpolation(img, scale):
    H, W = img.shape[:2]
    new_H, new_W = int(H * scale), int(W * scale)  # 동일한 scale 적용
    if img.ndim == 3:
        result = np.zeros((new_H, new_W, 3), dtype=np.uint8)
    else:
        result = np.zeros((new_H, new_W), dtype=np.uint8)

    for i in range(new_H):
        for j in range(new_W):
            x = i / scale
            y = j / scale
            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            x2 = min(x1 + 1, H - 1)
            y2 = min(y1 + 1, W - 1)
            a = x - x1
            b = y - y1

            if img.ndim == 3:
                top = (1 - b) * img[x1, y1] + b * img[x1, y2]
                bottom = (1 - b) * img[x2, y1] + b * img[x2, y2]
                pixel = (1 - a) * top + a * bottom
                result[i, j] = pixel.astype(np.uint8)
            else:
                top = (1 - b) * img[x1, y1] + b * img[x1, y2]
                bottom = (1 - b) * img[x2, y1] + b * img[x2, y2]
                pixel = (1 - a) * top + a * bottom
                result[i, j] = np.clip(pixel, 0, 255)
    return result

def cubic_weight(t, a=-0.5):
    t = abs(t)
    if t <= 1:
        return (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1
    elif t < 2:
        return a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a
    else:
        return 0

def cubic_interpolation(img, scale):
    H, W = img.shape[:2]
    new_H, new_W = int(H * scale), int(W * scale)  # 동일 scale로 가로, 세로 확대/축소
    if img.ndim == 3:
        result = np.zeros((new_H, new_W, 3), dtype=np.uint8)
    else:
        result = np.zeros((new_H, new_W), dtype=np.uint8)

    for i in range(new_H):
        for j in range(new_W):
            x = i / scale
            y = j / scale
            ix, iy = int(np.floor(x)), int(np.floor(y))

            if img.ndim == 3:
                sum_val = np.zeros(3, dtype=np.float64)
            else:
                sum_val = 0.0

            for m in range(-1, 3):
                for n in range(-1, 3):
                    x_index = min(max(ix + m, 0), H - 1)
                    y_index = min(max(iy + n, 0), W - 1)
                    if img.ndim == 3:
                        pixel = img[x_index, y_index].astype(np.float64)
                    else:
                        pixel = float(img[x_index, y_index])
                    weight = cubic_weight(x - (ix + m)) * cubic_weight(y - (iy + n))
                    sum_val += weight * pixel
            if img.ndim == 3:
                result[i, j] = np.clip(sum_val, 0, 255).astype(np.uint8)
            else:
                result[i, j] = int(np.clip(sum_val, 0, 255))
    return result

def main():
    file_path = r"C:\Users\shb06\Desktop\lulu5.jpg"
    print("2 입력 → 가로, 세로 각각 2배 확대 (면적은 4배) \n0.5 입력 → 50% 축소")
    scale = float(input("확대 배수 입력 : "))
    img = cv2.imread(file_path)
    if img is None:
        print("이미지 파일을 찾을 수 없습니다.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("번호 선택:")
    print("1: Bilinear 보간 (원본과 함께)")
    print("2: Nearest Neighbor 보간 (원본과 함께)")
    print("3: General (Cubic) 보간 (원본과 함께)")
    print("4: 원본 및 3개 보간 결과 동시에 출력")
    sel = input("번호 입력: ")

    if sel == '1':
        result = bilinear_interpolation(img, scale)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title("Bilinear Interpolation")
        plt.tight_layout()
        plt.show()
    elif sel == '2':
        result = nearest_neighbor_interpolation(img, scale)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title("Nearest Neighbor")
        plt.tight_layout()
        plt.show()
    elif sel == '3':
        result = cubic_interpolation(img, scale)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title("Cubic Interpolation")
        plt.tight_layout()
        plt.show()
    elif sel == '4':
        res_bi = bilinear_interpolation(img, scale)
        res_nn = nearest_neighbor_interpolation(img, scale)
        res_cubic = cubic_interpolation(img, scale)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.subplot(1, 4, 2)
        plt.imshow(res_bi)
        plt.title("Bilinear")
        plt.subplot(1, 4, 3)
        plt.imshow(res_nn)
        plt.title("Nearest")
        plt.subplot(1, 4, 4)
        plt.imshow(res_cubic)
        plt.title("Cubic")
        plt.tight_layout()
        plt.show()
    else:
        print("잘못된 입력입니다.")

if __name__ == '__main__':
    main()
