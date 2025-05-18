import matplotlib.pyplot as plt
from PIL import Image

def Img_to_List(img_path) :
    with Image.open(img_path) as img :
        gray = img.convert('L')
        w, h = gray.size
        flat = list(gray.getdata())
        return [flat[i*w : (i+1)*w] for i in range(h)]
    
def bin_img(gray_img) :
    h, w = len(gray_img), len(gray_img[0])
    for y in range(h) :
        for x in range(w) :
            if gray_img[y][x] > 128 :
                gray_img[y][x] = 1
            else :
                gray_img[y][x] = 0
    return gray_img

def Erosion(img_array) :
    kernel = [
              [0,1,0],
              [1,1,1],
              [0,1,0]
              ]
    height = len(img_array)
    width = len(img_array[0])
    
    result_array = [[0 for _ in range(width)] for _ in range(height)]

    for y in range(height) :
        for x in range(width) :
            sum = 0
            for dx in [-1,0,1] :
                for dy in [-1,0,1] :
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height :
                        if abs(dx) + abs(dy) < 2 :
                            if img_array[ny][nx] == 1:
                                sum += 1
            
            if sum == 5 :
                result_array[y][x] = 1
            else :
                result_array[y][x] = 0
            
    return result_array

def Dilation(img_array) :
    # X자 커널
    kernel = [
            [1,0,1],
            [0,1,0],
            [1,0,1]
            ]
    
    height = len(img_array)
    width = len(img_array[0])
    
    result_array = [[0 for _ in range(width)] for _ in range(height)]

    for y in range(height) :
        for x in range(width) :
            # 원래 Pixel 값이 흰색인 곳에 대해서만
            if img_array[y][x] == 1 :
                for dx in [-1,0,1] :
                    for dy in [-1,0,1] :
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height :
                            if kernel[dy+1][dx+1] == 1 :
                                result_array[ny][nx] = 1
    return result_array

def plotting(original, result) :
    figure, axis = plt.subplots(1,2, figsize=(10,5))    
    axis[0].imshow(original, cmap='gray')
    axis[0].set_title('Original Image')
    axis[0].axis('off')

    axis[1].imshow(result, cmap='gray')
    axis[1].set_title('Processed Image')
    axis[1].axis('off')

    plt.tight_layout()
    plt.show()

def multiple_Erosion_Dilation(img_array,E=1,D=1,reversed=False) :
    # E = Erosion 적용 횟수,
    # D = Dilation 적용 횟수,
    # reversed = Erosion 적용 후 Dilation 할 거임?
    # False -> Dilation 반복 적용 후 Erosion
    # True -> Erosion 반복 적용 후 Dilation

    if not reversed :
        for _ in range(E) :
            img_array = Erosion(img_array)
        for _ in range(D) :
            img_array = Dilation(img_array)
    else :
        for _ in range(D) :
            img_array = Dilation(img_array)
        for _ in range(E) :
            img_array = Erosion(img_array)
    return img_array

# main()
img_path = r"C:\Users\leesooho\Desktop\Digital_image_processing\Sample Image\sample2.png"
origin_img = Img_to_List(img_path)
Binary_img = bin_img(origin_img)

closing = multiple_Erosion_Dilation(Binary_img,reversed=True)
opening = multiple_Erosion_Dilation(Binary_img,reversed=False)

plotting(opening,multiple_Erosion_Dilation(opening,reversed=True))
