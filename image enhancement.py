import cv2
import os
import numpy as np


'''
    1. 水平垂直翻转
    2. 高斯噪声
    3. 45°旋转
    4. 昏暗
'''


# 添加高斯噪声
def gaussian_noise(img, mean=0, sigma=0.1):
    '''
    此函数用将产生的高斯噪声加到图片上
    均值为0，是保证图像的亮度不会有变化，而方差大小则决定了高斯噪声的强度。
    方差/标准差越大，噪声越强。
    
    传入参数:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回值:
        gaussian_out : 噪声处理后的图片
    '''
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out * 255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out  # 这里也会返回噪声，注意返回值


# 变暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 变亮
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# 水平垂直翻转
def flip(pic):
    '''
    1 : 水平翻转
    0 ； 垂直翻转
    -1： 水平垂直翻转
    '''
    img = cv2.flip(pic, -1)
    return img


# 45度旋转
def rotate(pic):
    rows, cols = pic.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    pic = cv2.warpAffine(pic, M, (cols, rows))
    return pic

# oldpath: 原始图像位置
# oldpath:处理后的图像位置
oldpath = r'..\dataset\train'
newpath = r'..\data_augmentation\dataset\train'
number = 0  

# 遍历训练集的每张图片
folders = os.listdir(oldpath)
for folder in folders:
    for image in os.listdir(os.path.join(oldpath, folder)):
        img = cv2.imread(os.path.join(oldpath, folder, image))

# 添加高斯噪声
        img_gaussnoise = gaussian_noise(img, mean=0, sigma=0.1)
        cv2.imwrite(os.path.join(newpath, folder, str(image[0:-4]) + '_gaussnoise.jpg'), img_gaussnoise)

# 水平垂直翻转
        img_flip = flip(img)
        cv2.imwrite(os.path.join(newpath, folder, str(image[0:-4]) + '_flip.jpg'), img_flip)

# 旋转45°
        img_rotate45 = rotate(img)
        cv2.imwrite(os.path.join(newpath, folder, str(image[0:-4]) + '_rotate45.jpg'), img_rotate45)

# 降低亮度
        img_darker = darker(img, percetage=0.8)
        cv2.imwrite(os.path.join(newpath, folder, str(image[0:-4]) + '_darker.jpg'), img_darker)

        number += 1
        print('第{0}张图片处理完成'.format(number))

print("全部转换完成！")

