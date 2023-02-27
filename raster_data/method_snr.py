import numpy as np
import cv2
from PIL import Image
from math import log10, sqrt
import os
from matplotlib import pyplot as plt
# 1. background 특정 픽셀(가장 잔잔한 픽셀을 따서 노이즈값 비교)을 추출해서 snr 구하기



# 3. 가우시안 blur처리를 한 이미지 - 이미지 = 그나마 깨끗한 이미지.
# 원본이미지 - 깨끗한 이미지 = PSNR (F-R IQA)를 이용하여 차이값 비교. noise의 객관적인 점수는 아니지만 크기비교가 가능하지 않을까.


path = './imageset'
file = os.listdir(path)
len_file = len(file)


#1번 코드

def get_snr(num):
      name = path + '/' + file[num]  #noise iamge로 test
      img = cv2.imread(name)
      img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      dst = cv2.fastNlMeansDenoisingColored(img_rgb,None,10,10,7,21)
      plt.subplot(121),plt.imshow(img_rgb)
      plt.subplot(122),plt.imshow(dst)
      plt.show()

      signal = np.array(dst)                                   ## input orignal data
      mean_signal = np.mean(signal)
      signal_diff = signal - mean_signal
      var_signal = np.sum(np.mean(signal_diff**2))          ## variance of orignal data

      noisy_signal = np.array(img)                             ## input noisy data
      noise = noisy_signal - signal
      mean_noise = np.mean(noise)
      noise_diff = noise - mean_noise
      var_noise = np.sum(np.mean(noise_diff**2))            ## variance of noise

      if var_noise == 0:
            snr = 100                                       ## clean image
      else:
            snr = (np.log10(var_signal/var_noise))*10     ## SNR of the data
            
      print(snr)


# show results
get_snr(-2)

#------------------------------------------------------------------------------------

# 3번 코드

list_d = []

for i in range(len_file):
    name = path+ '/' + file[i]
    # read the image
    img = cv2.imread(name)

    # convert to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # blur
    smooth = cv2.GaussianBlur(gray, None, sigmaX=100, sigmaY=100)

    # divide gray by morphology image
    division = cv2.divide(gray, smooth, scale=255)

    original = gray
    compressed = division

    mse = np.mean((original - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    list_d.append(psnr)

# print(list_d)

    #division이랑 gray랑 비교하기
    def PSNR(original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal .
                    # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr
    
    def main():
        original = gray
        compressed = division
        value = PSNR(original, compressed)
        print(f"PSNR value is {value}dB")

if __name__ == "__main__":
    main()


# show results
cv2.imshow('smooth', smooth)  
cv2.imshow('division', division)  
cv2.waitKey(0)
cv2.destroyAllWindows()