import cv2
from matplotlib import pyplot as plt
import numpy.fft as fft
import numpy as np
import os
from scipy.stats import norm
import seaborn as sns

file = os.listdir('C:/Users/inbum/OneDrive - 충남대학교/workspace/sia/color_histogram/imageset')
# path = 'C:/Users/inbum/OneDrive - 충남대학교/workspace/sia/color_histogram/imageset'
path = 'color_histogram/imageset'
number_file = len(file)

#원본 이미지 
def show_image(num):
    img_num = file[num]
    img = path + '/'  + img_num
    img_array = np.fromfile(img, np.uint8)
    image = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    cv2.imshow(f'{img_num}',image)
    cv2.waitKey()

#gray - 이미지
def show_gray_image(num):
    img_num = file[num]
    img = path + '/'  + img_num
    img_array = np.fromfile(img, np.uint8)
    image = cv2.imdecode(img_array,cv2.IMREAD_GRAYSCALE)
    cv2.imshow(f'{img_num}',image)
    cv2.waitKey()

#rgb 히스토그램
def make_rgb_histr(num):
    img_num = file[num]
    img = path + '/'  + img_num

    # 경로에 한글 처리과정
    img_array = np.fromfile(img, np.uint8)
    image = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    color = ('r','g','b')


    for i,col in enumerate(color):
        histr = cv2.calcHist([image_rgb],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('RGB_Histogram')
    plt.show()

#gray 히스토그램
def make_gray_histr(num):
    img_num = file[num]
    img = path + '/'  + img_num

    # 경로에 한글 처리과정
    img_array = np.fromfile(img, np.uint8)
    image = cv2.imdecode(img_array,cv2.IMREAD_GRAYSCALE)
    
    histr = cv2.calcHist([image],[0],None,[256],[0,256])
    plt.plot(histr, color = 'gray')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('GRAY_Histogram')
    plt.show()

# hv (색상,채도) 히스토그램(jupyter 에서 개별적으로 실행해야함)
def make_hsv_histr(num):
    img_num = file[num]
    img = path + '/'  + img_num

    # 경로에 한글 처리과정
    img_array = np.fromfile(img, np.uint8)
    image = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
    
    # 색상,채도

    histt = cv2.calcHist([image_hsv],[0,1],None,[180,256],[0,180,0,256])
    cv2.imshow('df',histt)
    plt.xlabel("saturation")
    plt.ylabel("Hue")
    plt.imshow(histt)

#------------------------------------------------------------

#rgb 히스토그램 저장
def load_histr(num):
    img_num = file[num]
    img = path + '/'  + img_num

    # 경로에 한글 처리과정
    img_array = np.fromfile(img, np.uint8)
    image = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    color = ('r','g','b')


    for i,col in enumerate(color):
        histr = cv2.calcHist([image_rgb],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    # plt.hist(image_rgb.ravel(),256,[0,256])
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('RGB_Histogram')
    plt.savefig(f'color_histogram/imageset/histogram_{img_num}.png')

# ------------------------------------------------------------

# 이미지 pixels (width,height,chnnel 개수)
def get_shape(num):
    img_num = file[num]
    img = path + '/'  + img_num

    image = cv2.imread(img)

    print(image.shape) # height, width, RGB(number_channel)
    print('size:' , image.size) #  


# snr 값
def get_snr(num):
    img_num = file[num]
    img = path + '/' + img_num
    
    img_array = np.fromfile(img, np.uint8)
    image = cv2.imdecode(img_array,cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    red = cv2.split(image_rgb)[0]
    green = cv2.split(image_rgb)[1]
    blue = cv2.split(image_rgb)[2]
    
    list_k = [image,red,green,blue]
    
    for i in range(4):
        mean = np.mean(list_k[i])
        noise = list_k[i] - mean
        snr = mean / np.std(noise)
        if i == 0:
            print('gray',snr)
        elif i == 1 : 
            print('red',snr)
        elif i == 2 : 
            print('green',snr)
        else :
            print('blue',snr)

# 가장 많은 rgb값을 가지는 rgb 값
def frequent_rgb(num):
    img_num = file[num]
    img = path + '/'  + img_num

    img_array = np.fromfile(img, np.uint8)
    image = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    chans = cv2.split(image_rgb)
    colors = ('r', 'g', 'b')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            red = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            blue = str(elem)
            feature_data = 'R: ' + red + ',' + 'G: '  + green + ',' + 'B: ' + blue
            print(feature_data) # RGB


#-----------------------------------------------------
# parameter(file_num)
k = 10

show_image(k)
show_gray_image(k)
make_rgb_histr(k)
make_gray_histr(k)
make_hsv_histr(k)
load_histr(k)

get_shape(k)
frequent_rgb(k)
get_snr(k)

