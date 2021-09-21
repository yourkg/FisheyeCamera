import cv2
import numpy as np
from tqdm import tqdm


def get_useful_area(image):
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,image_binary=cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours,_=cv2.findContours(image_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contour_fisheye=sorted(contours, key=cv2.contourArea, reverse=True)[0]
    center, radius = cv2.minEnclosingCircle(contour_fisheye)
    mask=np.zeros_like(image, dtype=np.uint8)
    mask=cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), (1,1,1),-1)
    image_useful=image*mask
    image_fisheye=image_useful[int(center[1])-int(radius):int(center[1])+int(radius),int(center[0])-int(radius):int(center[0])+int(radius),:]
    return image_fisheye

##载入任意一张原图
image=cv2.imread('./images/0.jpg')
##提取有效圆形区域
image=get_useful_area(image)

if image.shape[0]!=image.shape[1]:
    raise ValueError('Image width isn\'t equal to height!')
##计算mapx和mapy并保存
R=image.shape[0]//2
mapx=np.zeros([2*R,2*R],dtype=np.float32)
mapy=np.zeros([2*R,2*R],dtype=np.float32)
for i in tqdm(range(mapx.shape[0])):
    for j in range(mapx.shape[1]):
        mapx[i,j]=j
        mapy[i,j]=(i-R)/R*(R**2-(j-R)**2)**0.5+R
np.save('./npy/mapx.npy',mapx)
np.save('./npy/mapy.npy',mapy)


##载入remap矩阵和鱼眼原图，提取有效区域并做矫正
mapx=np.load('./npy/mapx.npy')
mapy=np.load('./npy/mapy.npy')
image=cv2.imread('./images/0.jpg')
image=get_useful_area(image)
image_remap=cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
cv2.imwrite('./images/4.jpg',image_remap)

