#version1.0 2021-8-27
#基本相当于copy整合版本

import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import shutil

def get_useful_area(image):
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,image_binary=cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,contours,_=cv2.findContours(image_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contour_fisheye=sorted(contours, key=cv2.contourArea, reverse=True)[0]
    center, radius = cv2.minEnclosingCircle(contour_fisheye)
    mask=np.zeros_like(image, dtype=np.uint8)
    mask=cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), (1,1,1),-1)
    image_useful=image*mask
    image_fisheye=image_useful[int(center[1])-int(radius):int(center[1])+int(radius),int(center[0])-int(radius):int(center[0])+int(radius),:]
    return image_fisheye

class Calibrate:
   '含几个基本处理方法'
   ##设置超参数
   CHECKERBOARD = (6,9)#棋盘大小 w=6, h=9
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #subpix时的终止条件
   radius=(11,11) #subpix时的搜索半径
   def __init__(self,path="inputs"):
       ##获取特征点并做可视化
       print('Start to draw feature points...\n')
       objpoints = [] # 3d point in real world space
       imgpoints = [] # 2d points in image plane.
       objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
       objp[0,:,:2] = np.mgrid[:CHECKERBOARD[0], :CHECKERBOARD[1]].T.reshape(-1, 2)
       if os.path.isdir('./shows'):
           shutil.rmtree('./shows')
       os.makedirs('./shows')
       path_all = glob.glob('./inputs/*.jpg')
       for path in tqdm(path_all):
           image = cv2.imread(path)
           gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
           ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,None)
           if ret == True:
               objpoints.append(objp)
               corners=cv2.cornerSubPix(gray,corners, radius, (-1,-1), criteria)
               imgpoints.append(corners)
               #可视化
               path_out=path.replace('/inputs/','/shows/')
               cv2.drawChessboardCorners(image, CHECKERBOARD , corners, True) #对应的世界坐标系点顺序是按先x轴后y轴，右手法则定z轴
               cv2.imwrite(path_out,image)    
           else:
               print(path)
       ##计算相机参数矩阵K和畸变参数向量D
       print('Start to calibrate...\n')
       h, w = gray.shape[:2]
       flags = 0
       flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
       flags |= cv2.fisheye.CALIB_CHECK_COND
       flags |= cv2.fisheye.CALIB_FIX_SKEW
       criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
       ret, K, D, rvecs, tvecs =cv2.fisheye.calibrate(objpoints,imgpoints,(w,h),K=None,D=None,rvecs=None,tvecs=None,flags=flags,criteria=criteria)
       print(ret)
       print(K)
       print(D)
       ##计算mapx和mapy并保存
       map_combine, _ = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w,h),cv2.CV_16SC2)
       self.mapy=map_combine[:,:,1].astype(np.float32)
       self.mapx=map_combine[:,:,0].astype(np.float32)
       #np.save('./npy/mapx.npy',mapx)
       #np.save('./npy/mapy.npy',mapy)
       return
   
   def demo1(self,):
       '棋盘标定法的还原'
       ##载入remap矩阵和鱼眼原图，并做矫正
       mapx=self.mapx#np.load('./npy/mapx.npy')
       mapy=self.mapy#np.load('./npy/mapy.npy')
       image=cv2.imread('./images/0.jpg')
       image_remap=cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
       cv2.imwrite('./images/1.jpg',image_remap)
   def demo2(self,):
       ##载入任意一张原图
image=cv2.imread('./images/0.jpg')
##提取有效圆形区域
image=get_useful_area(image)

if image.shape[0]!=image.shape[1]:
    raise ValueError('Image width isn\'t equal to height!')
##计算mapx和mapy并保存
R=image.shape[0]//2
W=int(2*np.pi*R)
H=R
mapx=np.zeros([H,W],dtype=np.float32)
mapy=np.zeros([H,W],dtype=np.float32)
for i in tqdm(range(mapx.shape[0])):
    for j in range(mapx.shape[1]):
        angle=j/W*np.pi*2
        radius=H-i
        mapx[i,j]=R+np.sin(angle)*radius
        mapy[i,j]=R-np.cos(angle)*radius
np.save('./npy/mapx.npy',mapx)
np.save('./npy/mapy.npy',mapy)


##载入remap矩阵和鱼眼原图，提取有效区域并做矫正
mapx=np.load('./npy/mapx.npy')
mapy=np.load('./npy/mapy.npy')
image=cv2.imread('./images/0.jpg')
image=get_useful_area(image)
image_remap=cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
cv2.imwrite('./images/2.jpg',image_remap)
