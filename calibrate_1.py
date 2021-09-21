import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import shutil


##设置超参数
CHECKERBOARD = (6,9)#棋盘大小 w=6, h=9
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #subpix时的终止条件
radius=(11,11) #subpix时的搜索半径


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
mapy=map_combine[:,:,1].astype(np.float32)
mapx=map_combine[:,:,0].astype(np.float32)
np.save('./npy/mapx.npy',mapx)
np.save('./npy/mapy.npy',mapy)


##载入remap矩阵和鱼眼原图，并做矫正
mapx=np.load('./npy/mapx.npy')
mapy=np.load('./npy/mapy.npy')
image=cv2.imread('./images/0.jpg')
image_remap=cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
cv2.imwrite('./images/1.jpg',image_remap)