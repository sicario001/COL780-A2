import numpy as np
import cv2

def warp(coord,p):
    # coord_n   |T| X 3
    # p         3 X 3
    # return    |T| X 2
    
    d = np.dot(coord,p[2])
    x = np.dot(coord,p[0])/d
    y = np.dot(coord,p[1])/d
    f = np.stack([x,y],axis=1)
    return np.round(f).astype('int32')


def getDiff(prev_frame,curr_frame,coord_p,p):
    # prev_frame    m X n
    # curr_frame    m X n
    # coord_p       |T| X 2
    # p             3 X 3
    # return        |T| X 1

    warped_coord = warp(coord_p,p)              # |T| X 2
    diff = prev_frame[coord_p[:,0],coord_p[:,1]] - curr_frame[warped_coord[:,0],warped_coord[:,1]] # |T| X 1
    return diff

def getDeltaW(coord,a):
    # coord_n   |T| X 3
    # a         3 X 3
    # returns   |T| X 2 X 9
    d0 = np.expand_dims(np.dot(coord,a[0]),axis=-1)
    d1 = np.expand_dims(np.dot(coord,a[1]),axis=-1)
    d2 = np.expand_dims(np.dot(coord,a[2]),axis=-1)

    J0_012 = coord/d2
    J0_345 = np.zeros_like(J0_012)
    J0_678 = -coord*d0/d2**2

    J1_012 = np.zeros_like(J0_012)
    J1_345 = coord/d2
    J1_678 = -coord*d1/d2**2

    # row 1
    r1 = np.concatenate([J0_012,J0_345,J0_678],axis=1)
    # row 2
    r2 = np.concatenate([J1_012,J1_345,J1_678],axis=1)
    # final
    return np.stack([r1,r2],axis=1)

def getCoords(bounding_box):
    pass

def getDeltaI(frame):
    # frame     m x n
    # return    m X n X 1 X 2

    # TODO: Same shape as original?
    Ix = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    Ix_n = np.asarray( Ix[:,:] )
    Iy_n = np.asarray( Iy[:,:] )
    return np.stack([Ix_n,Iy_n],axis=2)    

def getDeltaP(prev_frame,curr_frame,bounding_box,p):
    # bounding_box      4 X 2
    coord_p = getCoords(bounding_box)                           # |T| X 3
    coord_n = warp(coord_p,p)                                   # |T| X 2
    deltaW = getDeltaW(coord_p,p)                               # |T| X 2 X 9
    deltaI = getDeltaI(curr_frame)[coord_n[:,0],coord_n[:,1]]   # |T| X 1 X 2
    deltaI_W = deltaI @ deltaW                                  # |T| X 1 X 9
    deltaI_W_T = np.transpose(deltaI_W,(0,2,1))                 # |T| X 9 X 1
    H = np.sum(deltaI_W_T @ deltaI_W,axis=0)                    # 9 X 9
    H_inv = np.linalg.inv(H)
    diff = getDiff(prev_frame,curr_frame,coord_p,p)             # |T| X 1
    deltaP = H_inv @ np.sum(deltaI_W_T[:,:,0] * diff,axis=0)    # 9

    bounding_box_new = warp(np.concatenate(bounding_box,np.ones(1,4),axis=1),p)

    return deltaP,bounding_box_new