import cv2
import numpy as np
import os

hyper_params_bolt = {"error":0.0001, "layers":1, "iterations":500}
hyper_params_car = {"error":0.001, "layers":3, "iterations":500}
hyper_params_liquor = {"error":0.001, "layers":3, "iterations":500}
hyper_params = hyper_params_liquor



def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def getMeanIOUScore(bounding_rect, groundtruth_rect):
    score = 0.0
    N = min(len(bounding_rect), len(groundtruth_rect))
    for i in range(1, N):
        a = [bounding_rect[i][0], bounding_rect[i][1], bounding_rect[i][0]+bounding_rect[i][2], bounding_rect[i][1]+bounding_rect[i][3]]
        b = [groundtruth_rect[i][0], groundtruth_rect[i][1], groundtruth_rect[i][0]+groundtruth_rect[i][2], groundtruth_rect[i][1]+groundtruth_rect[i][3]]
        score += bb_intersection_over_union(a, b)
        # print(score)
    return score/(N-1)


def checkConverge(p,delta_p,e):
    a = np.array([[0,0,1],[1,0,1],[0,1,1],[1,1,1]])
    w_p = warp(a,p)
    p_n = p + delta_p
    w_p_n = warp(a,p_n)
    diff = np.linalg.norm(w_p-w_p_n)
    return diff < e

def bilinearInterpolate(arr,coord):
    x = np.asarray(coord[:,0])
    y = np.asarray(coord[:,1])
    # arr_y, arr_x = arr.shape[0], arr.shape[1]
    # return arr[y.astype('int32'),x.astype('int32')]
    x0 = np.floor(x).astype('int32')
    x1 = x0 + 1
    y0 = np.floor(y).astype('int32')
    y1 = y0 + 1

    x0 = np.clip(x0, 0, arr.shape[1]-1)
    x1 = np.clip(x1, 0, arr.shape[1]-1)
    y0 = np.clip(y0, 0, arr.shape[0]-1)
    y1 = np.clip(y1, 0, arr.shape[0]-1)

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    if len(arr.shape) == 4:
        wa = np.expand_dims(wa,(1,2))
        wb = np.expand_dims(wb,(1,2))
        wc = np.expand_dims(wc,(1,2))
        wd = np.expand_dims(wd,(1,2))

    return wa*arr[y0,x0] + wb*arr[y1,x0] + wc*arr[y0,x1] + wd*arr[y1,x1]

def warp(coord,p):
    # coord_n   |T| X 3
    # p         3 X 3
    # return    |T| X 2
    
    d = np.dot(coord,p[2])
    x = np.dot(coord,p[0])/d
    y = np.dot(coord,p[1])/d
    f = np.stack([x,y],axis=1)
    return f


def getDiff(prev_frame,curr_frame,coord_p,p):
    # prev_frame    m X n
    # curr_frame    m X n
    # coord_p       |T| X 2
    # p             3 X 3
    # return        |T| X 1

    warped_coord = warp(coord_p,p)              # |T| X 2
    diff = bilinearInterpolate(prev_frame,coord_p) - bilinearInterpolate(curr_frame,warped_coord) # |T| X 1
    return np.expand_dims(diff,-1)

def getDeltaW_translation(coord):
    N = coord.shape[0]
    J = [np.eye(2) for i in range(N)]
    return np.array(J)
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
    x1,y1 = np.min(bounding_box,axis=0)
    x2,y2 = np.max(bounding_box,axis=0)
    return np.array([[[i,j,1] for i in range(x1,x2+1)] for j in range(y1,y2+1)]).reshape((-1,3))

def getDeltaI(frame):
    # frame     m x n
    # return    m X n X 1 X 2

    # TODO: Same shape as original?
    Ix = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    Ix_n = np.asarray( Ix[:,:] )
    Iy_n = np.asarray( Iy[:,:] )
    return np.expand_dims(np.stack([Ix_n,Iy_n],axis=2),axis=2)

def getDeltaP(prev_frame,curr_frame,coord_p,p, Jacobian):
    # bounding_box      4 X 2
    # coord_p = getCoords(bounding_box)                           # |T| X 3
    coord_n = warp(coord_p,p)                                   # |T| X 2
    # print("coord_n",coord_n.shape)
    deltaW = Jacobian                                           # |T| X 2 X 9
    # print("deltaW",deltaW.shape)
    deltaI = bilinearInterpolate(getDeltaI(curr_frame),coord_n) # |T| X 1 X 2
    # print("deltaI",deltaI.shape)
    deltaI_W = deltaI @ deltaW                                  # |T| X 1 X 9
    # print("deltaI_W",deltaI_W.shape)
    deltaI_W_T = np.transpose(deltaI_W,(0,2,1))                 # |T| X 9 X 1
    # print("deltaI_W_T",deltaI_W_T.shape)
    H = np.sum(deltaI_W_T @ deltaI_W,axis=0)                    # 9 X 9
    # print(H)
    # print("H",H.shape)
    H_inv = np.linalg.pinv(H)
    # print("H_inv",H_inv.shape)
    diff = getDiff(prev_frame,curr_frame,coord_p,p)             # |T| X 1
    # print("diff",diff.shape)
    deltaP = H_inv @ np.sum(deltaI_W_T[:,:,0] * diff,axis=0)    # 9
    # print("deltaP",deltaP.shape)
    deltaP = np.array([[0, 0, deltaP[0]], [0, 0, deltaP[1]], [0, 0, 0]])
    # deltaP = np.reshape(deltaP,(3,3))
    # print(np.sum(deltaP))
    # assert False
    # bounding_box_new = warp(np.concatenate(bounding_box,np.ones(1,4),axis=1),p)

    return deltaP

def iterate(prev_frame,curr_frame,coord_p,p, Jacobian,layer):
    # p = np.eye(3)
    for i in range(hyper_params["iterations"]):
        deltaP = getDeltaP(prev_frame,curr_frame,coord_p,p, Jacobian)
        # print(deltaP)
        if i>1 and np.square(deltaP).sum() < hyper_params["error"] / (1 << (2*layer)):
            break

        # print(norm)
        # print(i)
        p+=deltaP
        # print(np.reshape(p,(9,)))
    # print(np.around(p))
    # print(i)
    return p
def getRectBound(params, template_box, frame):
    corner_points = [template_box[0], [template_box[0][0], template_box[1][1]], template_box[1], [template_box[1][0], template_box[0][1]]]
    warped_corner_points = warp(np.concatenate([np.array(corner_points),np.ones([4,1])],axis=1),params).astype('int32')
    max_x = np.max(warped_corner_points[:, 0])
    max_y = np.max(warped_corner_points[:, 1])
    min_x = np.min(warped_corner_points[:, 0])
    min_y = np.min(warped_corner_points[:, 1])

    start_point = (min_x, min_y)
    end_point = (max_x, max_y)
    # print(start_point)
    # print(end_point)
    # print(warped_corner_points)
    frame_bound = cv2.rectangle(frame.copy(), start_point, end_point, (255, 0, 0), 2)
    cv2.imshow("rect-bound", frame_bound)
    return (min_x, min_y, max_x-min_x, max_y-min_y)

def drawBound(params,template_box,frame):
    corner_points = [template_box[0], [template_box[0][0], template_box[1][1]], template_box[1], [template_box[1][0], template_box[0][1]]]
    warped_corner_points = warp(np.concatenate([np.array(corner_points),np.ones([4,1])],axis=1),params).astype('int32')
    warped_corner_points = warped_corner_points.reshape((-1, 1, 2))
    # print(warped_corner_points)
    # print(warped_corner_points,corner_points)
    return cv2.polylines(frame.copy(), [warped_corner_points], True, (255, 0, 0), 2)

def getRect(val):
    val = val.replace("\t", ",")
    bounds = val.rstrip().split(",")
    bounds = [(int(x)) for x in bounds]
    return bounds

def getImgPyr(img, pyr_layers):
    pyr = [img]
    for i in range(pyr_layers):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr
def pyrUpParams(p):
    scale = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 1]])
    return p*scale

def pyrDownParams(p):
    scale = np.array([[1, 1, 0.5], [1, 1, 0.5], [1, 1, 1]])
    return p*scale

def LK_run():
    frames = []
    path_vid = "A2/Liquor/"
    filenames = sorted(os.listdir(path_vid+'img/'))
    groundtruth_file = open(path_vid+'groundtruth_rect.txt')
    groundtruth_rect = groundtruth_file.readlines()
    groundtruth_rect = [getRect(x) for x in groundtruth_rect]
    bounding_rect = [groundtruth_rect[0]]
    for filename in filenames:
        frame = cv2.imread(os.path.join(path_vid+'img/', filename),0)
        frames.append(cv2.resize(frame, None, fx= 1, fy= 1, interpolation= cv2.INTER_LINEAR))

    frames = np.array(frames)
    #Get initial template
    template = frames[0][groundtruth_rect[0][1]:(groundtruth_rect[0][1]+groundtruth_rect[0][3]), groundtruth_rect[0][0]:(groundtruth_rect[0][0]+groundtruth_rect[0][2])].copy()
    template_start_point = [groundtruth_rect[0][0], groundtruth_rect[0][1]]
    template_end_point = [groundtruth_rect[0][0]+groundtruth_rect[0][2], groundtruth_rect[0][1]+groundtruth_rect[0][3]]
    template_box = [template_start_point, template_end_point]
    template_box = np.array(template_box)
    pyr_layers = hyper_params["layers"]
    template_box_pyr = []
    coord_pyr = []
    Jacobian_pyr = []
    frame_0_pyr = getImgPyr(frames[0], pyr_layers)
    for i in range(pyr_layers+1):
        template_box_pyr.append(template_box//(1<<i))
        coord = getCoords(np.array(template_box_pyr[i]))
        coord_pyr.append(coord)
        Jacobian = getDeltaW_translation(coord)
        Jacobian_pyr.append(Jacobian)
    # print("coord",coord.shape,np.min(coord,axis=0),np.max(coord,axis=0))
    p = np.eye(3)
    for i in range(1, len(frames)):
        print(i)
        frame = frames[i]
        frame_pyr = getImgPyr(frame, pyr_layers)
        for layer in range(pyr_layers):
            p = pyrDownParams(p)
        for layer in range(pyr_layers, -1, -1):
            p = iterate(frame_0_pyr[layer], frame_pyr[layer], coord_pyr[layer], p, Jacobian_pyr[layer],layer)
            if (layer>0):
                p = pyrUpParams(p)
       
        cv2.imshow("template tracking LK", drawBound(p,template_box,frame))

        rect_bound = getRectBound(p,template_box, frame)
        bounding_rect.append(rect_bound)

        start_point = (groundtruth_rect[i][0], groundtruth_rect[i][1])
        end_point = (groundtruth_rect[i][0]+groundtruth_rect[i][2], groundtruth_rect[i][1]+groundtruth_rect[i][3])
        frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.imshow("template tracking groundtruth", frame)
        k = cv2.waitKey(1)
        # if k==27:
        #     break
    print("mIOU: "+str(getMeanIOUScore(bounding_rect, groundtruth_rect)))
    cv2.destroyAllWindows()

if __name__ == "__main__":
    LK_run()