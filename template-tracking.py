import os
import numpy as np
import cv2
from numpy.core.fromnumeric import shape


def projective_transformation(x, y, params):
    x1 = (params[0]*x+params[1]*y+params[2])/(params[6]*x+params[7]*y+params[8])
    y1 = (params[3]*x+params[4]*y+params[5])/(params[6]*x+params[7]*y+params[8])
    return [x1,y1]

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

def getCoord(width, height):
    coord = [[[x, y, 1] for x in range(width)] for y in range(height)]
    coord = np.array(coord)
    coord = coord.reshape((height*width, 3))
    return coord

def getJacobian(width, height, coord, params) -> np.ndarray:
    # returns jacobian for projective tranformation - a height*width*2*9 matrix
    J = getDeltaW(coord, params)
    J = J.reshape((height, width, 2, 9))
    return J

def jacobian_projective_tranformation(x, y, params):
    J = [[0 for i in range(9)] for j in range(2)]
    J[0][2] = 1/(params[6]*x+params[7]*y+params[8])
    J[0][0] = x*J[0][2]
    J[0][1] = y*J[0][2]

    J[0][8] = -(params[0]*x+params[1]*y+params[2])/((params[6]*x+params[7]*y+params[8])**(2))
    J[0][6] = x*J[0][8]
    J[0][7] = y*J[0][8]

    J[1][5] = 1/(params[6]*x+params[7]*y+params[8])
    J[1][3] = x*J[1][5]
    J[1][4] = y*J[1][5]

    J[1][8] = -(params[3]*x+params[4]*y+params[5])/((params[6]*x+params[7]*y+params[8])**(2))
    J[1][6] = x*J[1][8]
    J[1][7] = y*J[1][8]

    return np.array(J)

def crop(image, box):
    return image[box[0][1]:(box[1][1]+1), box[0][0]:(box[1][0]+1)]

def LK_parameterized_tracking(old_frame, template_box, new_frame, coord, params):
    T = crop(cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY), template_box)
    I = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)
    # grad_I = np.stack((Ix, Iy), axis = -1).reshape((Ix.shape[0], Ix.shape[1], 1, 2))                                  # height*width*1*2 dimension
    # print(Ix.shape)
    height, width = I.shape[0], I.shape[1]
    # print(height, width)
    for i in range(1):
        # inv_params = np.linalg.inv(params)
        warped_Ix = cv2.warpPerspective(Ix, params, dsize=(width, height), flags = (cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP))
        warped_Iy = cv2.warpPerspective(Iy, params, dsize=(width, height), flags = (cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP))
        warped_I = crop(cv2.warpPerspective(I, params, dsize=(width, height), flags = (cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)), template_box)
        warped_grad_I = crop(np.stack((warped_Ix, warped_Iy), axis = -1).reshape((height, width, 1, 2)), template_box) # height*width*1*2 dimension
        # print(warped_I)
        # print(warped_grad_I)
        # print(T.shape)
        # print(params)
        # need to compute J only once, so we can take this out of this function
        J = crop(getJacobian(width, height, coord, params), template_box)                   # height*width*2*num_params dimension
        # print(J)
        steepest_descent = np.matmul(warped_grad_I, J)                                      # height*width*1*num_params dimension
        steepest_descent_T = steepest_descent.transpose((0,1,3,2))                          # height*width*num_params*1 dimension
        H = np.sum(np.matmul(steepest_descent_T, steepest_descent), axis=(0, 1))            # num_params*num_params dimension
        inv_H = np.linalg.pinv(H)                                                           # num_params*num_params dimension
        delta_I = T-warped_I                                                                
        delta_I = delta_I.reshape((delta_I.shape[0], delta_I.shape[1], 1, 1))               # height*width*1*1 dimension
        delta_params = np.sum(np.matmul(steepest_descent_T, delta_I), axis=(0, 1))          
        delta_params = np.matmul(inv_H, delta_params)                                       # num_params*1 dimension
        delta_params = delta_params.reshape((3,3))
        print(delta_params)
        new_p = delta_params+params
        params = new_p

    params = params.reshape((9))
    corner_points = [template_box[0], [template_box[0][0], template_box[1][1]], template_box[1], [template_box[1][0], template_box[0][1]]]
    warped_corner_points = [projective_transformation(corner_points[i][0], corner_points[i][1], params) for i in range(4)]
    warped_corner_points = np.array(warped_corner_points, np.int32)
    warped_corner_points = warped_corner_points.reshape((-1, 1, 2))
    frame_track = cv2.polylines(new_frame.copy(), [warped_corner_points], True, (255, 0, 0), 2)
    return frame_track, params.reshape((3,3))


#returns bounding rectangle as x, y, w, h (w is along x and h is along y)
def getRect(val):
    val = val.replace("\t", ",")
    bounds = val.rstrip().split(",")
    bounds = [int(x) for x in bounds]
    return bounds

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

def blockBasedTracking(frame, template, template_start_point, method):
    max_translation_limit = 20
    height, width  = template.shape[0], template.shape[1]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(frame_gray, template, method)
    res_h, res_w = res.shape[0], res.shape[1]
    # print(frame.shape)
    # print(res.shape)
    # print(template.shape)
    window_top_left = (max(0, template_start_point[0]-max_translation_limit), max(0, template_start_point[1]-max_translation_limit))
    window_bottom_right = (min(res_w, template_start_point[0]+max_translation_limit), min(res_h, template_start_point[1]+max_translation_limit))
    mask = np.zeros((res_h, res_w), dtype="uint8")
    cv2.rectangle(mask, window_top_left, window_bottom_right, 255, -1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # Assuming small translation
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res, mask = mask)
    if (method==cv2.TM_SQDIFF or method==cv2.TM_SQDIFF_NORMED):
        top_left = min_loc 
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)
    frame_tracking = cv2.rectangle(frame.copy(), top_left, bottom_right, (255, 0, 0), 2)
    new_template = frame_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
    return frame_tracking, template, top_left
    # Updating template at every frame
    # return frame_tracking, new_template, top_left

frames = []
path_vid = "A2/BlurCar2/"
filenames = os.listdir(path_vid+'img/')
groundtruth_file = open(path_vid+'groundtruth_rect.txt')
groundtruth_rect = groundtruth_file.readlines()
groundtruth_rect = [getRect(x) for x in groundtruth_rect]
bounding_rect = [groundtruth_rect[0]]
for filename in filenames:
    frame = cv2.imread(os.path.join(path_vid+'img/', filename))
    frames.append(frame)

frames = np.array(frames)
#Get initial template
template = frames[0][groundtruth_rect[0][1]:(groundtruth_rect[0][1]+groundtruth_rect[0][3]), groundtruth_rect[0][0]:(groundtruth_rect[0][0]+groundtruth_rect[0][2])].copy()
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_start_point = (groundtruth_rect[0][0], groundtruth_rect[0][1])
template_end_point = (groundtruth_rect[0][0]+groundtruth_rect[0][2], groundtruth_rect[0][1]+groundtruth_rect[0][3])
template_box = (template_start_point, template_end_point)
coord = getCoord(frames[0].shape[1], frames[0].shape[0])
params = [1, 0, 0, 0, 1, 0, 0, 0, 1]
params = np.array(params, dtype=np.float32)
params = params.reshape((3,3))
for i in range(1, len(frames)):
    frame = frames[i]
    LK_track, params = LK_parameterized_tracking(frames[0], template_box, frame, coord, params)
    cv2.imshow("template tracking LK", LK_track)
    # template_track, template, template_start_point = blockBasedTracking(frame, template, template_start_point, cv2.TM_CCORR_NORMED)
    # bounding_rect.append((template_start_point[0], template_start_point[1], template.shape[1], template.shape[0]))
    # cv2.imshow("template tracking block based", template_track)
    start_point = (groundtruth_rect[i][0], groundtruth_rect[i][1])
    end_point = (groundtruth_rect[i][0]+groundtruth_rect[i][2], groundtruth_rect[i][1]+groundtruth_rect[i][3])
    frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
    cv2.imshow("template tracking groundtruth", frame)
    k = cv2.waitKey(0)
    if k==27:
        break
# print("mIOU: "+str(getMeanIOUScore(bounding_rect, groundtruth_rect)))
cv2.destroyAllWindows()


