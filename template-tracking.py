import os
import numpy as np
import cv2


def projective_transformation(x, y, params):
    x1 = (params[0]*x+params[1]*y+params[2])/(params[6]*x+params[7]*y+params[8])
    y1 = (params[3]*x+params[4]*y+params[5])/(params[6]*x+params[7]*y+params[8])
    return np.array([x1, y1])

def getJacobian(width, height, params):
    # returns jacobian for projective tranformation - a height*width*2*9 matrix
    return None
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

def LK_parameterized_tracking(template, new_frame):
    new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    Ix = cv2.Sobel(new_frame_gray, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(new_frame_gray, cv2.CV_64F, 0, 1, ksize=5)
    height, width = new_frame_gray.shape[0], new_frame_gray.shape[1]
    params = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    J = getJacobian(width, height, params)
    return None


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
path_vid = "A2/Liquor/"
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
for i in range(1, len(frames)):
    frame = frames[i]

    template_track, template, template_start_point = blockBasedTracking(frame, template, template_start_point, cv2.TM_CCORR_NORMED)
    bounding_rect.append((template_start_point[0], template_start_point[1], template.shape[1], template.shape[0]))
    cv2.imshow("template tracking block based", template_track)
    start_point = (groundtruth_rect[i][0], groundtruth_rect[i][1])
    end_point = (groundtruth_rect[i][0]+groundtruth_rect[i][2], groundtruth_rect[i][1]+groundtruth_rect[i][3])
    frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
    cv2.imshow("template tracking groundtruth", frame)
    k = cv2.waitKey(0)
    if k==27:
        break
print("mIOU: "+str(getMeanIOUScore(bounding_rect, groundtruth_rect)))
cv2.destroyAllWindows()


