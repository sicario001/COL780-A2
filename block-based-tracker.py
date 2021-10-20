import os
import numpy as np
import cv2
from numpy.core.fromnumeric import shape


hyper_parameters_car = {"update-template":False, "translation-limit-factor":2, "scale-low":0.6, "scale-high":1.5, "scale-increment":0.05}
hyper_parameters_bolt = {"update-template":True, "translation-limit-factor":100,"scale-low":1.0, "scale-high":1.05, "scale-increment":0.05}
hyper_parameters_liquor = {"update-template":False, "translation-limit-factor":2, "scale-low":0.6, "scale-high":1.5, "scale-increment":0.05}
hyper_parameters = hyper_parameters_liquor
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
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best = None
    scale = hyper_parameters["scale-low"]
    N = (hyper_parameters["scale-high"]-hyper_parameters["scale-low"])//hyper_parameters["scale-increment"]
    for i in range(int(N)):
        template_scale = cv2.resize(template.copy(), None, fx= scale, fy= scale, interpolation= cv2.INTER_LINEAR)
        height, width  = template_scale.shape[0], template_scale.shape[1]
        max_translation_limit = (height+width)//hyper_parameters["translation-limit-factor"]
        
        res = cv2.matchTemplate(frame_gray, template_scale, method)
        res_h, res_w = res.shape[0], res.shape[1]
        # print(frame.shape)
        # print(res.shape)
        # print(template.shape)
        window_top_left = (max(0, template_start_point[0]-max_translation_limit), max(0, template_start_point[1]-max_translation_limit))
        window_bottom_right = (min(res_w, template_start_point[0]+max_translation_limit), min(res_h, template_start_point[1]+max_translation_limit))
        mask = np.zeros((res_h, res_w), dtype="uint8")
        cv2.rectangle(mask, window_top_left, window_bottom_right, 255, -1)
        if (hyper_parameters["update-template"]):
            # Assuming small translation
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res, mask = mask)    
        else:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if (method==cv2.TM_SQDIFF or method==cv2.TM_SQDIFF_NORMED):
            top_left = min_loc 
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)
        if (method==cv2.TM_SQDIFF or method==cv2.TM_SQDIFF_NORMED):
            if (best==None or min_val<best[0]):
                best = [min_val, scale, top_left, bottom_right]
        else:
            if (best==None or max_val>best[0]):
                best = [max_val, scale, top_left, bottom_right]

        scale +=hyper_parameters["scale-increment"]
    top_left, bottom_right = best[2], best[3]
    frame_tracking = cv2.rectangle(frame.copy(), top_left, bottom_right, (255, 0, 0), 2)
    new_template = frame_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
    if (hyper_parameters["update-template"]):
        # Updating template at every frame
        return frame_tracking, new_template, top_left
    else:
        return frame_tracking, new_template, top_left
    
    

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
template_end_point = (groundtruth_rect[0][0]+groundtruth_rect[0][2], groundtruth_rect[0][1]+groundtruth_rect[0][3])
template_box = (template_start_point, template_end_point)
for i in range(1, len(frames)):
    frame = frames[i]
    template_track, matched_template, template_start_point = blockBasedTracking(frame, template, template_start_point, cv2.TM_CCORR_NORMED)
    bounding_rect.append((template_start_point[0], template_start_point[1], matched_template.shape[1], matched_template.shape[0]))
    if(hyper_parameters["update-template"]):
        template = matched_template
    cv2.imshow("template tracking block based", template_track)
    start_point = (groundtruth_rect[i][0], groundtruth_rect[i][1])
    end_point = (groundtruth_rect[i][0]+groundtruth_rect[i][2], groundtruth_rect[i][1]+groundtruth_rect[i][3])
    frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
    cv2.imshow("template tracking groundtruth", frame)
    k = cv2.waitKey(1)
    # if k==27:
    #     break
print("mIOU: "+str(getMeanIOUScore(bounding_rect, groundtruth_rect)))
cv2.destroyAllWindows()


