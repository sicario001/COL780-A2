import os
import numpy as np
import cv2

#returns bounding rectangle as x, y, w, h (w is along x and h is along y)
def getRect(val):
    bounds = val.rstrip().split(",")
    bounds = [int(x) for x in bounds]
    return bounds


frames = []
path_vid = "A2/Bolt/"
filenames = os.listdir(path_vid+'img/')
groundtruth_file = open(path_vid+'groundtruth_rect.txt')
groundtruth_rect = groundtruth_file.readlines()
groundtruth_rect = [getRect(x) for x in groundtruth_rect]
for filename in filenames:
    frame = cv2.imread(os.path.join(path_vid+'img/', filename))
    frames.append(frame)

frames = np.array(frames)


for i in range(len(frames)):
    frame = frames[i]
    start_point = (groundtruth_rect[i][0], groundtruth_rect[i][1])
    end_point = (groundtruth_rect[i][0]+groundtruth_rect[i][2], groundtruth_rect[i][1]+groundtruth_rect[i][3])
    frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
    cv2.imshow("template tracking", frame)
    k = cv2.waitKey(0)
    if k==27:
        break

cv2.destroyAllWindows()


