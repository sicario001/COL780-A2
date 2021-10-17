import numpy as np

def warp(m,n,p):
    # p should be 3x3
    
    coord_n = np.array([[[i,j,1] for j in range(n)] for i in range(m)])
    
    d = np.dot(coord_n,p[2])
    x = np.dot(coord_n,p[0])/d
    y = np.dot(coord_n,p[1])/d
    f = np.stack([x,y],axis=2)
    return np.round(f).astype('int32')


def getDiff(prev_frame,curr_frame,t_mask,p):
    # t_mask is list of coordinates in template, shape n X 2
    warped_coord = warp(prev_frame.shape[0],prev_frame.shape[1],p)
    # TODO: round off?
    curr_t_mask = warped_coord[t_mask[:,0],t_mask[:,1]]
    print(curr_t_mask)
    diff = prev_frame[t_mask[:,0],t_mask[:,1]] - curr_frame[curr_t_mask[:,0],curr_t_mask[:,1]]
    return diff

def getDelta(m,n,a):
    coord_n = np.array([[[i,j,1] for j in range(n)] for i in range(m)])
    # for x,y return 2d matrix 
    # [
    #   a11x+a12y+a13   a21x+a22y+a23
    #   a31x+a32y+a33   a41x+a42y+a43
    # ]
    
    # cell 11
    c11 = np.dot(coord_n,a[0])
    # cell 12
    c12 = np.dot(coord_n,a[1])
    # cell 21
    c21 = np.dot(coord_n,a[2])
    # cell 22
    c22 = np.dot(coord_n,a[3])
    # row 1
    r1 = np.stack([c11,c12],axis=2)
    # row 2
    r2 = np.stack([c21,c22],axis=2)
    # final
    return np.stack([r1,r2],axis=2)