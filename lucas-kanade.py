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
    d0 = np.expand_dims(np.dot(coord_n,a[0]),axis=-1)
    d1 = np.expand_dims(np.dot(coord_n,a[1]),axis=-1)
    d2 = np.expand_dims(np.dot(coord_n,a[2]),axis=-1)

    J0_012 = coord_n/d2
    J0_345 = np.zeros_like(J0_012)
    J0_678 = -coord_n*d0/d2**2

    J1_012 = np.zeros_like(J0_012)
    J1_345 = coord_n/d2
    J1_678 = -coord_n*d1/d2**2

    # row 1
    r1 = np.concatenate([J0_012,J0_345,J0_678],axis=2)
    # row 2
    r2 = np.concatenate([J1_012,J1_345,J1_678],axis=2)
    # final
    return np.stack([r1,r2],axis=2)