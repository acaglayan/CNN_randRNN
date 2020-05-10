import warnings
from itertools import product

import numpy as np
import numpy.matlib as npm
from PIL import Image
from scipy import signal

"""
# This function fills out the missing depth values (0s) in a depth image of the Kinect sensor.
# The missing zeros are filled out by using an iterative process where each
# missing pixel is calculated by the mean of the 5X5 pixel window around it.
# It takes depth image and the corresponding mask image as inputs and returns interpolated filtered result. 
"""


def depth_interpolation(depth_img):
    depth_img = np.asarray(depth_img)
    filtered = depth_img.copy()

    tolerance = 0.01
    missings = np.where(depth_img == 0)
    miss_x, miss_y = missings[0], missings[1]

    dist_bound = np.zeros(np.size(miss_x))

    i = np.meshgrid(np.arange(len(miss_x)), sparse=True)
    dist_bound = np.fmin(np.fmin(miss_x[i], depth_img.shape[0] - miss_x[i]),
                         np.fmin(miss_y[i], depth_img.shape[1] - miss_y[i]))

    stacked = np.vstack((miss_x, miss_y, dist_bound)).transpose()
    sorted_stack = stacked[stacked[:, 2].argsort(),]  # Sort rows (by third row)
    miss_x = sorted_stack[:, 0]
    miss_y = sorted_stack[:, 1]

    iter = 0
    diff_sum = np.Inf

    while diff_sum > tolerance and iter < 10:
        diff_sum = 0

        for i in range(len(miss_x)):
            cur_x = miss_x[i]
            cur_y = miss_y[i]

            a_x = np.fmax(0, cur_x - 2)
            b_x = np.fmin(depth_img.shape[0], cur_x + 2)
            a_y = np.fmax(0, cur_y - 2)
            b_y = np.fmin(depth_img.shape[1], cur_y + 2)

            window = filtered[a_x: b_x + 1, a_y: b_y + 1]

            mask_zero_depth = (window != np.asarray(0))
            non_z_window = np.ma.masked_array(window, mask_zero_depth)
            non_z_window = window[np.where(non_z_window > 0)]
            if not (np.all(non_z_window == 0)):
                avg_val = np.median(non_z_window.ravel())
                diff_sum = diff_sum + np.abs(filtered[cur_x, cur_y] - avg_val) / avg_val
                filtered[cur_x, cur_y] = avg_val
        iter += 1

    filtered = np.uint16(filtered)
    return filtered


def fast_depth_interpolation(depth_img):
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # skip the 'All-NaN slice encountered' warning
    depth_img = np.asarray(depth_img, dtype=np.float32)
    filtered = depth_img.copy()

    missings = np.array(np.where(depth_img == 0))

    window = npm.repmat([-2, -1, 0, 1, 2], missings.shape[1], 1)

    missing_rows = npm.repmat(missings[0], 25, 1)
    missing_cols = npm.repmat(missings[1], 25, 1)

    while True:
        row_bounds = missings[0, :, np.newaxis] + window
        row_bounds = np.where(row_bounds >= depth_img.shape[0], -1, row_bounds)  # check for the boundary
        col_bounds = missings[1, :, np.newaxis] + window
        col_bounds = np.where(col_bounds >= depth_img.shape[1], -1, col_bounds)  # check for the boundary

        missing_bounds = np.array(list(product(row_bounds.transpose(), col_bounds.transpose())))
        missing_bounds = np.transpose(missing_bounds, (1, 0, 2))

        if np.count_nonzero(missing_bounds < 0) > 0:
            cond_rows = (missing_bounds[0] < 0)
            cond_cols = (missing_bounds[1] < 0)
            (missing_bounds[0])[cond_rows] = missing_rows[cond_rows]
            (missing_bounds[1])[cond_rows] = missing_cols[cond_rows]
            (missing_bounds[0])[cond_cols] = missing_rows[cond_cols]
            (missing_bounds[1])[cond_cols] = missing_cols[cond_cols]

        missing_wins = filtered[missing_bounds[0], missing_bounds[1]]

        # to be able to skip zeros inside win, we use nanmedian and for that first initialize zeros with nan
        missing_wins[missing_wins == 0] = np.nan
        median_of_nonzeros = np.nanmedian(missing_wins, axis=0)

        # if all values are nan then make it zero for the next iteration
        median_of_nonzeros[np.isnan(median_of_nonzeros)] = 0
        filtered[missings[0], missings[1]] = median_of_nonzeros

        if np.count_nonzero(filtered == 0) == 0:
            break

    filtered = np.uint16(filtered)
    return filtered


""" 
# Convert depth image into 3D point cloud
# Original source: depthToCloud.m - Liefeng Bo and Kevin Lai
# Numpy version: acaglayan
# Input: 
# depth - the depth image
# topleft - the position of the top-left corner of depth in the original depth image. 
# Assumes depth is uncropped if this is not provided
#
# Output:
# pcloud - the point cloud, where each channel is the x, y, and z euclidean coordinates respectively. 
# Missing values are NaN.
# distance - euclidean distance from the sensor to each point
"""


def depth_to_pcl(depth, top_left=(1, 1)):
    depth = np.asarray(depth)
    depth = np.where(depth == 0, np.nan, depth)

    # RGB - D camera constants
    center = (320, 240)
    imh, imw = depth.shape
    constant = 570.3
    MM_PER_M = 1000
    cam_constant = constant * MM_PER_M

    pcloud = np.zeros(shape=(imh, imw, 3))
    xgrid = np.matmul(np.ones(shape=(imh, 1)), np.expand_dims(np.arange(imw), axis=0)) + \
            top_left[0] - center[0]
    ygrid = np.matmul(np.expand_dims(np.arange(imh), axis=0).transpose(), np.ones(shape=(1, imw))) + \
            top_left[1] - center[1]
    pcloud[:, :, 0] = np.multiply(xgrid, np.divide(depth, cam_constant))
    pcloud[:, :, 1] = np.multiply(ygrid, np.divide(depth, cam_constant))
    pcloud[:, :, 2] = np.divide(depth, MM_PER_M)
    # distance = np.sqrt(np.sum(np.square(pcloud), axis=2))

    return pcloud


def depth_to_pcl_sunrgbd(depth, sunrgbd_img):
    imsize = np.shape(depth)
    depth_inpaint = (depth >> 3) | (depth << (16 - 3))
    depth_inpaint = np.float32(depth_inpaint) / 1000
    depth_inpaint[depth_inpaint > 8] = 8
    # K is [fx 0 cx; 0 fy cy; 0 0 1];
    # for uncrop image crop =[1, 1];
    cx, cy = sunrgbd_img.K[0, 2], sunrgbd_img.K[1, 2]
    fx, fy = sunrgbd_img.K[0, 0], sunrgbd_img.K[1, 1]

    invalid = depth_inpaint == 0
    x, y = np.meshgrid(range(1, imsize[1] + 1), range(1, imsize[0] + 1))
    x3 = np.multiply((x - cx), depth_inpaint) * 1 / fx
    y3 = np.multiply((y - cy), depth_inpaint) * 1 / fy
    z3 = depth_inpaint

    points_3d_matrix = np.dstack((x3, y3, -z3))
    points_3d_matrix[np.dstack((invalid, invalid, invalid))] = np.NaN
    # points3d = np.squeeze(np.dstack((x3.ravel(), z3.ravel(), -y3.ravel())))
    # points3d[invalid.ravel(), :] = np.NaN

    # points3d = np.float32(np.transpose(np.matmul(sunrgbd_img.Rtilt, np.transpose(points3d))))
    pcloud = np.float32(np.dot(points_3d_matrix, sunrgbd_img.Rtilt.T))

    return pcloud


# Expand x,y,z so the interpolation is valid at the boundaries.
def expand_dim(dim, m, n):
    dim = np.append(np.append(np.array(3 * dim[0, :] - 3 * dim[1, :] + dim[2, :])[np.newaxis], dim, axis=0),
                    np.array(3 * dim[m - 1, :] - 3 * dim[m - 2, :] + dim[m - 3, :])[np.newaxis], axis=0)
    dim = np.append(np.append(np.array(3 * dim[:, 0] - 3 * dim[:, 1] + dim[:, 2])[np.newaxis].transpose(), dim, axis=1),
                    np.array(3 * dim[:, n - 1] - 3 * dim[:, n - 2] + dim[:, n - 3])[np.newaxis].transpose(), axis=1)
    return dim


"""
# Nx, Ny, Nz = surfnorm(pcl) returns the components of the 3-D surface normal for the surface with components (X,Y,Z).  
# The normal is normalized to length 1.

# The surface normals returned are based on a bicubic fit of the data.
# This code is based on the matlab surfnorm function.
# It takes pcl point cloud which is basically x-y-3 (3D) data.
"""


def surfnorm(pcl):
    assert pcl.ndim == 3 and pcl.shape[2] == 3
    x = pcl[:, :, 0]
    y = pcl[:, :, 1]
    z = pcl[:, :, 2]

    m, n = x.shape
    stencil1 = np.divide(np.array([[1], [0], [-1]]), 2).transpose()
    stencil2 = np.divide(np.array([[-1], [0], [1]]), 2)

    xx = x.copy()
    yy = y.copy()
    zz = z.copy()

    # expansion for a valid interpolation at the boundaries
    xx = expand_dim(xx, m, n)
    yy = expand_dim(yy, m, n)
    zz = expand_dim(zz, m, n)

    rows = slice(1, m + 1)
    cols = slice(1, n + 1)

    ax = signal.convolve2d(xx, stencil1, mode='same')
    ax = ax[rows, cols]

    ay = signal.convolve2d(yy, stencil1, mode='same')
    ay = ay[rows, cols]

    az = signal.convolve2d(zz, stencil1, mode='same')
    az = az[rows, cols]

    bx = signal.convolve2d(xx, stencil2, mode='same')
    bx = bx[rows, cols]

    by = signal.convolve2d(yy, stencil2, mode='same')
    by = by[rows, cols]

    bz = signal.convolve2d(zz, stencil2, mode='same')
    bz = bz[rows, cols]

    # perform cross product to get the normals
    nx = -(np.multiply(ay, bz) - np.multiply(az, by))
    ny = -(np.multiply(az, bx) - np.multiply(ax, bz))
    nz = -(np.multiply(ax, by) - np.multiply(ay, bx))

    # Normalize the length of the surface normals to 1.
    mag = np.sqrt(np.multiply(nx, nx) + np.multiply(ny, ny) + np.multiply(nz, nz))
    d_x, d_y = np.nonzero(mag == 0)
    if not (len(d_x) == 0):
        mag[d_x, d_y] = np.dot(np.spacing(1), np.ones(d_x.shape))

    nx_out = np.divide(nx, mag)
    ny_out = np.divide(ny, mag)
    nz_out = np.divide(nz, mag)

    return nx_out, ny_out, nz_out


def colorized_depth(path):
    with open(path, 'rb') as f:
        img = Image.open(f)

        zy, zx = np.gradient(img)
        # Sobel to get a joint Gaussian smoothing and differentiation to reduce noise
        # zx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        # zy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

        normal = np.dstack((-zx, -zy, np.ones_like(img)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n

        # offset and rescale values to be in 0-255
        normal += 1
        normal /= 2
        normal *= 255

        return np.uint8(normal[:, :, ::-1])


def colorized_surfnorm(path):
    end_ind = path.find('depthcrop.png')
    loc_file_name = path[:end_ind] + 'loc.txt'

    img = Image.open(path, 'r')
    loc = np.loadtxt(loc_file_name, delimiter=',', dtype=np.int)

    img = fast_depth_interpolation(img)

    pcl = depth_to_pcl(img, top_left=loc)

    nx, ny, nz = surfnorm(pcl)
    img = np.dstack((nx, ny, nz))

    return img


def colorized_surfnorm_sunrgbd(sunrgbd_img):
    depth_img = Image.open(sunrgbd_img.path, 'r')

    img = fast_depth_interpolation(depth_img)

    pcl = depth_to_pcl_sunrgbd(img, sunrgbd_img)

    nx, ny, nz = surfnorm(pcl)
    img = np.dstack((nx, ny, nz))

    return img
