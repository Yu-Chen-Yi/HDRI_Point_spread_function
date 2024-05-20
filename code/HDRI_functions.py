''' HDR flow '''
import numpy as np
import os
import cv2 as cv

Z_max = 255
Z_min = 0
gamma = 2.2

def ReadImg(path, flag=1):
    img = cv.imread(path, flag)  # flag = 1 means to load a color image
    img = img[:,:,[2,1,0]]
    return img


def SaveImg(img, path):
    img = img[:,:,[2,1,0]]
    # 取得資料夾的路徑
    folder = os.path.dirname(path)
    
    # 如果資料夾不存在，則創建資料夾
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv.imwrite(path, img)
    

def LoadExposures(source_dir):
    """ load bracketing images folder

    Args:
        source_dir (string): folder path containing bracketing images and a image_list.txt file
                             image_list.txt contains lines of image_file_name, exposure time, ... 
    Returns:
        img_list (uint8 ndarray, shape (N, height, width, ch)): N bracketing images (3 channel)
        exposure_times (list of float, size N): N exposure times
    """
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *_) = line.split()
        filenames += [filename]
        exposure_times += [float(exposure)]
    img_list = [ReadImg(os.path.join(source_dir, f)) for f in filenames]
    img_list = np.array(img_list)
    
    return img_list, exposure_times


def PixelSample(img_list):
    """ Sampling

    Args:
        img_list (uint8 ndarray, shape (N, height, width, ch))
        
    Returns:
        sample (uint8 ndarray, shape (N, height_sample_size, width_sample_size, ch))
    """
    # trivial periodic sample
    sample = img_list[:, ::32, ::32, :]
    
    return sample
    


def EstimateResponse(img_samples, etime_list, lambda_=50):
    """ Estimate camera response for bracketing images

    Args:
        img_samples (uint8 ndarray, shape (N, height_sample_size, width_sample_size)): N bracketing sampled images (1 channel)
        etime_list (list of float, size N): N exposure times
        lambda_ (float): Lagrange multiplier (Defaults to 50)
    
    Returns:
        response (float ndarray, shape (256)): response map
    """
    
    ''' TODO '''
    etime_list = np.array(etime_list, dtype=np.float32)
    img_samples = np.reshape(img_samples, (np.size(img_samples, 0), np.size(img_samples, 1)*np.size(img_samples, 2)))
    
    n = 256
    A = np.zeros((np.size(img_samples, 0) * np.size(img_samples, 1) + n + 1, n + np.size(img_samples, 1)), dtype=np.float32)
    b = np.zeros((np.size(A, 0), 1), dtype=np.float32)
    w = [z if z <= 0.5*Z_max else Z_max-z for z in range(256)]

    k = 0
    for i in range(np.size(img_samples, 1)):
        for j in range(np.size(img_samples, 0)):
            z = int(img_samples[j][i])
            wij = w[z]
            A[k][z] = wij
            A[k][n+i] = -wij
            b[k] = wij * np.log(etime_list[j])
            k += 1
    
    A[k][128] = 1
    k += 1

    for i in range(n-1):
        A[k][i]   =      lambda_ * w[i+1]
        A[k][i+1] = -2 * lambda_ * w[i+1]
        A[k][i+2] =      lambda_ * w[i+1]
        k += 1

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    response = x[:256].reshape(256,)
    #lE = x[256:]
    return response


def ConstructRadiance(img_list, response, etime_list):
    """ Construct radiance map from brackting images

    Args:
        img_list (uint8 ndarray, shape (N, height, width)): N bracketing images (1 channel)
        response (float ndarray, shape (256)): response map
        etime_list (list of float, size N): N exposure times
    
    Returns:
        radiance (float ndarray, shape (height, width)): radiance map
    """

    etime_list = np.array(etime_list, dtype=np.float32)
    img_list = img_list.astype(np.float32)
    img_size = img_list[0].shape
    w = [z if z <= 0.5*Z_max else Z_max-z for z in range(256)]
    ln_t = np.log(etime_list)

    radiance = np.zeros_like(img_list[0], dtype=np.float32)

    Z = [img.flatten().tolist() for img in img_list]
    acc_E = [0]*len(Z[0])
    ln_E = [0]*len(Z[0])

    vfunc = np.vectorize(lambda x:np.exp(x))
    pixels, imgs = len(Z[0]), len(Z)
    for i in range(pixels):
        acc_w = 0
        for j in range(imgs):
            z = int(Z[j][i])
            acc_E[i] += w[z]*(response[z] - ln_t[j])
            acc_w += w[z]
        ln_E[i] = acc_E[i]/acc_w if acc_w > 0 else acc_E[i]
        acc_w = 0

    # Exponational each channels and reshape to 2D-matrix
    radiance = np.reshape(vfunc(ln_E), img_size)

    return radiance

def CameraResponseCalibration(src_path, lambda_):
    img_list, exposure_times = LoadExposures(src_path)
    radiance = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_samples = PixelSample(img_list)
    for ch in range(3):
        response = EstimateResponse(pixel_samples[:, :,:, ch], exposure_times, lambda_)
        radiance[:,:,ch] = ConstructRadiance(img_list[:,:,:,ch], response, exposure_times)

    return radiance