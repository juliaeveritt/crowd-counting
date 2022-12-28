import numpy as np
import scipy
from scipy import spatial
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt


# Based on https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(img,points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

    return:
    density: the density-map we want. Same shape as input image but only has one channel.

    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt_count.shape))/2./2. #case: 1 point - "MODIFIED FROM gt.shape"
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    print('density:', np.sum(density))
    return density

if __name__=="__main__":

    # Generate ground truth density maps
    # Before creating our own dataset we used Shanghai Tech Part A Dataset to train
    train_path = 'C:\\Users\\julia\\PycharmProjects\\context-aware-crowd-counting\\clark_data\\train\\images\\'
    test_path = 'C:\\Users\\julia\\PycharmProjects\\context-aware-crowd-counting\\clark_data\\test\\images\\'
    paths = [train_path,test_path]

    img_paths = []
    for path in paths:
        print(path)
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
           img_paths.append(img_path)
    print(img_paths)

    for img_path in img_paths:
        # load each image and its associated .mat file
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'gt').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        # Recall the .mat file is 5-dimensional matrix containing R,G,B,A,"points"
        points = mat["image_info"][0, 0][0, 0][0]
        k = gaussian_filter_density(img, points)
        np.save(img_path.replace('.jpg', '.npy').replace('images', 'gt'), k)

