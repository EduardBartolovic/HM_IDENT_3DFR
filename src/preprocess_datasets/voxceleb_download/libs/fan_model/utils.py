from __future__ import print_function
import os
import sys
import torch
import math
import numpy as np
import cv2

import torchvision.transforms as transforms


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # print(type(image))
    #
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1:
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    # print(type(g))
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1

    # quit()
    return image


def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """

    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = float(resolution / h)
    t[1, 1] = float(resolution / h)
    t[0, 2] = float(resolution * (-center[0] / h + 0.5))
    t[1, 2] = float(resolution * (-center[1] / h + 0.5))

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()


def crop(image, center, scale, resolution=256.0):
    """Center crops an image or set of heatmaps

    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face

    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})

    Returns:
        [type] -- [description]
    """  # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)

    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
    return newImg


def crop_torch(image, center, scale, resolution = 256.0):

    # print(type(image))
    l1 = transform([1, 1], center, scale, resolution, True)
    l2 = transform([resolution, resolution], center, scale, resolution, True)

    newImg = torch.zeros((image.shape[0],image.shape[1], l2[1] - l1[1], l2[0] - l1[0]))
    height, width = image.shape[2],image.shape[3]

    newX = torch.Tensor([max(1, -l1[0] + 1), min(l2[0], width) - l1[0]])
    newY = torch.Tensor([max(1, -l1[1] + 1), min(l2[1], height) - l1[1]])
    oldX = torch.Tensor([max(1, l1[0]  + 1), min(l2[0], width)])
    oldY = torch.Tensor([max(1, l1[1]  + 1), min(l2[1], height)])

    newImg[:,:,int(newY[0].data.item()) - 1:int(newY[1].data.item()),
            int(newX[0].data.item())- 1:int(newX[1].data.item())] = image[:,:,int(oldY[0].data.item()) - 1:int(oldY[1].data.item()),
                 int(oldX[0].data.item()) - 1:int(oldX[1].data.item())]

    # newImg = newImg.resize(resolution,resolution)
    # newImg = torch.nn.functional.interpolate(newImg, size=resolution)

    transformations = transforms.Resize((256,256))
    newImg = transformations(newImg)

    return newImg


def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).add_(1) # edw eixe .floor()

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if 0 < pX < 63 and 0 < pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1], hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), True)

    return preds, preds_orig


def create_target_heatmap(target_landmarks, centers, scales):
    # print(type(target_landmarks))
    # print(centers,scales)
    # quit()
    heatmaps = np.zeros((target_landmarks.shape[0], 68, 64, 64), dtype=np.float32)
    for i in range(heatmaps.shape[0]):
        for p in range(68):
            landmark_cropped_coor = transform(target_landmarks[i, p] + 1, centers[i], scales[i], 64, invert=False)
            heatmaps[i, p] = draw_gaussian(heatmaps[i, p], landmark_cropped_coor + 1, 1)
    return torch.tensor(heatmaps)


def create_bounding_box(target_landmarks, expansion_factor=0.0):
    """
    gets a batch of landmarks and calculates a bounding box that includes all the landmarks per set of landmarks in
    the batch
    :param target_landmarks: batch of landmarks of dim (n x 68 x 2). Where n is the batch size
    :param expansion_factor: expands the bounding box by this factor. For example, a `expansion_factor` of 0.2 leads
    to 20% increase in width and height of the boxes
    :return: a batch of bounding boxes of dim (n x 4) where the second dim is (x1,y1,x2,y2)
    """
    # Calc bounding box
    x_y_min, _ = target_landmarks.reshape(-1, 68, 2).min(dim=1)
    x_y_max, _ = target_landmarks.reshape(-1, 68, 2).max(dim=1)
    # expanding the bounding box
    expansion_factor /= 2
    bb_expansion_x = (x_y_max[:, 0] - x_y_min[:, 0]) * expansion_factor
    bb_expansion_y = (x_y_max[:, 1] - x_y_min[:, 1]) * expansion_factor
    x_y_min[:, 0] -= bb_expansion_x
    x_y_max[:, 0] += bb_expansion_x
    x_y_min[:, 1] -= bb_expansion_y
    x_y_max[:, 1] += bb_expansion_y
    return torch.cat([x_y_min, x_y_max], dim=1)


def shuffle_lr(parts, pairs=None):
    """Shuffle the points left-right according to the axis of symmetry
    of the object.

    Arguments:
        parts {torch.tensor} -- a 3D or 4D object containing the
        heatmaps.

    Keyword Arguments:
        pairs {list of integers} -- [order of the flipped points] (default: {None})
    """
    if pairs is None:
        pairs = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35,
                 34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41,
                 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63,
                 62, 61, 60, 67, 66, 65]
    if parts.ndimension() == 3:
        parts = parts[pairs, ...]
    else:
        parts = parts[:, pairs, ...]

    return parts


def flip(tensor, is_label=False):
    """Flip an image or a set of heatmaps left-right

    Arguments:
        tensor {numpy.array or torch.tensor} -- [the input image or heatmaps]

    Keyword Arguments:
        is_label {bool} -- [denote wherever the input is an image or a set of heatmaps ] (default: {False})
    """
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)

    if is_label:
        tensor = shuffle_lr(tensor).flip(tensor.ndimension() - 1)
    else:
        tensor = tensor.flip(tensor.ndimension() - 1)

    return tensor

# From pyzolib/paths.py (https://bitbucket.org/pyzo/pyzolib/src/tip/paths.py)


def appdata_dir(appname=None, roaming=False):
    """ appdata_dir(appname=None, roaming=False)

    Get the path to the application directory, where applications are allowed
    to write user specific files (e.g. configurations). For non-user specific
    data, consider using common_appdata_dir().
    If appname is given, a subdir is appended (and created if necessary).
    If roaming is True, will prefer a roaming directory (Windows Vista/7).
    """

    # Define default user directory
    userDir = os.getenv('FACEALIGNMENT_USERDIR', None)
    if userDir is None:
        userDir = os.path.expanduser('~')
        if not os.path.isdir(userDir):  # pragma: no cover
            userDir = '/var/tmp'  # issue #54

    # Get system app data dir
    path = None
    if sys.platform.startswith('win'):
        path1, path2 = os.getenv('LOCALAPPDATA'), os.getenv('APPDATA')
        path = (path2 or path1) if roaming else (path1 or path2)
    elif sys.platform.startswith('darwin'):
        path = os.path.join(userDir, 'Library', 'Application Support')
    # On Linux and as fallback
    if not (path and os.path.isdir(path)):
        path = userDir

    # Maybe we should store things local to the executable (in case of a
    # portable distro or a frozen application that wants to be portable)
    prefix = sys.prefix
    if getattr(sys, 'frozen', None):
        prefix = os.path.abspath(os.path.dirname(sys.executable))
    for reldir in ('settings', '../settings'):
        localpath = os.path.abspath(os.path.join(prefix, reldir))
        if os.path.isdir(localpath):  # pragma: no cover
            try:
                open(os.path.join(localpath, 'test.write'), 'wb').close()
                os.remove(os.path.join(localpath, 'test.write'))
            except IOError:
                pass  # We cannot write in this directory
            else:
                path = localpath
                break

    # Get path specific for this app
    if appname:
        if path == userDir:
            appname = '.' + appname.lstrip('.')  # Make it a hidden directory
        path = os.path.join(path, appname)
        if not os.path.isdir(path):  # pragma: no cover
            os.mkdir(path)

    # Done
    return path

def show_landmarks(image, heatmap, pred_landmarks):
    """Show image with pred_landmarks"""
    # pred_landmarks = []
    # pred_landmarks, _ = get_preds_fromhm(torch.from_numpy(heatmap).unsqueeze(0))
    # pred_landmarks = pred_landmarks.squeeze()*4

    # pred_landmarks2 = get_preds_fromhm2(heatmap)
    heatmap = np.max(gt_heatmap, axis=0)
    heatmap = heatmap / np.max(heatmap)
    # image = ski_transform.resize(image, (64, 64))*255
    image = image.astype(np.uint8)
    heatmap = np.max(gt_heatmap, axis=0)
    heatmap = ski_transform.resize(heatmap, (image.shape[0], image.shape[1]))
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    plt.imshow(image)
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], s=0.5, marker='.', c='g')
    plt.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], s=0.5, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated