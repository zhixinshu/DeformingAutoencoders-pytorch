import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path
#import scipy.io
import random
from torchvision import transforms, utils
import string
#import zx

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

IMG_JPGS = ['.jpg', '.jpeg', '.JPG', '.JPEG']
IMG_PNGS = ['.png', '.PNG']

NUMPY_EXTENSIONS = ['.npy', '.NPY']


def duplicates(lst, item, match = True):
    if match:
        return [i for i, x in enumerate(lst) if x == item]
    else:
        return [i for i, x in enumerate(lst) if not x == item]

def intersect(*d):
    sets = iter(map(set, d))
    result = sets.next()
    for s in sets:
        result = result.intersection(s)
    return result

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

###
def resize_loader(imgPath0, rgb=True, resize=64):
    with open(imgPath0, 'rb') as f0:
        with Image.open(f0) as img0:
            if rgb:
                img0 = img0.convert('RGB')
            if resize:
                img0 = img0.resize((resize, resize),Image.ANTIALIAS)
            img0 = np.array(img0)
    return img0

def make_dataset_singlefolder(dirpath_root):
    img_list = []     # list of path to the images
    print(dirpath_root)
    assert os.path.isdir(dirpath_root)
    for root, _, fnames in sorted(os.walk(dirpath_root)):
        for fname in fnames:
            if is_image_file(fname):
                path_img = os.path.join(root, fname)
                img_list.append(path_img)
    return img_list

class DAEImageFolderResize(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, transform=None, return_paths=False, rgb = True, resize = 64, loader=resize_loader):
        super(DAEImageFolderResize, self).__init__()
        imgs = make_dataset_singlefolder(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.length = len(imgs)
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader 
        self.rgb = rgb
        self.resize = resize

    def __getitem__(self, index):
        imgPath0 = self.imgs[index]
        img0 = self.loader(imgPath0, rgb = self.rgb, resize = self.resize)
        return img0

    def __len__(self):
        return len(self.imgs)
