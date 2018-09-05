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

def DefaultAxisRotate(R):
    DefaultR = torch.Tensor(4,4).fill_(0)
    DefaultR[0,0] = -1
    DefaultR[1,1] = 1
    DefaultR[2,2] = -1
    DefaultR[3,3] = 1
    return torch.mm(DefaultR,R)


def intersect(*d):
    sets = iter(map(set, d))
    result = sets.next()
    for s in sets:
        result = result.intersection(s)
    return result

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_jpeg_file(filename):
    return any(filename.endswith(extension) for extension in IMG_JPGS)

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in IMG_PNGS)

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def parse_imgfilename_Wasp(fn):
    ids = fn[str.rfind(fn,'_ids_')+5 : str.rfind(fn,'_ide_')]
    ide = fn[str.rfind(fn,'_ide_')+5 : str.rfind(fn,'_idp_')]
    idp = fn[str.rfind(fn,'_idp_')+5 : str.rfind(fn,'_idt_')]
    idt = fn[str.rfind(fn,'_idt_')+5 : str.rfind(fn,'_idl_')]
    idl = fn[str.rfind(fn,'_idl_')+5 : -4]
    return ids, ide, idp, idt , idl

def make_dataset_Wasp_singlefolder(dirpath_root):
    img_list = []     # list of path to the images
    print(dirpath_root)
    assert os.path.isdir(dirpath_root)
    for root, _, fnames in sorted(os.walk(dirpath_root)):
        for fname in fnames:
            if is_image_file(fname):
                path_img = os.path.join(root, fname)
                img_list.append(path_img)
    return img_list

def make_dataset_Wasp_singlefolder_jpeg(dirpath_root):
    img_list = []     # list of path to the images
    print(dirpath_root)
    assert os.path.isdir(dirpath_root)
    for root, _, fnames in sorted(os.walk(dirpath_root)):
        for fname in fnames:
            if is_jpeg_file(fname):
                path_img = os.path.join(root, fname)
                img_list.append(path_img)
    return img_list

def dict_to_list_depth2(x):
    y = []
    for key, vs in x.items():
        for v in vs:
            y.append([key, v])
    return y

def make_dataset_Wasp_multifolder(label_path_dict):
    img_dicts = {}
    for label, path in label_path_dict.items():
        print label
        print path
        img_dicts[label] =  make_dataset_Wasp_singlefolder(path)
        if len(img_dicts[label]) == 0:
            raise(RuntimeError("Found 0 images in: " + path + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    img_lists = dict_to_list_depth2(img_dicts)
    return img_lists

def default_loader(imgPath):
    with open(imgPath, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            img = np.array(img)
    return img

def default_loader_resize(imgPath, resize=64):
    with open(imgPath, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            if resize:
                img = img.resize((resize, resize),Image.ANTIALIAS)
            img = np.array(img)
    return img

def default_loader_resize_with_mask(imgPath, maskPath, resize=64):
    with open(imgPath, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            if resize:
                img = img.resize((resize, resize),Image.ANTIALIAS)
            img = np.array(img)
    with open(maskPath, 'rb') as g:
        with Image.open(g) as mask:
            mask = mask.convert('RGB')
            if resize:
                mask = mask.resize((resize, resize),Image.ANTIALIAS)
            mask = np.array(mask) 
    return img, mask

def default_loader_pair(imgPath0, imgPath1):
    with open(imgPath0, 'rb') as f0:
        with Image.open(f0) as img0:
            #img0 = img0.convert('RGB')
            img0 = np.array(img0)
    with open(imgPath1, 'rb') as f1:
        with Image.open(f1) as img1:
            #img1 = img1.convert('RGB')
            img1 = np.array(img1)
    return img0, img1



def default_loader_triplet(imgPath0, imgPath1, imgPath2):
    with open(imgPath0, 'rb') as f0:
        with Image.open(f0) as img0:
            #img0 = img0.convert('RGB')
            img0 = np.array(img0)

    with open(imgPath1, 'rb') as f1:
        with Image.open(f1) as img1:
            #img1 = img1.convert('RGB')
            img1 = np.array(img1)

    with open(imgPath2, 'rb') as f2:
        with Image.open(f2) as img2:
            #img1 = img1.convert('RGB')
            img2 = np.array(img2)

    return img0, img1, img2


class WaspImageFolderJPEG(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, transform=None, return_paths=False, resize = 64, loader=default_loader_resize):
        super(WaspImageFolderJPEG, self).__init__()
        imgs = make_dataset_Wasp_singlefolder_jpeg(root)
    
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_JPGS)))

        self.root = root
        self.imgs = imgs
        self.return_paths = return_paths
        self.loader = loader 
        self.resize = resize
        self.transform = transform

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        img = self.loader(imgPath, resize=self.resize)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

class WaspImageFolderPNG(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, transform=None, return_paths=False, resize = 64, loader=default_loader_resize):
        super(WaspImageFolderPNG, self).__init__()
        imgs = make_dataset_Wasp_singlefolder_jpeg(root)
    
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_PNGS)))

        self.root = root
        self.imgs = imgs
        self.return_paths = return_paths
        self.loader = loader 
        self.resize = resize
        self.transform = transform

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        img = self.loader(imgPath, resize=self.resize)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

class WaspImageFolderJPEGWithMaskPNG(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, transform=None, return_paths=False, resize = 64, loader=default_loader_resize_with_mask):
        super(WaspImageFolderJPEGWithMaskPNG, self).__init__()
        imgs = make_dataset_Wasp_singlefolder_jpeg(root)
    
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_JPGS)))

        self.root = root
        self.imgs = imgs
        self.return_paths = return_paths
        self.loader = loader 
        self.resize = resize
        self.transform = transform

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        maskPath = imgPath
        if imgPath.endswith('.jpg'):
            maskPath=imgPath[:-4]+'.png'
        elif imgPath.endswith('.jpeg'):
            maskPath=imgPath[:-5]+'.png'
        else:
            maskPath = imgPath
        img, mask = self.loader(imgPath,maskPath , resize=self.resize)
        return img, mask

    def __len__(self):
        return len(self.imgs)


class WaspImageFolder(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        super(WaspImageFolder, self).__init__()
        imgs = make_dataset_Wasp_singlefolder(root)
    
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader 

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        img = self.loader(imgPath)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

class WaspImageFolderPair(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, transform=None, return_paths=False, loader=default_loader_pair):
        super(WaspImageFolderPair, self).__init__()
        imgs = make_dataset_Wasp_singlefolder(root)
    
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.length = len(imgs)
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader 

    def __getitem__(self, index):
        imgPath0 = self.imgs[index]
        coindex = random.sample(range(self.length),1)[0]
        imgPath9 = self.imgs[coindex]
        img0, img9 = self.loader(imgPath0, imgPath9)
        if self.transform is not None:
            img0 = self.transform(img0)
            img9 = self.transform(img9)
        return img0, img9

    def __len__(self):
        return len(self.imgs)


class WaspImageFolderMultilabelPairWithinlabels(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, info_dict, transform=None, return_paths=False, loader=default_loader_pair):
        super(WaspImageFolderMultilabelPairWithinlabels, self).__init__()
        self.dps = make_dataset_Wasp_multifolder(info_dict)
        self.info_dict = info_dict
        self.length = len(self.dps)
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader 

    def __getitem__(self, index):
        label0 = self.dps[index][0]
        imgPath0 = self.dps[index][1]
        coindex = random.sample(range(self.length),1)[0]
        label9 = self.dps[coindex][0]
        while not label9 == label0:
            coindex = random.sample(range(self.length),1)[0]
            label9 = self.dps[coindex][0]
        imgPath9 = self.dps[coindex][1]
        img0, img9 = self.loader(imgPath0, imgPath9)
        if self.transform is not None:
            img0 = self.transform(img0)
            img9 = self.transform(img9)
        return img0, label0, img9, label9

    def __len__(self):
        return len(self.dps)



class WaspImageFolderMultilabelPairShufflelabels(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, info_dict, transform=None, return_paths=False, loader=default_loader_pair):
        super(WaspImageFolderMultilabelPairShufflelabels, self).__init__()
        self.dps = make_dataset_Wasp_multifolder(info_dict)
        self.info_dict = info_dict
        self.length = len(self.dps)
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader 
    def __getitem__(self, index):
        label0 = self.dps[index][0]
        imgPath0 = self.dps[index][1]
        coindex = random.sample(range(self.length),1)[0]
        label9 = self.dps[coindex][0]
        imgPath9 = self.dps[coindex][1]
        img0, img9 = self.loader(imgPath0, imgPath9)
        if self.transform is not None:
            img0 = self.transform(img0)
            img9 = self.transform(img9)
        return img0, label0, img9, label9

    def __len__(self):
        return len(self.dps)


def resize_loader_pair(imgPath0, imgPath1, rgb=True, resize=64):
    with open(imgPath0, 'rb') as f0:
        with Image.open(f0) as img0:
            if rgb:
                img0 = img0.convert('RGB')
            if resize:
                img0 = img0.resize((resize, resize),Image.ANTIALIAS)
            img0 = np.array(img0)
    with open(imgPath1, 'rb') as f1:
        with Image.open(f1) as img1:
            if rgb:
                img1 = img1.convert('RGB')
            if resize:
                img1 = img1.resize((resize, resize),Image.ANTIALIAS)
            img1 = np.array(img1)
    return img0, img1

def resize_loader_pair_flip(imgPath0, rgb=True, resize=64):
    with open(imgPath0, 'rb') as f0:
        with Image.open(f0) as img0:
            if rgb:
                img0 = img0.convert('RGB')
            if resize:
                img0 = img0.resize((resize, resize),Image.ANTIALIAS)
            img1 = img0.transpose(Image.FLIP_LEFT_RIGHT)
            img0 = np.array(img0)
            img1 = np.array(img1)
    return img0, img1

def resize_loader(imgPath0, rgb=True, resize=64):
    with open(imgPath0, 'rb') as f0:
        with Image.open(f0) as img0:
            if rgb:
                img0 = img0.convert('RGB')
            if resize:
                img0 = img0.resize((resize, resize),Image.ANTIALIAS)
            img0 = np.array(img0)
    return img0

class WaspImageFolderPairResize(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, transform=None, return_paths=False, rgb = True, resize = 64, loader=resize_loader_pair):
        super(WaspImageFolderPairResize, self).__init__()
        imgs = make_dataset_Wasp_singlefolder(root)
    
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
        coindex = random.sample(range(self.length),1)[0]
        imgPath9 = self.imgs[coindex]
        img0, img9 = self.loader(imgPath0, imgPath9, rgb = self.rgb, resize = self.resize)
        if self.transform is not None:
            img0 = self.transform(img0)
            img9 = self.transform(img9)
        return img0, img9

    def __len__(self):
        return len(self.imgs)


class WaspImageFolderPairResize_Flip(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, transform=None, return_paths=False, rgb = True, resize = 64, loader=resize_loader_pair_flip):
        super(WaspImageFolderPairResize_Flip, self).__init__()
        imgs = make_dataset_Wasp_singlefolder(root)
    
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
        img0, img9 = self.loader(imgPath0, rgb = self.rgb, resize = self.resize)
        return img0, img9

    def __len__(self):
        return len(self.imgs)


def resize_loader_pair_MAFLzxWithMask(imgPath0, maskPath0, imgPath1, maskPath1):
    with open(imgPath0, 'rb') as f0:
        with Image.open(f0) as img0:
            img0 = img0.convert('RGB')
            img0 = np.array(img0)
    with open(maskPath0, 'rb') as fm0:
        with Image.open(fm0) as mask0:
            mask0 = np.array(mask0)

    with open(imgPath1, 'rb') as f1:
        with Image.open(f1) as img1:
            img1 = img1.convert('RGB')
            img1 = np.array(img1)
    with open(maskPath1, 'rb') as fm1:
        with Image.open(fm1) as mask1:
            mask1 = np.array(mask1)
    return img0, mask0, img1, mask1

def resize_loader_pair_MAFLzxWithMatte(imgPath0, maskPath0, imgPath1, maskPath1):
    with open(imgPath0, 'rb') as f0:
        with Image.open(f0) as img0:
            img0 = img0.convert('RGB')
            img0 = np.array(img0)
    with open(maskPath0, 'rb') as fm0:
        with Image.open(fm0) as mask0:
            mask0 = np.array(mask0)

    with open(imgPath1, 'rb') as f1:
        with Image.open(f1) as img1:
            img1 = img1.convert('RGB')
            img1 = np.array(img1)
    with open(maskPath1, 'rb') as fm1:
        with Image.open(fm1) as mask1:
            mask1 = np.array(mask1)
    return img0, mask0, img1, mask1

def make_dataset_MAFLzxWithMask(dirpath_root):
    mask_list = []
    img_list = []     # list of path to the images
    dirpath_mask_root  = dirpath_root +'_mask/'
    dirpath_root = dirpath_root+'/'
    print(dirpath_root)
    print(dirpath_mask_root)
    assert os.path.isdir(dirpath_root)
    assert os.path.isdir(dirpath_mask_root)
    for root, _, fnames in sorted(os.walk(dirpath_mask_root)):
        for fname in fnames:
            if is_image_file(fname):
                path_mask = os.path.join(root, fname)
                path_img  = os.path.join(dirpath_root,fname[0:-8]+'.png')
                mask_list.append(path_mask)
                img_list.append(path_img)
    return img_list, mask_list

def make_dataset_MAFLzxWithMatte(dirpath_root):
    mask_list = []
    img_list = []     # list of path to the images
    dirpath_mask_root  = dirpath_root +'_matte/'
    dirpath_root = dirpath_root+'/'
    print(dirpath_root)
    print(dirpath_mask_root)
    assert os.path.isdir(dirpath_root)
    assert os.path.isdir(dirpath_mask_root)
    for root, _, fnames in sorted(os.walk(dirpath_mask_root)):
        for fname in fnames:
            if is_image_file(fname):
                path_mask = os.path.join(root, fname)
                path_img  = os.path.join(dirpath_root,fname[0:-8]+'.png')
                mask_list.append(path_mask)
                img_list.append(path_img)
    return img_list, mask_list

class WaspMAFLzxWithMaskPairResize(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, transform=None, return_paths=False, rgb = True, resize = 64, loader=resize_loader_pair_MAFLzxWithMask):
        super(WaspMAFLzxWithMaskPairResize, self).__init__()
        imgs, masks = make_dataset_MAFLzxWithMask(root)
    
        if len(masks) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.length = len(imgs)
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader 
        self.rgb = rgb
        self.resize = resize

    def __getitem__(self, index):
        maskPath0 = self.masks[index]
        imgPath0 = self.imgs[index]
        coindex = random.sample(range(self.length),1)[0]
        maskPath9 = self.masks[coindex]
        imgPath9 = self.imgs[coindex]
        img0, mask0, img9, mask9 = self.loader(imgPath0, maskPath0, imgPath9, maskPath9)
        #if self.transform is not None:
        #    img0 = self.transform(img0)
        #    img9 = self.transform(img9)
        return img0, mask0, img9, mask9

    def __len__(self):
        return len(self.masks)





class WaspMAFLzxWithMattePairResize(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, transform=None, return_paths=False, rgb = True, resize = 64, loader=resize_loader_pair_MAFLzxWithMatte):
        super(WaspMAFLzxWithMattePairResize, self).__init__()
        imgs, masks = make_dataset_MAFLzxWithMatte(root)
    
        if len(masks) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.length = len(imgs)
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader 
        self.rgb = rgb
        self.resize = resize

    def __getitem__(self, index):
        maskPath0 = self.masks[index]
        imgPath0 = self.imgs[index]
        coindex = random.sample(range(self.length),1)[0]
        maskPath9 = self.masks[coindex]
        imgPath9 = self.imgs[coindex]
        img0, mask0, img9, mask9 = self.loader(imgPath0, maskPath0, imgPath9, maskPath9)
        #if self.transform is not None:
        #    img0 = self.transform(img0)
        #    img9 = self.transform(img9)
        return img0, mask0, img9, mask9

    def __len__(self):
        return len(self.masks)


##########################################
#####################################
################################
## The font data loader
################################
#####################################
##########################################

class WaspImageFolderFontWithinlabels(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, info_dict, transform=None, return_paths=False, loader=default_loader_pair):
        super(WaspImageFolderFontWithinlabels, self).__init__()
        self.dps = make_dataset_Wasp_multifolder(info_dict)
        self.info_dict = info_dict
        self.length = len(self.dps)
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader 

    def __getitem__(self, index):
        label0 = self.dps[index][0]
        imgPath0 = self.dps[index][1]
        coindex = random.sample(range(self.length),1)[0]
        label9 = self.dps[coindex][0]
        while not label9 == label0:
            coindex = random.sample(range(self.length),1)[0]
            label9 = self.dps[coindex][0]
        imgPath9 = self.dps[coindex][1]
        img0, img9 = self.loader(imgPath0, imgPath9)
        if self.transform is not None:
            img0 = self.transform(img0)
            img9 = self.transform(img9)
        return img0, label0, img9, label9

    def __len__(self):
        return len(self.dps)



class WaspImageFolderFontWithinlabelsTriplet(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, info_dict, li = [1,2,3], ali = ['A','B','C'], transform=None, return_paths=False, loader=default_loader_triplet):
        super(WaspImageFolderFontWithinlabelsTriplet, self).__init__()
        self.dps = make_dataset_Wasp_multifolder(info_dict)
        self.info_dict = info_dict
        self.length = len(self.dps)
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader 
        self.li = li
        self.ali = ali

    def __getitem__(self, index):
        label0 = self.dps[index][0]
        imgPath0 = self.dps[index][1]
        coindex = random.sample(range(self.length),1)[0]
        label9 = self.dps[coindex][0]
        while not label9 == label0:
            coindex = random.sample(range(self.length),1)[0]
            label9 = self.dps[coindex][0]
        imgPath9 = self.dps[coindex][1]

        coindex2 = random.sample(range(len(self.li)),1)[0]
        label7 = self.li[coindex2]
        while label7 == label0:
            coindex2 = random.sample(range(len(self.li)),1)[0]
            label7 = self.li[coindex2]

        tag0 = 'Exp_font_' + self.ali[label0] + '_sub_'
        tag7 = 'Exp_font_' + self.ali[label7] + '_sub_'

        imgPath7 = string.replace(imgPath0, tag0, tag7)

        img0, img9, img7 = self.loader(imgPath0, imgPath9, imgPath7)
        if self.transform is not None:
            img0 = self.transform(img0)
            img9 = self.transform(img9)
            img7 = self.transform(img7)
        return img0, label0, img9, label9, img7, label7

    def __len__(self):
        return len(self.dps)

##########################################
#####################################
################################
## The CelebA data loader
################################
#####################################
##########################################

def default_loader_img(imgPath):
    with open(imgPath, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            img = np.array(img)
    return img


def default_loader_csv(csvPath):
    numbers = np.genfromtxt(csvPath, delimiter=',') 
    return numbers


class WaspAugmentedCelebA2(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, load_all = False, transform=None, return_paths=False, img_loader=default_loader_img, csv_loader = default_loader_csv):
        super(WaspAugmentedCelebA2, self).__init__()
        self.load_all = load_all
        self.root = root
        self.path_img = self.root + '/crop_resize'
        self.path_normal = self.root + '/normal/crop_resize/'
        self.path_coord = self.root + '/coord/crop_resize/'
        self.path_mask = self.root + '/mask/crop_resize/'
        self.path_matte = self.root + '/matte_crop_resize/'
        self.path_itnormal = self.root + '/itnormal/crop_resize/'
        self.path_Lvecs = self.root + '/Lvecs/'
        self.imgs, self.normals, self.coords, self.masks, self.mattes, self.itnormals, self.Lvecs = self.make_dataset()

        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.transform = transform
        self.return_paths = return_paths
        self.img_loader = img_loader 
        self.csv_loader = csv_loader


    def make_dataset(self):
        img_list = []     # list of path to the images
        normal_list = []
        coord_list = []
        mask_list = []
        matte_list = []
        itnormal_list = []
        Lvecs_list = []
        for root, _, fnames in sorted(os.walk(self.path_normal)):
            for fname in fnames:
                if is_image_file(fname):
                    path_normal   = os.path.join(self.path_normal, fname)
                    path_img      = os.path.join(self.path_img, fname)
                    path_coord    = os.path.join(self.path_coord, fname)
                    path_mask     = os.path.join(self.path_mask, fname)
                    path_matte    = os.path.join(self.path_matte, fname)
                    path_itnormal = os.path.join(self.path_itnormal, fname) 
                    path_Lvecs    = os.path.join(self.path_Lvecs, fname[:-4]+'.csv')
                    img_list.append(path_img)
                    normal_list.append(path_normal)
                    coord_list.append(path_coord)
                    mask_list.append(path_mask)
                    matte_list.append(path_matte)
                    itnormal_list.append(path_itnormal)
                    Lvecs_list.append(path_Lvecs)

        return img_list, normal_list, coord_list, mask_list, matte_list, itnormal_list, Lvecs_list

    def __getitem__(self, index):
        if self.load_all:
            imgPath      = self.imgs[index]
            normalPath   = self.normals[index]
            coordPath    = self.coords[index]
            maskPath     = self.masks[index]
            mattePath    = self.mattes[index]
            itnormalPath = self.itnormals[index]
            LvecPath     = self.Lvecs[index]

            img      = self.img_loader(imgPath)  
            normal   = self.img_loader(normalPath)  
            coord    = self.img_loader(coordPath)  
            mask     = self.img_loader(maskPath)   
            matte    = self.img_loader(mattePath)  
            itnormal = self.img_loader(itnormalPath)
            Lvec     = self.csv_loader(LvecPath)

            if self.transform is not None:
                img = self.transform(img)
            return img, normal, coord, mask, matte, itnormal, Lvec
        else:
            imgPath = self.imgs[index]
            img = self.img_loader(imgPath)
            if self.transform is not None:
                img = self.transform(img)
            return img

    def __len__(self):
        return len(self.imgs)


class WaspImageFolderWithRandomNormalsAndLights(data.Dataset):
    # an object that iterates over an image folder
    def __init__(self, root, path_normals, path_Lvecs, transform=None, return_paths=False, rgb = True, resize = 64, loader=resize_loader):
        super(WaspImageFolderWithRandomNormalsAndLights, self).__init__()
        imgs = make_dataset_Wasp_singlefolder(root)
        self.path_normal = path_normals
        self.path_Lvecs = path_Lvecs
        self.normals, self.Lvecs = self.make_dataset()
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
        self.img_loader = default_loader_img 
        self.csv_loader = default_loader_csv

    def make_dataset(self):
        normal_list = []
        Lvecs_list = []
        for root, _, fnames in sorted(os.walk(self.path_normal)):
            for fname in fnames:
                if is_image_file(fname):
                    path_normal   = os.path.join(self.path_normal, fname)
                    path_Lvecs    = os.path.join(self.path_Lvecs, fname[:-4]+'.csv')
                    normal_list.append(path_normal)
                    Lvecs_list.append(path_Lvecs)
        return normal_list, Lvecs_list

    def __getitem__(self, index):
        imgPath0 = self.imgs[index]
        img0 = self.loader(imgPath0, rgb = self.rgb, resize = self.resize)
        index2 = index
        if index2> 9600:
            index2 = index -1000
        normalPath   = self.normals[index2]
        LvecPath     = self.Lvecs[index2]
        normal0   = self.img_loader(normalPath)
        Lvec0     = self.csv_loader(LvecPath)

        if self.transform is not None:
            img0 = self.transform(img0)
        return img0, normal0, Lvec0

    def __len__(self):
        return len(self.imgs)