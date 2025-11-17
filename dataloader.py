import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
from testloader import *
from PIL import Image
import numpy as np
from pycocotools import mask
from util.utils import *

class Rain200L_dataset(Dataset):
    def __init__(self, img_size, type, data_dir = './anno/Rain200L'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')

        self.index_list = os.listdir(self.anno_root)
        self.img_size = img_size
        self.anno_root = os.path.join(data_dir,type)
        self.normal_img_root = os.path.join('./data/Rain200L',type,'target')
        self.deg_img_root = os.path.join('./data/Rain200L',type, 'input')

        print('len of index list:',len(self))

    def __getitem__(self, index):
        normal_img_name = os.path.join(self.normal_img_root, os.path.splitext(self.index_list[index//2])[0] +'.png')
        deg_img_name = os.path.join(self.deg_img_root, os.path.splitext(self.index_list[index//2])[0] +'.png')
        
        normal_img_raw = Image.open(normal_img_name)
        normal_img_raw = np.array(normal_img_raw.convert('RGB'))

        deg_img_raw = Image.open(deg_img_name)
        deg_img_raw = np.array(deg_img_raw.convert('RGB'))

        normal_img = transforms.ToTensor()(normal_img_raw)
        deg_img = transforms.ToTensor()(deg_img_raw)
        
        normal_clipfeat = torch.load( os.path.splitext(normal_img_name)[0] + ".pt").to(torch.float32)
        deg_clipfeat = torch.load( os.path.splitext(deg_img_name)[0] + ".pt").to(torch.float32)
        
        anno_path = os.path.join(self.anno_root, self.index_list[index//2])

        # masks = None
        # points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))
        normal_img = size_fix(normal_img, self.img_size)
        deg_img, points, masks = size_fix_all(deg_img, points, masks, self.img_size)

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 1, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) *2
    
class DDN_dataset(Dataset):
    def __init__(self, img_size, type, data_dir = './anno/DDN'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')

        self.index_list = os.listdir(self.anno_root)
        self.img_size = img_size
        self.anno_root = os.path.join(data_dir,type)
        self.normal_img_root = os.path.join('./data/DDN',type,'target')
        self.deg_img_root = os.path.join('./data/DDN', type, 'input')

        print('len of index list:',len(self))

    def __getitem__(self, index):
        normal_img_name = os.path.join(self.normal_img_root, os.path.splitext(self.index_list[index//2])[0] +'.jpg')
        deg_img_name = os.path.join(self.deg_img_root, os.path.splitext(self.index_list[index//2])[0] +'.jpg')
        
        normal_img_raw = Image.open(normal_img_name)
        normal_img_raw = np.array(normal_img_raw.convert('RGB'))

        deg_img_raw = Image.open(deg_img_name)
        deg_img_raw = np.array(deg_img_raw.convert('RGB'))

        normal_img = transforms.ToTensor()(normal_img_raw)
        deg_img = transforms.ToTensor()(deg_img_raw)

        normal_clipfeat = torch.load( os.path.splitext(normal_img_name)[0] + ".pt").to(torch.float32)
        deg_clipfeat = torch.load( os.path.splitext(deg_img_name)[0] + ".pt").to(torch.float32)
        
        anno_path = os.path.join(self.anno_root, self.index_list[index//2])
        # masks = None
        # points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))
        normal_img = size_fix(normal_img, self.img_size)
        deg_img, points, masks = size_fix_all(deg_img, points, masks, self.img_size)

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 2, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) *2

class GoPro_dataset(Dataset):
    def __init__(self, img_size, type, data_dir = './anno/GoPro'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')

        self.index_list = os.listdir(self.anno_root)
        self.img_size = img_size
        self.anno_root = os.path.join(data_dir,type)
        self.normal_img_root = os.path.join('./data/GoPro',type,'target')
        self.deg_img_root = os.path.join('./data/GoPro', type, 'input')

        print('len of index list:',len(self))

    def __getitem__(self, index):
        normal_img_name = os.path.join(self.normal_img_root, os.path.splitext(self.index_list[index//2])[0] +'.png')
        deg_img_name = os.path.join(self.deg_img_root, os.path.splitext(self.index_list[index//2])[0] +'.png')
        
        normal_img_raw = Image.open(normal_img_name)
        normal_img_raw = np.array(normal_img_raw.convert('RGB'))

        deg_img_raw = Image.open(deg_img_name)
        deg_img_raw = np.array(deg_img_raw.convert('RGB'))

        normal_img = transforms.ToTensor()(normal_img_raw)
        deg_img = transforms.ToTensor()(deg_img_raw)

        normal_clipfeat = torch.load( os.path.splitext(normal_img_name)[0] + ".pt").to(torch.float32)
        deg_clipfeat = torch.load( os.path.splitext(deg_img_name)[0] + ".pt").to(torch.float32)

        anno_path = os.path.join(self.anno_root, self.index_list[index//2])
        # masks = None
        # points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))
        normal_img = size_fix(normal_img, self.img_size)
        deg_img, points, masks = size_fix_all(deg_img, points, masks, self.img_size)

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 3, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) *2

class LIS_dataset(Dataset):
    def __init__(self, img_size, type, data_dir = './anno/LIS/'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')

        self.index_list = os.listdir(self.anno_root)
        self.normal_img_root = os.path.join('./data/LIS',type,'target')
        self.deg_img_root = os.path.join('./data/LIS',type, 'input')
        self.anno_root = os.path.join(data_dir,type)
        self.img_size = img_size
        print('len of index list:',len(self))

    def __getitem__(self, index):
        deg_img_name = os.path.join(self.deg_img_root, os.path.splitext(self.index_list[index//2])[0] +'.png')
        normal_img_name = os.path.join(self.normal_img_root, os.path.splitext(self.index_list[index//2])[0] +'.png')

        deg_img_raw = Image.open(deg_img_name)
        deg_img_raw = np.array(deg_img_raw.convert('RGB'))
        
        normal_img_raw = Image.open(normal_img_name)
        normal_img_raw = np.array(normal_img_raw.convert('RGB'))

        normal_img = transforms.ToTensor()(normal_img_raw)
        deg_img = transforms.ToTensor()(deg_img_raw)

        normal_clipfeat = torch.load( os.path.splitext(normal_img_name)[0] + ".pt").to(torch.float32)
        deg_clipfeat = torch.load( os.path.splitext(deg_img_name)[0] + ".pt").to(torch.float32)
        
        anno_path = os.path.join(self.anno_root, self.index_list[index//2])
        # masks = None
        # points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))
        normal_img = size_fix(normal_img, self.img_size)
        deg_img, points, masks = size_fix_all(deg_img, points, masks, self.img_size)

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 4, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) * 2  

class snow100k_dataset(Dataset):
    def __init__(self, img_size, type, data_dir = './anno/snow100k'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')

        self.index_list = os.listdir(self.anno_root)
        self.img_size = img_size
        self.anno_root = os.path.join(data_dir,type)
        self.normal_img_root = os.path.join('./data/snow100k',type,'target')
        self.deg_img_root = os.path.join('./data/snow100k',type, 'input')

        print('len of index list:',len(self))

    def __getitem__(self, index):
        normal_img_name = os.path.join(self.normal_img_root, os.path.splitext(self.index_list[index//2])[0] +'.png')
        deg_img_name = os.path.join(self.deg_img_root, os.path.splitext(self.index_list[index//2])[0] +'.png')

        deg_img_raw = Image.open(deg_img_name)
        deg_img_raw = np.array(deg_img_raw.convert('RGB'))
        
        normal_img_raw = Image.open(normal_img_name)
        normal_img_raw = np.array(normal_img_raw.convert('RGB'))

        normal_img = transforms.ToTensor()(normal_img_raw)
        deg_img = transforms.ToTensor()(deg_img_raw)

        normal_clipfeat = torch.load( os.path.splitext(normal_img_name)[0] + ".pt").to(torch.float32)
        deg_clipfeat = torch.load( os.path.splitext(deg_img_name)[0] + ".pt").to(torch.float32)
        
        anno_path = os.path.join(self.anno_root, self.index_list[index//2])
        # masks = None
        # points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))
        normal_img = size_fix(normal_img, self.img_size)
        deg_img, points, masks = size_fix_all(deg_img, points, masks, self.img_size)

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 5, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) *2



def get_dataset(args, sam):
    if args.datatype == 'Rain200L':
        train_dataset = Rain200L_dataset(sam, 'train')
        test_dataset = Rain200L_dataset(sam, 'test')
    elif args.datatype == 'Rain200L_cal':
        train_dataset = test_Rain200L_dataset('train')
        test_dataset = test_Rain200L_dataset('test')
    elif args.datatype == 'DDN':
        train_dataset = DDN_dataset(sam, 'train')
        test_dataset = DDN_dataset(sam, 'test')
    elif args.datatype == 'DDN_cal':
        train_dataset = test_DDN_dataset('train')
        test_dataset = test_DDN_dataset('test')
    elif args.datatype == 'GoPro':
        train_dataset = GoPro_dataset(sam, 'train')
        test_dataset = GoPro_dataset(sam, 'test')
    elif args.datatype == 'LIS':
        train_dataset = LIS_dataset(sam, 'train')
        test_dataset = LIS_dataset(sam, 'test')
    elif args.datatype == 'snow100k':
        train_dataset = snow100k_dataset(sam, 'train')
        test_dataset = snow100k_dataset(sam, 'test')

    elif args.datatype == 'APcal':
        Rain200L_train = test_Rain200L_dataset(type='train')
        Rain200L_test = test_Rain200L_dataset(type='test')
        DDN_train = test_DDN_dataset(type='train')
        DDN_test = test_DDN_dataset(type='test')
        GoPro_train = test_GoPro_dataset(type='train')
        GoPro_test = test_GoPro_dataset(type='test')
        LIS_train = test_LIS_dataset(type='train')
        LIS_test = test_LIS_dataset(type='test')
        snow100k_train = test_snow100k_dataset(type='train')
        snow100k_test = test_snow100k_dataset(type='test')

        train_loaders = []
        
        train_loaders.append(torch.utils.data.DataLoader(
            Rain200L_train, 
            batch_size = 1, 
            ))
        train_loaders.append(torch.utils.data.DataLoader(
            DDN_train, 
            batch_size = 1, 
            ))
        train_loaders.append(torch.utils.data.DataLoader(
            GoPro_train, 
            batch_size = 1, 
            ))
        train_loaders.append(torch.utils.data.DataLoader(
            LIS_train, 
            batch_size = 1, 
            ))
        train_loaders.append(torch.utils.data.DataLoader(
            snow100k_train, 
            batch_size = 1, 
            ))
        
        test_loaders = []
        test_loaders.append(torch.utils.data.DataLoader(
            Rain200L_test, 
            batch_size = 1, 
            ))
        test_loaders.append(torch.utils.data.DataLoader(
            DDN_test, 
            batch_size = 1, 
            ))
        test_loaders.append(torch.utils.data.DataLoader(
            GoPro_test, 
            batch_size = 1, 
            ))
        test_loaders.append(torch.utils.data.DataLoader(
            LIS_test, 
            batch_size = 1, 
            ))
        test_loaders.append(torch.utils.data.DataLoader(
            snow100k_test, 
            batch_size = 1, 
            ))

        return train_loaders, test_loaders

    elif args.datatype == 'all':
        Rain200L_train = Rain200L_dataset(img_size = sam.image_encoder.img_size, type='train')
        Rain200L_test = Rain200L_dataset(img_size = sam.image_encoder.img_size, type='test')
        DDN_train = DDN_dataset(img_size = sam.image_encoder.img_size, type='train')
        DDN_test = DDN_dataset(img_size = sam.image_encoder.img_size, type='test')
        GoPro_train = GoPro_dataset(img_size = sam.image_encoder.img_size, type='train')
        GoPro_test = GoPro_dataset(img_size = sam.image_encoder.img_size, type='test')
        LIS_train = LIS_dataset(img_size = sam.image_encoder.img_size, type='train')
        LIS_test = LIS_dataset(img_size = sam.image_encoder.img_size, type='test')
        snow100k_train = snow100k_dataset(img_size = sam.image_encoder.img_size, type='train')
        snow100k_test = snow100k_dataset(img_size = sam.image_encoder.img_size, type='test')
        train_loaders = []
        train_loaders.append(torch.utils.data.DataLoader(
            Rain200L_train, 
            batch_size = args.batch_size, 
            shuffle = True,
            ))
        train_loaders.append(torch.utils.data.DataLoader(
            DDN_train, 
            batch_size = args.batch_size, 
            shuffle = True,
            ))
        train_loaders.append(torch.utils.data.DataLoader(
            GoPro_train, 
            batch_size = args.batch_size, 
            shuffle = True,
            ))
        train_loaders.append(torch.utils.data.DataLoader(
            LIS_train, 
            batch_size = args.batch_size, 
            shuffle = True,
            ))
        train_loaders.append(torch.utils.data.DataLoader(
            snow100k_train, 
            batch_size = args.batch_size, 
            shuffle = True,
            ))

        
        test_loaders = []
        test_loaders.append(torch.utils.data.DataLoader(
            Rain200L_test, 
            batch_size = args.batch_size, 
            ))
        test_loaders.append(torch.utils.data.DataLoader(
            DDN_test, 
            batch_size = args.batch_size, 
            ))
        test_loaders.append(torch.utils.data.DataLoader(
            GoPro_test, 
            batch_size = args.batch_size, 
            ))
        test_loaders.append(torch.utils.data.DataLoader(
            LIS_test, 
            batch_size = args.batch_size, 
            ))
        test_loaders.append(torch.utils.data.DataLoader(
            snow100k_test, 
            batch_size = args.batch_size, 
            ))
        return train_loaders, test_loaders
        
    trainloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        shuffle = True,
        )
    testloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = args.batch_size, 
        )
    return [trainloader], [testloader]