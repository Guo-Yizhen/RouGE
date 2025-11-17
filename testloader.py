import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from util.utils import *

class test_Rain200L_dataset(Dataset):
    def __init__(self, type, data_dir = './anno/Rain200L'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')

        self.index_list = os.listdir(self.anno_root)
        self.normal_img_root = os.path.join('./data/Rain200L',type,'target')
        self.deg_img_root = os.path.join('./data/Rain200L',type, 'input')

        print('len of Rain200L_dataset index list:',len(self))

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
        masks = None
        points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 1, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) *2
    
class test_DDN_dataset(Dataset):
    def __init__(self, type, data_dir = './anno/DDN'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')

        self.index_list = os.listdir(self.anno_root)
        self.normal_img_root = os.path.join('./data/DDN',type,'target')
        self.deg_img_root = os.path.join('./data/DDN', type, 'input')

        print('len of DDN_dataset index list:',len(self))

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
        masks = None
        points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 2, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) *2
class test_GoPro_dataset(Dataset):
    def __init__(self, type, data_dir = './anno/GoPro'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')

        self.index_list = os.listdir(self.anno_root)
        self.normal_img_root = os.path.join('./data/GoPro',type,'target')
        self.deg_img_root = os.path.join('./data/GoPro', type, 'input')

        print('len of GoPro_dataset index list:',len(self))

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
        masks = None
        points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 3, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) *2
    
class test_LIS_dataset(Dataset):
    def __init__(self, type, data_dir = './anno/LIS_/'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')

        self.index_list = os.listdir(self.anno_root)
        self.normal_img_root = os.path.join('./data/LIS',type,'target')
        self.deg_img_root = os.path.join('./data/LIS',type, 'input')
        print('len of LIS_dataset index list:',len(self))
        

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
        masks = None
        points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 4, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) * 2
    
class test_snow100k_dataset(Dataset):
    def __init__(self, type, data_dir = './anno/snow100k_'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')
        self.index_list = os.listdir(self.anno_root)
        self.normal_img_root = os.path.join('./data/snow100k',type,'target')
        self.deg_img_root = os.path.join('./data/snow100k',type, 'input')

        print('len of snow100k_dataset index list:',len(self))

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
        masks = None
        points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))
        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 5, points, masks  
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) *2
    
class test_Cityrain_dataset(Dataset):
    def __init__(self, type, data_dir = './anno/Cityrain/'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')
        self.index_list = os.listdir(self.anno_root)
        self.normal_img_root = os.path.join('./data/Cityrain',type,'target')
        self.deg_img_root = os.path.join('./data/Cityrain',type, 'input')
        print('len of LIS_dataset index list:',len(self))
        

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
        masks = None
        points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 1, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) * 2

class test_Cityfoggy_dataset(Dataset):
    def __init__(self, type, data_dir = './anno/Cityfoggy/'):
        if type != 'train':
            self.anno_root = os.path.join(data_dir,'test_clean')
        else:
            self.anno_root = os.path.join(data_dir,'train_clean')
        self.index_list = os.listdir(self.anno_root)
        self.normal_img_root = os.path.join('./data/Cityfoggy',type,'target')
        self.deg_img_root = os.path.join('./data/Cityfoggy',type, 'input')
        print('len of LIS_dataset index list:',len(self))
        
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
        masks = None
        points = None
        with open(anno_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
            masks = torch.tensor(js['masks'][index%2]).unsqueeze(0)
            points = torch.tensor(js['points'][index%2]).unsqueeze(0)
        points = torch.flip(points, dims=(2,))

        return deg_img, normal_img, deg_clipfeat, normal_clipfeat, 1, points, masks
 
    #获取数据集的大小
    def __len__(self):
        return len(self.index_list) * 2
