//图像超分辨技术，SRGAN 是一种监督算法，你需要高分辨率图像和低分辨率图像成对使用
//使用了python语言和pytorch框架，在pycharm上运行
//使用的数据集是DIV2K，这个数据集包括800张训练图片和100张验证图片，测试集未开放，可自行寻找图片作为测试集

from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
# torchvision.transforms - 图像预处理包
# Compose - 把多个步骤整合一起
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


# 通过后缀检查是否为图片文件
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

# 实际有效的图片区域范围
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),  # 在随机位置裁剪
        ToTensor(),  # convert a PIL image to tensor (H*W*C)
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),  # convert a tensor to PIL image
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),  # 通过双三次插值把图像resize成lr
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),  # 把图像调整到400标准格式
        CenterCrop(400),
        ToTensor()
    ])


# 从文件夹获取训练集
class traindataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super().__init__()
        # 获取图片列表
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        # 定义hr lr转化函数
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        # 获取该index的高清图像，同时转化得到低清图像
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

# 验证集
class valdataset(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])  # 原始图片为高清图
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)  # 裁剪
        lr_image = lr_scale(hr_image)  # 双三次resize成lr
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

# 测试集
class testdataset(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super().__init__()
        # 有hr lr两个文件目录
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        # 获取hr lr 图像
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
