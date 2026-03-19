
from os import listdir
from os.path import join
import random
from torch import nn
from PIL import Image
from args_CNN import args
from scipy.misc import imread
from torchvision import transforms
from typing import Type, Union
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scipy.io as scio
import torch
import torch.nn.functional as F
import numpy as np
from math import exp
import cv2

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    )
    if args.cuda:
        mat = mat.to(args.device)

    bias = torch.tensor([0.0 / 255, -0.5, -0.5])
    if args.cuda:
        bias = bias.to(args.device)
    temp = (im_flat + bias).mm(mat)
    if args.cuda:
        temp = temp.to(args.device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
            .transpose(1, 3)
            .transpose(2, 3)
    )
    return out


def expenlarge_stable_ratio(a, b, temperature=0.1):
    """
    使用指数函数拉大a/(a+b)和b/(a+b)之间的差距

    Args:
        a, b: 原始数值
        temperature: 温度参数，越小差距越大
    """
    # 原始比例
    p_a = stable_ratio(a,b)
    p_b = stable_ratio(b,a)

    # 应用指数函数
    exp_a = torch.exp(p_a / temperature)
    exp_b = torch.exp(p_b / temperature)

    # 重新归一化
    new_a = stable_ratio(exp_a,exp_b)
    new_b = stable_ratio(exp_b,exp_a)

    return new_a, new_b

def suitable_ratio(a,b):
    x1 = stable_softmax_ratio(a,b)
    x2 = stable_ratio(a,b)
    return (x1+x2)/2

def absenlarge_stable_ratio(a, b,k=0.3):
    m = a>b
    m= m.int()
    na = a+k*m*b-(k*(1-m)*a)
    nb = b+k*(1-m)*a-(k*m*b)
    newa = stable_ratio(na, nb)
    newb = stable_ratio(nb, na)
    return newa,newb
def stable_ratio(a, b):
    """数值稳定的比值计算，避免指数溢出"""
    return a / (a + b + 1e-8)
def stable_softmax_ratio(a, b):
    """数值稳定的比值计算，避免指数溢出"""
    max_val = torch.max(a, b)
    exp_a = torch.exp(a)
    exp_b = torch.exp(b)
    return exp_a / (exp_a + exp_b + 1e-8)
def create_sobel_kernel(size=5):
    """创建指定大小的Sobel核"""
    if size == 3:
        # 3x3 Sobel核
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)
    elif size == 5:
        # 5x5 Sobel核
        sobel_x = torch.tensor([[-1, -2, 0, 2, 1],
                                [-4, -8, 0, 8, 4],
                                [-6, -12, 0, 12, 6],
                                [-4, -8, 0, 8, 4],
                                [-1, -2, 0, 2, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -4, -6, -4, -1],
                                [-2, -8, -12, -8, -2],
                                [0, 0, 0, 0, 0],
                                [2, 8, 12, 8, 2],
                                [1, 4, 6, 4, 1]], dtype=torch.float32)
    elif size == 7:
        # 7x7 Sobel核
        sobel_x = torch.tensor([
            [-1, -4, -5, 0, 5, 4, 1],
            [-6, -24, -30, 0, 30, 24, 6],
            [-15, -60, -75, 0, 75, 60, 15],
            [-20, -80, -100, 0, 100, 80, 20],
            [-15, -60, -75, 0, 75, 60, 15],
            [-6, -24, -30, 0, 30, 24, 6],
            [-1, -4, -5, 0, 5, 4, 1]
        ], dtype=torch.float32)
        sobel_y = torch.tensor([
            [-1, -6, -15, -20, -15, -6, -1],
            [-4, -24, -60, -80, -60, -24, -4],
            [-5, -30, -75, -100, -75, -30, -5],
            [0, 0, 0, 0, 0, 0, 0],
            [5, 30, 75, 100, 75, 30, 5],
            [4, 24, 60, 80, 60, 24, 4],
            [1, 6, 15, 20, 15, 6, 1]
        ], dtype=torch.float32)
    else:
        raise ValueError("支持的Sobel核大小: 3, 5, 7")

    return sobel_x, sobel_y


def fast_gradient(image_tensor, kernel_size=5):
    """基于梯度幅值的快速纹理提取 - 支持大Sobel核"""
    # 创建指定大小的Sobel滤波器核
    sobel_x, sobel_y = create_sobel_kernel(kernel_size)
    if (args.cuda):
        sobel_x = sobel_x.cuda(args.device)
        sobel_y = sobel_y.cuda(args.device)
    sobel_x = sobel_x.view(1, 1, kernel_size, kernel_size)
    sobel_y = sobel_y.view(1, 1, kernel_size, kernel_size)

    # 计算合适的填充大小以保持输出尺寸
    padding = kernel_size // 2

    # 应用Sobel滤波器
    grad_x = F.conv2d(image_tensor, sobel_x, padding=padding)
    grad_y = F.conv2d(image_tensor, sobel_y, padding=padding)

    # 数值稳定的梯度幅值计算
    grad_x = torch.clamp(grad_x, -1e6, 1e6)  # 防止溢出
    grad_y = torch.clamp(grad_y, -1e6, 1e6)

    # 计算梯度幅值
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)


    # 归一化到0-255
    eps = 1e-8
    texture_map = (gradient_magnitude - gradient_magnitude.min()) / (
            gradient_magnitude.max() - gradient_magnitude.min() + eps)
    # print(texture_map.shape)
    # texture_map = texture_map.squeeze().cpu().byte()  # [H, W], uint8
    return texture_map




def showLossChart(path, savedName,name):
    plt.cla();
    plt.clf();
    if (path == ""):
        return;
    data = scio.loadmat(path)
    loss = data[name][0];

    x_data = range(0, len(loss));
    y_data = loss;

    plt.plot(x_data, y_data);
    plt.xlabel("Step");
    plt.ylabel(name);
    plt.savefig(savedName);

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window




def gmsd_value(dis_img: Type[Union[torch.Tensor, np.ndarray]], ref_img: Type[Union[torch.Tensor, np.ndarray]], c=170,
               device='cuda'):
    dis_img = dis_img.unsqueeze(0).unsqueeze(0)
    ref_img = ref_img.unsqueeze(0).unsqueeze(0)

    if torch.max(dis_img) <= 1:
        dis_img = dis_img * 255
    if torch.max(ref_img) <= 1:
        ref_img = ref_img * 255

    hx = torch.tensor([[1 / 3, 0, -1 / 3]] * 3, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # Prewitt算子
    if (args.cuda):
        hx = hx.cuda(args.device);
    ave_filter = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float).unsqueeze(0).unsqueeze(0)  # 均值滤波核
    if (args.cuda):
        ave_filter = ave_filter.cuda(int(args.device));
    down_step = 2  # 下采样间隔
    hy = hx.transpose(2, 3)

    dis_img = dis_img.float()
    if (args.cuda):
        dis_img = dis_img.cuda(args.device);
    ref_img = ref_img.float()
    if (args.cuda):
        ref_img = ref_img.cuda(args.device);

    ave_dis = F.conv2d(dis_img, ave_filter, stride=1)
    ave_ref = F.conv2d(ref_img, ave_filter, stride=1)

    ave_dis_down = ave_dis[:, :, 0::down_step, 0::down_step]
    ave_ref_down = ave_ref[:, :, 0::down_step, 0::down_step]

    mr_sq = F.conv2d(ave_ref_down, hx) ** 2 + F.conv2d(ave_ref_down, hy) ** 2
    md_sq = F.conv2d(ave_dis_down, hx) ** 2 + F.conv2d(ave_dis_down, hy) ** 2
    mr = torch.sqrt(mr_sq)
    md = torch.sqrt(md_sq)
    GMS = (2 * mr * md + c) / (mr_sq + md_sq + c)
    GMSD = torch.std(GMS.view(-1))
    return GMSD.item()


#
def gmsd(img1, img2):
    num = img1.shape[0]
    value = 0.
    for i in range(num):
        img_ref = img1[i, 0, :, :]
        img_dis = img2[i, 0, :, :]
        value += gmsd_value(img_dis, img_ref)
    return value / num


def gradient(x):
    dim = x.shape;
    if (args.cuda):
        x = x.cuda(args.device);
    kernel = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]];
    # kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]];
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1], dim[1], 1, 1);
    weight = nn.Parameter(data=kernel, requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(args.device);
    gradMap = F.conv2d(x, weight=weight, stride=1, padding=1);
    # showTensor(gradMap);
    return gradMap;


def gradient2(x):
    dim = x.shape;
    if (args.cuda):
        x = x.cuda(args.device);
    # kernel = [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]];
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]];
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1], dim[1], 1, 1);
    weight = nn.Parameter(data=kernel, requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(args.device);
    gradMap = F.conv2d(x, weight=weight, stride=1, padding=1);
    # showTensor(gradMap);
    return gradMap;


def loadPatchesPairPaths2(directory):
    imagePatchesIR = [];
    imagePatchesVIS = [];
    for i in range(0 + 1, args.trainNumber + 1):
        irPatchPath = directory + "/IR/" + str(i) + ".png";
        visPatchPath = directory + "/VIS_gray/" + str(i) + ".png";
        imagePatchesIR.append(irPatchPath);
        imagePatchesVIS.append(visPatchPath);
    return imagePatchesIR, imagePatchesVIS;


def loadPatchesPairPaths():
    imagePatches = [];
    for i in range(0 + 1, args.trainNumber + 1):
        imagePatches.append(str(i));
    return imagePatches;


def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])

    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U, D, V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


def load_datasetPair(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        image_path = image_path[:-mod];
    num_imgs -= mod
    original_img_path = image_path[:num_imgs]

    # random
    random.shuffle(original_img_path)
    batches = int(len(original_img_path) // BATCH_SIZE)
    return original_img_path, batches


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = imread(path, mode=mode)
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')

    image = image / 255;
    return image

def get_train_images_auto(pre, paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(pre + "/" + path + ".png", height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def get_vi_y(pre, paths, height=256, width=256, mode='RGB'):
    totensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        img_path = pre + "/" + path + ".png"
        # image = get_image_vi(pre + "/" + path + ".png", height, width, mode=mode)
        vi_0 = cv2.imread(img_path)
        vi_0 = totensor(cv2.cvtColor(vi_0, cv2.COLOR_BGR2YCrCb))  # CHW
        y_0 = vi_0[0, :, :].unsqueeze(dim=0).clone()
        cb = vi_0[1, :, :].unsqueeze(dim=0)
        cr = vi_0[2, :, :].unsqueeze(dim=0)
        images.append(y_0)
    images = torch.stack(images, dim=0)
    return images

def get_ir(pre, paths, height=256, width=256, mode='RGB'):
    totensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        img_path = pre + "/" + path + ".png"
        # image = get_image_vi(pre + "/" + path + ".png", height, width, mode=mode)
        ir_0 = cv2.imread(img_path)
        ir_0 = totensor(cv2.cvtColor(ir_0, cv2.COLOR_BGR2GRAY))
        images.append(ir_0)
    images = torch.stack(images, dim=0)
    return images

def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            # test_rgb = ImageToTensor(image).numpy()
            # shape = ImageToTensor(image).size()
            image = ImageToTensor(image).float().numpy() * 255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images