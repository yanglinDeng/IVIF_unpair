import os
from scipy.misc import imread, imsave
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
from args import args
from CNN_net import CNN_network
from Trans_net import IR_Visible_Fusion_Model
from Generator import Generator_net
from Discriminator import Discriminator_net


def gaussian_weight(ps):
    """生成高斯权重矩阵，用于加权平均"""
    x = np.linspace(-3, 3, ps)
    y = np.linspace(-3, 3, ps)
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g


def load_model(path, input_nc, output_nc):
    """加载预训练模型"""
    nest_model = CNN_network(2,1)
    nest_model.load_state_dict(torch.load(path))
    total_param = 0
    print("MODEL DETAILS:\n")
    print(nest_model)
    for param in nest_model.parameters():
        print(param.dtype)
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', nest_model._get_name(), total_param)

    bytes_per_param = 4
    total_bytes = total_param * bytes_per_param
    total_megabytes = total_bytes / (1024 * 1024)
    total_kilobytes = total_bytes / 1024

    print("Total parameters in MB:", total_megabytes)
    print("Total parameters in KB:", total_kilobytes)

    nest_model.eval()
    return nest_model


def run_demo(model, infrared_path, visible_path, output_path_root, index, mode):

    ir_img = imread(infrared_path, mode='L')
    vi_img = imread(visible_path, mode='L')
    ir_img = ir_img / 255.0
    vi_img = vi_img / 255.0
    h, w = vi_img.shape
    ps = 128
    stride = 64
    weight = gaussian_weight(ps)
    batch_size = 4
    patches_coords = []
    for i in range(0, h - ps + 1, stride):
        for j in range(0, w - ps + 1, stride):
            patches_coords.append((i, j))
    for i in range(0, h - ps + 1, stride):
        patches_coords.append((i, w - ps))
    for j in range(0, w - ps + 1, stride):
        patches_coords.append((h - ps, j))
    patches_coords.append((h - ps, w - ps))

    fuseImage = np.zeros((h, w))
    fuseCnt = np.zeros((h, w))
    model.eval()
    if args.cuda:
        model = model.cuda(args.device)

    for batch_start in range(0, len(patches_coords), batch_size):
        batch_coords = patches_coords[batch_start:batch_start + batch_size]

        batch_ir = []
        batch_vi = []
        for (i, j) in batch_coords:
            ir_patch = ir_img[i:i + ps, j:j + ps]
            vi_patch = vi_img[i:i + ps, j:j + ps]
            batch_ir.append(ir_patch.reshape(1, ps, ps))
            batch_vi.append(vi_patch.reshape(1, ps, ps))


        batch_ir = np.stack(batch_ir, axis=0)
        batch_vi = np.stack(batch_vi, axis=0)
        batch_ir = torch.from_numpy(batch_ir).float()
        batch_vi = torch.from_numpy(batch_vi).float()

        if args.cuda:
            batch_ir = batch_ir.to(args.device)
            batch_vi = batch_vi.to(args.device)

        with torch.no_grad():
            inputs = torch.cat([batch_ir, batch_vi], dim=1)
            outputs = model(inputs).cpu().numpy()


        for idx, (i, j) in enumerate(batch_coords):
            fuseImage[i:i + ps, j:j + ps] += outputs[idx][0] * weight
            fuseCnt[i:i + ps, j:j + ps] += weight


    fuseImage = fuseImage / fuseCnt

    fuseImage = gaussian_filter(fuseImage, sigma=0.3)

    file_name = str(index) + '.png'
    output_path = output_path_root+"/" + file_name
    imsave(output_path, fuseImage)
    return


def main():
    """主函数"""
    output_path = args.root_testoutput
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    in_c = 2
    out_c = 1
    mode = 'L'
    model_path = args.pretrained_cnn_path

    num_imgs = 361
    with torch.no_grad():
        model = load_model(model_path, in_c, out_c)
        for i in range(num_imgs):
            index = i + 1
            infrared_path = args.test_ir_path + str(index) + '.png'
            visible_path = args.test_vi_path + str(index) + '.png'
            run_demo(model, infrared_path, visible_path, output_path, index, mode)

    print('Done......')




if __name__ == '__main__':
    main()