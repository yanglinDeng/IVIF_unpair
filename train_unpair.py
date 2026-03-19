import time
import os
import torch
from torch.optim import Adam
import utils_ablation
from utils_ablation import gradient2,showLossChart
import random
from tqdm import trange
import scipy.io as scio
import torch.nn.functional as F
from torch.autograd import Variable
from CNN_net import CNN_network
from Trans_net import IR_Visible_Fusion_Model
from Generator import Generator_net
from Discriminator import Discriminator_net
from args import args
import kornia.metrics as metrics
import numpy as np
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

re_times = 100
model_target = args.save_model+str(args.trainNumber*(re_times))
save_model_dir = args.root_savemodel+ model_target
save_loss_dir = save_model_dir +"/loss"
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
    os.makedirs(save_loss_dir)

save_loss_txt = save_loss_dir+"/value.txt"

# To generate trainable image pairs
# All unpaired relationships
# difaset = set()
# for i in range(1,args.trainNumber+1):
#     for j in range(1,args.trainNumber+1):
#         sg = (i,j)
#         if i!=j and sg not in difaset:
#             difaset.add(sg)

# All arbitrarily-paired relationships
allaset = set()
for i in range(1,args.trainNumber+1):
    for j in range(1,args.trainNumber+1):
        sg = (i,j)
        if sg not in allaset:
            allaset.add(sg)


def gen_groups(g,rset):
    nowlist = random.sample(list(rset),g)
    irlist = [str(now[0]) for now in nowlist]
    vislist = [str(now[1]) for now in nowlist]
    return irlist,vislist


def train():
    patchPrePath = args.Patch_path;
    patchPrePath2 = args.Patch_path2;
    PatchPaths = utils_ablation.loadPatchesPairPaths()
    batch_size = args.batch_size
    model = CNN_network(2,1)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-4)
    if (args.cuda):
        model.cuda(int(args.device));

    tbar = trange(args.epochs)
    Loss_content = []
    Loss_intensity = []
    Loss_grad = []
    Loss_ssim = []
    all_content_loss = 0.
    all_intensity_loss = 0.
    all_grad_loss = 0.
    all_ssim_loss = 0.


    print('Start training.....')
    for e in tbar:
        print('Epoch %d.....' % e)
        # load training database
        patchesPaths, batches = utils_ablation.load_datasetPair(PatchPaths, batch_size);
        ir_paths = []
        vi_paths = []

        ir_list_a, vi_list_a = gen_groups(args.trainNumber*re_times, allaset)
        ir_paths.extend(ir_list_a)
        vi_paths.extend(vi_list_a)
        print(len(ir_paths), ir_paths[len(ir_paths) - 100])
        print(len(vi_paths), vi_paths[len(vi_paths) - 100])

        model.train()
        count = 0
        for batch in range(batches*(re_times)):
            optimizer.zero_grad()
            image_paths = ir_paths[batch * batch_size:(batch * batch_size + batch_size)]
            image_paths_vis = vi_paths[batch * batch_size:(batch * batch_size + batch_size)]
            # load image patches of this batch.
            image_ir = utils_ablation.get_train_images_auto(patchPrePath + "/ir", image_paths, mode="L");
            image_vi = utils_ablation.get_train_images_auto(patchPrePath2 + "/vis", image_paths_vis, mode="L");

            count += 1

            img_ir = Variable(image_ir, requires_grad=False)
            img_vi = Variable(image_vi, requires_grad=False)
            if args.cuda:
                img_ir = img_ir.cuda(args.device)
                img_vi = img_vi.cuda(args.device)
                model = model.cuda(args.device)

            output = model(torch.cat([img_ir, img_vi], 1));

            illu_vi = utils_ablation.stable_ratio(img_vi,img_ir)
            illu_ir = utils_ablation.stable_ratio(img_ir, img_vi)
            illu_imitated = img_vi * illu_vi + img_ir * illu_ir
            loss_illu = F.l1_loss(output, illu_imitated)


            sobel_vi = utils_ablation.fast_gradient(img_vi, 3)
            sobel_ir = utils_ablation.fast_gradient(img_ir, 3)
            sobel_out = utils_ablation.fast_gradient(output, 3)
            grad_vi = utils_ablation.stable_ratio(sobel_vi, sobel_ir)
            grad_ir = utils_ablation.stable_ratio(sobel_ir, sobel_vi)
            grad_imitated = grad_vi * sobel_vi + grad_ir * sobel_ir
            grad_loss_value = F.l1_loss(sobel_out, grad_imitated)
            #

            ssim_ir = torch.mean(metrics.ssim(img_ir, output, window_size=11, max_val=1.0, eps=1e-12, padding="same"))
            ssim_vi = torch.mean(metrics.ssim(img_vi, output, window_size=11, max_val=1.0, eps=1e-12, padding="same"))
            e_vi = utils_ablation.stable_ratio(ssim_vi,ssim_ir)
            e_ir = utils_ablation.stable_ratio(ssim_ir,ssim_vi)
            loss_ssim = 1 - (ssim_ir * e_ir + ssim_vi * e_vi)
            loss_ssim = loss_ssim / 5


            L_content = loss_illu  + grad_loss_value + loss_ssim;

            total_loss = L_content

            total_loss.backward()
            optimizer.step()


            all_content_loss += L_content.item();
            all_intensity_loss += grad_loss_value.item();
            all_grad_loss += grad_loss_value.item()
            all_ssim_loss += loss_ssim.item()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\t intensity: {:.6f}\t gradient:{:.6f}\tillu:{:.6f}".format(
                    time.ctime(), e + 1, count, batches*(re_times),
                                  all_content_loss / args.log_interval,
                    all_intensity_loss / args.log_interval,
                    all_grad_loss / args.log_interval,all_ssim_loss / args.log_interval

                )
                tbar.set_description(mesg)
                Loss_content.append(all_content_loss / args.log_interval);
                Loss_intensity.append(all_intensity_loss / args.log_interval)
                Loss_grad.append(all_grad_loss/args.log_interval)
                Loss_ssim.append(all_ssim_loss/args.log_interval)
                all_content_loss = 0.
                all_intensity_loss = 0.
                all_grad_loss = 0.
                all_ssim_loss = 0.

            if (batch + 1) % (args.log_interval) == 0:

                model.eval()
                model.cpu()
                save_model_filename = "40" + str(e) + ".model"
                save_model_path = os.path.join(save_model_dir, save_model_filename)
                torch.save(model.state_dict(), save_model_path)
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)


                loss_data_content = torch.tensor(Loss_content).data.cpu().numpy()
                loss_filename_path = "loss_content_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                              '_') + ".mat"
                save_loss_path = os.path.join(save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_content})
                showLossChart(save_loss_path, save_loss_dir + '/content.png','Loss')

                #Lmemory loss
                loss_data_ssim = torch.tensor(Loss_ssim).data.cpu().numpy()
                loss_filename_path = "loss_ssim_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                              '_') + ".mat"
                save_loss_path = os.path.join(save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_ssim})
                showLossChart(save_loss_path, save_loss_dir + '/ssim.png','Loss')

                with open(save_loss_txt, "a", encoding="utf-8") as file:
                   res = "Content Intensity Gradient SSIM"+"\n"+str(loss_data_content)+" "+str(loss_data_ssim)
                   file.write("\n")
                   file.write(res)
                model.train()



    loss_data_content = torch.tensor(Loss_content).data.cpu().numpy()
    loss_filename_path = "Final_loss_content_epoch_" + str(
        args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
    save_loss_path = os.path.join(save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_content})
    showLossChart(save_loss_path, save_loss_dir + "/content.png",'Loss');

    loss_data_memory = torch.tensor(Loss_ssim).data.cpu().numpy()
    loss_filename_path = "loss_ssim_epoch_" + str(
        args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                  '_') + ".mat"
    save_loss_path = os.path.join(save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_memory})
    showLossChart(save_loss_path, save_loss_dir + '/intensity.png','Loss')

    model.eval()
    model.cpu()
    save_model_filename = "40" + str(e) + ".model"
    save_model_path = os.path.join(save_model_dir, save_model_filename)
    torch.save(model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

def main():
    train()


if __name__ == "__main__":
    main()


