import random, torch, os, numpy as np
import torch.nn as nn
import config
import copy
    
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pandas as pd


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint" + ' => ' + str(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    # 不写下面这行，优化器会重用old checkpoint的学习率。
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# img1为真实图像y，img2为合成图像G(x)
def calculate_mae(img1, img2):
    m = np.mean(abs(img1 - img2))
    return m


# img1为真实图像y，img2为合成图像G(x)
def calculate_nmse(img1, img2):
    k1 = np.sum((img1 - img2) ** 2)
    k2 = np.sum(img1 ** 2)
    return k1 / k2


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization #        [-1,1]->[0,1]
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")  # save_image函数接受的就是归一化为[0,1]的tensor，
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_some_validation_examples(gen, val_loader, if_save_image=False,
                                  folder_img1=None, folder_img2=None,
                                  if_save_eva_index=False,
                                  csv_path=None,DWI2FLAIR=True):
    # x, y = next(iter(val_loader))
    psnr_list = []
    ssmi_list = []
    nmse_list = []
    
    for i, (x, y) in enumerate(val_loader): # x-->DWI_image, y -->FLAIR_image
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        if DWI2FLAIR:
            pass
        else:
            y,x = x,y # FLAIR2DWI
        gen.eval()
        with torch.no_grad():
            y_fake = gen(x)
            psnr = compare_psnr(y.squeeze(0).cpu().numpy(), y_fake.squeeze(0).cpu().numpy())
            ssmi = compare_ssim(y.squeeze(0).cpu().numpy(), y_fake.squeeze(0).cpu().numpy(), channel_axis=0)
            # mae = calculate_mae(y.squeeze(0).cpu().numpy(), y_fake.squeeze(0).cpu().numpy())
            nmse = calculate_nmse(y.squeeze(0).cpu().numpy(), y_fake.squeeze(0).cpu().numpy())
            psnr_list.append(psnr)
            ssmi_list.append(ssmi)
            nmse_list.append(nmse)
            if if_save_image:
                y_fake = y_fake * 0.5 + 0.5  # remove normalization = unnormalization
                save_image(y_fake, folder_img1 + f"/y_gen_{i}.png")
                save_image(x * 0.5 + 0.5, folder_img1 + f"/input_{i}.png")
                save_image(y * 0.5 + 0.5, folder_img1 + f"/label_{i}.png")
                image = torch.cat([x * 0.5 + 0.5, y * 0.5 + 0.5, y_fake], dim=3)
                save_image(image, folder_img2 + f"/input_label_gen{i}.png")
    # ddof=1 -> n 总体  ddof=0  ->(n-1) 样本：无偏估计
    print(f"psnr的均值为{np.mean(psnr_list)},psnr的标准差为{np.std(psnr_list, ddof=1)}")
    print(f"ssmi的均值为{np.mean(ssmi_list)},ssmi的标准差为{np.std(ssmi_list, ddof=1)}")
    print(f"nmse的均值为{np.mean(nmse_list)},nmse的标准差为{np.std(nmse_list, ddof=1)}")
    if if_save_eva_index:
        data = pd.DataFrame({'psnr': psnr_list, 'ssmi': ssmi_list, 'nmse': nmse_list})
        data.to_csv(csv_path, header=True)

    # print(psnr_list, ssmi_list, nmse_list)
    return psnr_list, ssmi_list, nmse_list





