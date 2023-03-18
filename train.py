import torch
from dataset_D2W import MRIDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(disc_DWI, disc_FLAIR, gen_FLAIR, gen_DWI, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    DWI_reals = 0
    DWI_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (dwi, flair) in enumerate(loop):
        dwi = dwi.to(config.DEVICE)
        flair = flair.to(config.DEVICE)

        # Train Discriminators
        with torch.cuda.amp.autocast():  # pytorch 使用amp.autocast半精度加速训练
            fake_dwi = gen_DWI(flair)
            D_DWI_real = disc_DWI(dwi)
            D_DWI_fake = disc_DWI(fake_dwi.detach())
            DWI_reals += D_DWI_real.mean().item()
            DWI_fakes += D_DWI_fake.mean().item()
            D_DWI_real_loss = mse(D_DWI_real, torch.ones_like(D_DWI_real))  # 论文中改进loss，训练更稳定
            D_DWI_fake_loss = mse(D_DWI_fake, torch.zeros_like(D_DWI_fake))
            D_DWI_loss = D_DWI_real_loss + D_DWI_fake_loss

            fake_flair = gen_FLAIR(dwi)
            D_FLAIR_real = disc_FLAIR(flair)
            D_FLAIR_fake = disc_FLAIR(fake_flair.detach())
            D_FLAIR_real_loss = mse(D_FLAIR_real, torch.ones_like(D_FLAIR_real))
            D_FLAIR_fake_loss = mse(D_FLAIR_fake, torch.zeros_like(D_FLAIR_fake))
            D_FLAIR_loss = D_FLAIR_real_loss + D_FLAIR_fake_loss

            # put it togethor
            D_loss = (D_DWI_loss + D_FLAIR_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_DWI_fake = disc_DWI(fake_dwi)
            D_FLAIR_fake = disc_FLAIR(fake_flair)
            loss_G_DWI = mse(D_DWI_fake, torch.ones_like(D_DWI_fake))
            loss_G_FLAIR = mse(D_FLAIR_fake, torch.ones_like(D_FLAIR_fake))

            # cycle loss
            cycle_flair = gen_FLAIR(fake_dwi)
            cycle_dwi = gen_DWI(fake_flair)
            cycle_flair_loss = l1(flair, cycle_flair)
            cycle_dwi_loss = l1(dwi, cycle_dwi)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_flair = gen_FLAIR(flair)
            identity_dwi = gen_DWI(dwi)
            identity_flair_loss = l1(flair, identity_flair)
            identity_dwi_loss = l1(dwi, identity_dwi)

            # add all togethor
            G_loss = (
                    loss_G_FLAIR
                    + loss_G_DWI
                    + cycle_dwi_loss * config.LAMBDA_CYCLE
                    + cycle_flair_loss * config.LAMBDA_CYCLE
                    + identity_dwi_loss * config.LAMBDA_IDENTITY
                    + identity_flair_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(flair * 0.5 + 0.5, f"saved_images/tflair_{idx}.png")
            save_image(dwi * 0.5 + 0.5, f"saved_images/tdwi_{idx}.png")
            save_image(fake_dwi * 0.5 + 0.5, f"saved_images/gdwi_{idx}.png")
            save_image(fake_flair * 0.5 + 0.5, f"saved_images/gflair_{idx}.png")

        loop.set_postfix(DWI_real=DWI_reals / (idx + 1), DWI_fake=DWI_fakes / (idx + 1))


def main():
    disc_DWI = Discriminator(in_channels=3).to(config.DEVICE)
    disc_FLAIR = Discriminator(in_channels=3).to(config.DEVICE)
    gen_FLAIR = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_DWI = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_DWI.parameters()) + list(disc_FLAIR.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_FLAIR.parameters()) + list(gen_DWI.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_DWI, gen_DWI, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_FLAIR, gen_FLAIR, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_DWI, disc_DWI, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_FLAIR, disc_FLAIR, opt_disc, config.LEARNING_RATE,
        )

    dataset = MRIDataset(
        root_dir=config.TRAIN_DIR, transform=config.transforms
    )
    # val_dataset = HorseZebraDataset(
    #    root_horse="cyclegan_test/horse1", root_zebra="cyclegan_test/zebra1", transform=config.transforms
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    # )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_DWI, disc_FLAIR, gen_FLAIR, gen_DWI, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_DWI, opt_gen, filename=config.CHECKPOINT_GEN_DWI)
            save_checkpoint(gen_FLAIR, opt_gen, filename=config.CHECKPOINT_GEN_FLAIR)
            save_checkpoint(disc_DWI, opt_disc, filename=config.CHECKPOINT_CRITIC_DWI)
            save_checkpoint(disc_FLAIR, opt_disc, filename=config.CHECKPOINT_CRITIC_FLAIR)


if __name__ == "__main__":
    main()
