import numpy as np
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data_folder = "./"
    test_data_names = ['IXI-T1-test']

    # Model checkpoints
    # srgan_checkpoint = "./checkpoint_srgan.pth.tar"
    srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

    # Load model, either the SRResNet or the SRGAN
    srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
    srresnet.eval()
    model = srresnet
    # srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
    # srgan_generator.eval()
    # model = srgan_generator

    # Evaluate
    for test_data_name in test_data_names:
        print("\nFor %s:\n" % test_data_name)

        # Custom dataloader
        test_dataset = SRDataset(data_folder,
                                split='test',
                                crop_size=0,
                                scaling_factor=4,
                                lr_img_type='imagenet-norm',
                                hr_img_type='[-1, 1]',
                                test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=1, 
                                                  shuffle=False, 
                                                  num_workers=4, 
                                                  pin_memory=True)

        # Keep track of the PSNRs and the SSIMs across batches
        PSNRs = AverageMeter()
        SSIMs = AverageMeter()

        # Prohibit gradient computation explicitly because I had some problems with memory
        with torch.no_grad():
            # Batches
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                # Move to default device
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 1, w / 4, h / 4), imagenet-normed
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 1, w, h), in [-1, 1]

                # Forward prop.
                sr_imgs = model(lr_imgs)  # (1, 1, w, h), in [-1, 1]

                # Calculate PSNR and SSIM
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='[0, 255]')
                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='[0, 255]')
                hr_imgs_np = hr_imgs_y.cpu().numpy()
                sr_imgs_np = sr_imgs_y.cpu().numpy()
                psnr = peak_signal_noise_ratio(np.squeeze(hr_imgs_np), 
                                               np.squeeze(sr_imgs_np), 
                                               data_range=255)
                ssim = structural_similarity(np.squeeze(hr_imgs_np), 
                                             np.squeeze(sr_imgs_np),
                                             data_range=255, 
                                             channel_axis=None)
                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))

                if i % 100 == 0:
                    print(f"{round((i / len(test_loader)) * 100, 2)}% done.....")

        # Print average PSNR and SSIM
        print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

    print("\n")
