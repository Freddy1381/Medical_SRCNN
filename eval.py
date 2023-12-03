import numpy as np
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data_folder = "./"
    test_data_names = ['IXI-T1-test']
    example_folder = "./adversarial_examples/Clean"
    output_folder = "./adversarial_outputs/Clean"
    os.makedirs(example_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

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

                # Convert tensor to PIL Image for perturbed image
                lr_img_pil = convert_image(lr_imgs.cpu().detach().squeeze(0), 
                                                  source='[-1, 1]', 
                                                  target='pil')
                
                # Save clean low-res image
                clean_lr_img_path = os.path.join(example_folder, f'example_img_{i}.png')
                lr_img_pil.save(clean_lr_img_path)

                # Forward prop.
                sr_imgs = model(lr_imgs)  # (1, 1, w, h), in [-1, 1]

                # Convert tensor to PIL Image for clean super-resolved image
                clean_sr_img_pil = convert_image(sr_imgs.cpu().detach().squeeze(0), source='[-1, 1]', target='pil')

                # Save clean super-resolved image
                clean_sr_img_path = os.path.join(output_folder, f'output_img_{i}.png')
                clean_sr_img_pil.save(clean_sr_img_path)

                # Calculate PSNR and SSIM
                psnr = get_psnr(hr_imgs, sr_imgs)
                ssim = get_ssim(hr_imgs, sr_imgs)
                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))

                if i % 100 == 0:
                    print(f"{round(i / len(test_loader) * 100)}% done......")

        # Print average PSNR and SSIM
        print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))
        
        update_results_csv('Clean', 0, PSNRs.avg, SSIMs.avg)

    del lr_imgs, hr_imgs, sr_imgs
    print("\n")
