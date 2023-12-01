import torch
import numpy as np
from utils import *
from PIL import Image
from datasets import SRDataset
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data_folder = "./"
    test_data_names = ['IXI-T1-test']
    example_folder = "./adversarial_examples/FGSM"
    output_folder = "./adversarial_outputs/FGSM"
    os.makedirs(example_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    epsilon = 1/255

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
        print(f"Epsilon = {epsilon}")

        # Custom dataloader
        test_dataset = SRDataset(data_folder,
                                split='test',
                                crop_size=0,
                                scaling_factor=4,
                                lr_img_type='imagenet-norm',
                                hr_img_type='[-1, 1]',
                                test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                                pin_memory=True)

        # Keep track of the PSNRs and the SSIMs across batches
        PSNRs = AverageMeter()
        SSIMs = AverageMeter()

        # Batches
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            if i >= 100:
                break
            # Move to default device
            lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            # lr_imgs_var = Variable(lr_imgs, requires_grad=True)
            hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]
            hr_imgs_y = convert_image(hr_imgs, 
                                      source='[-1, 1]', 
                                      target='[0, 255]')
            hr_imgs_np = np.squeeze(hr_imgs_y.cpu().numpy())

            # Forward prop.
            sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

            ori_images = lr_imgs.clone().detach().requires_grad_(True)
            loss = F.mse_loss(model(ori_images), hr_imgs)
            model.zero_grad()
            loss.backward(retain_graph=True)

            # Create perturbed image using FGSM
            perturbed_imgs = ori_images.data + epsilon * torch.sign(ori_images.grad.data)
            perturbed_imgs = torch.clamp(perturbed_imgs, min=-1, max=1)

            # Forward prop. with the perturbed image
            adversarial_sr_imgs = model(perturbed_imgs)
            adversarial_sr_imgs = torch.clamp(adversarial_sr_imgs, min=-1, max=1)

            # Convert tensor to PIL Image for perturbed image
            perturbed_img_pil = convert_image(perturbed_imgs.cpu().detach().squeeze(0), 
                                              source='[-1, 1]', 
                                              target='pil')

            # Save perturbed image
            perturbed_img_path = os.path.join(example_folder, f'perturbed_img_{i}.png')
            perturbed_img_pil.save(perturbed_img_path)

            # Convert tensor to PIL Image for adversarial super-resolved image
            adv_sr_img_pil = convert_image(adversarial_sr_imgs.cpu().detach().squeeze(0), source='[-1, 1]', target='pil')

            # Save adversarial super-resolved image
            adv_sr_img_path = os.path.join(output_folder, f'adversarial_sr_img_{i}.png')
            adv_sr_img_pil.save(adv_sr_img_path)

            # Calculate PSNR and SSIM for the adversarial example
            adv_sr_imgs_y = convert_image(adversarial_sr_imgs, 
                                          source='[-1, 1]', 
                                          target='[0, 255]')
            adv_sr_imgs_np = np.squeeze(adv_sr_imgs_y.cpu().detach().numpy())
            psnr_adv = peak_signal_noise_ratio(hr_imgs_np, 
                                               adv_sr_imgs_np, 
                                               data_range=255.)
            ssim_adv = structural_similarity(hr_imgs_np, 
                                             adv_sr_imgs_np, 
                                             data_range=255.)

            PSNRs.update(psnr_adv, lr_imgs.size(0))
            SSIMs.update(ssim_adv, lr_imgs.size(0))

            if i % 100 == 0:
                print(f"{round((i / len(test_loader)) * 100, 2)}% done.....")

        # Print average PSNR and SSIM
        print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

    print("\n")