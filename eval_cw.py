import torch
import numpy as np
from utils import *
from PIL import Image
from datasets import SRDataset
import torch.nn as nn
import torch.optim as optim

def tanh_space(x):
    return 1 / 2 * (torch.tanh(x) + 1)

def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

def inverse_tanh_space(x):
    return atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

def carlini_wagner_attack(model, input_image, target_image, confidence, max_iterations=100, learning_rate=0.01):
    """
    Carlini-Wagner Attack Implementation for SRCNN in PyTorch. Took inspiration from torchattack's cw.py

    Parameters:
        - model: The PyTorch SRCNN model to be attacked.
        - input_image: Low-resolution input image (PyTorch tensor).
        - target_image: Target high-resolution image (PyTorch tensor).
        - max_iterations: Maximum number of optimization iterations.
        - confidence: Confidence parameter for the attack.
        - learning_rate: Optimization learning rate.

    Returns:
        - perturbed_image: Adversarial example (perturbed input_image).
    """
    input_image = input_image.to(device)
    target_image = target_image.to(device)

    w = inverse_tanh_space(input_image).detach()
    w.requires_grad = True

    best_adv_images = input_image.clone().detach()
    best_L2 = 1e10 * torch.ones(len(input_image)).to(device)
    max_loss_index = 0
    prev_cost = 1e10
    dim = len(input_image.shape)

    criterion = SSIMLoss()
    optimizer = optim.Adam([w], lr=learning_rate)

    for step in range(max_iterations):
        adv_images = tanh_space(w)

        current_L2 = criterion(adv_images, input_image)
        L2_loss = current_L2.sum()
        output_image = model(input_image)
        loss = criterion(output_image, target_image).to(device)
        cost = L2_loss + confidence * loss

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        pre = torch.argmax(output_image.detach(), 1)
        condition = (pre != target_image).float()

        mask = condition * (best_L2 > current_L2.detach())
        best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

        mask = mask.view([-1] + [1] * (dim - 1))
        max_loss_index = torch.argmax(mask).item()
        best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

        # Early stop when loss does not converge.
        # max(.,1) To prevent MODULO BY ZERO error in the next step.
        if step % max(max_iterations // 10, 1) == 0:
            if loss.item() > prev_cost:
                return torch.clamp(best_adv_images[max_loss_index].unsqueeze(0), min=-1, max=1)
            prev_cost = loss.item()
    
    return torch.clamp(best_adv_images[max_loss_index].unsqueeze(0), min=-1, max=1)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data_folder = "./"
    test_data_names = ['IXI-T1-test']
    example_folder = "./adversarial_examples/CW"
    output_folder = "./adversarial_outputs/CW"
    os.makedirs(example_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    confidences = [0.1, 0.2, 0.25, 0.5, 0.8]

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
        print(f"Confidence = {confidences}")

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

        for confidence in confidences:
            # Keep track of the PSNRs and the SSIMs across batches
            PSNRs = AverageMeter()
            SSIMs = AverageMeter()

            # Batches
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                if i >= 1:
                    break
                # Move to default device
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 1, w / 4, h / 4), imagenet-normed
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 1, w, h), in [-1, 1]

                perturbed_data = carlini_wagner_attack(model, lr_imgs, hr_imgs, confidence=confidence)
                # Convert tensor to PIL Image for perturbed image
                perturbed_img_pil = convert_image(perturbed_data.cpu().detach().squeeze(0), 
                                                  source='[-1, 1]', 
                                                  target='pil')

                # Save perturbed image
                example_epsilon_folder = os.path.join(example_folder, f'{confidence}')
                os.makedirs(example_epsilon_folder, exist_ok=True)
                perturbed_img_path = os.path.join(example_epsilon_folder, f'perturbed_img_{i}.png')
                perturbed_img_pil.save(perturbed_img_path)

                # Forward prop.
                adversarial_sr_imgs = model(perturbed_data)  # (1, 1, w, h), in [-1, 1]
                # adversarial_sr_imgs = torch.clamp(adversarial_sr_imgs, min=-1, max=1)

                # Convert tensor to PIL Image for adversarial super-resolved image
                adv_sr_img_pil = convert_image(adversarial_sr_imgs.cpu().detach().squeeze(0), source='[-1, 1]', target='pil')

                # Save adversarial super-resolved image
                output_epsilon_folder = os.path.join(output_folder, f'{confidence}')
                os.makedirs(output_epsilon_folder, exist_ok=True)
                adv_sr_img_path = os.path.join(output_epsilon_folder, f'adversarial_sr_img_{i}.png')
                adv_sr_img_pil.save(adv_sr_img_path)

                # Calculate PSNR and SSIM
                psnr = get_psnr(hr_imgs, adversarial_sr_imgs)
                ssim = get_ssim(hr_imgs, adversarial_sr_imgs)
                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))

                if i % 100 == 0:
                    print(f"{round(i / len(test_loader) * 100)}% done......")

            # Print average PSNR and SSIM
            print(f'Confidence: {confidence}')
            print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
            print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

            update_results_csv('CW', confidence, PSNRs.avg, SSIMs.avg)

            del lr_imgs, hr_imgs, perturbed_data, adversarial_sr_imgs
        print("\n")