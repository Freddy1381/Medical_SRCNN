import numpy as np
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
import torch.nn as nn
import torch


# PGD attack code
def pgd_attack(model, images, labels, eps=1/255, alpha=0.1, iters=10):
    images = images.to(device)
    labels = labels.to(device)
    loss = SSIMLoss()
        
    ori_images = images.data
        
    for _ in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=-1, max=1).detach_()
    return images, eta


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data_folder = "./"
    test_data_names = ['IXI-T1-test']
    example_folder = "./adversarial_examples/PGD"
    output_folder = "./adversarial_outputs/PGD"
    grad_folder = "./grad/PGD"
    os.makedirs(example_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(grad_folder, exist_ok=True)

    epsilons = [0.01, 0.03, 0.05, 0.07, 0.09]

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
        print(f"Epsilons = {epsilons}")

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


        for epsilon in epsilons:
            # Keep track of the PSNRs and the SSIMs across batches
            PSNRs = AverageMeter()
            SSIMs = AverageMeter()

            # Batches
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                # Move to default device
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 1, w / 4, h / 4), imagenet-normed
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 1, w, h), in [-1, 1]
                hr_imgs_y = convert_image(hr_imgs, 
                                        source='[-1, 1]', 
                                        target='[0, 255]')
                hr_imgs_np = np.squeeze(hr_imgs_y.cpu().numpy())
                
                # Call PGD Attack
                perturbed_data, grad_data = pgd_attack(model, images=lr_imgs, labels=hr_imgs, eps=epsilon)

                # Forward prop.
                adversarial_sr_imgs = model(perturbed_data)  # (1, 1, w, h), in [-1, 1]
                # adversarial_sr_imgs = torch.clamp(adversarial_sr_imgs, min=-1, max=1)

                # Convert tensor to PIL Image for perturbed image
                perturbed_img_pil = convert_image(perturbed_data.cpu().detach().squeeze(0), 
                                                  source='[-1, 1]', 
                                                  target='pil')
                grad_data_pil = convert_image(grad_data.cpu().detach().squeeze(0), 
                                              source='[-1, 1]', 
                                              target='pil')

                # Save perturbed image
                example_epsilon_folder = os.path.join(example_folder, f'{epsilon}')
                os.makedirs(example_epsilon_folder, exist_ok=True)
                perturbed_img_path = os.path.join(example_epsilon_folder, f'perturbed_img_{i}.png')
                perturbed_img_pil.save(perturbed_img_path)

                # Save grad
                grad_epsilon_folder = os.path.join(grad_folder, f'{epsilon}')
                os.makedirs(grad_epsilon_folder, exist_ok=True)
                grad_data_path = os.path.join(grad_epsilon_folder, f'grad_{i}.png')
                grad_data_pil.save(grad_data_path)

                # Convert tensor to PIL Image for adversarial super-resolved image
                adv_sr_img_pil = convert_image(adversarial_sr_imgs.cpu().detach().squeeze(0), source='[-1, 1]', target='pil')

                # Save adversarial super-resolved image
                output_epsilon_folder = os.path.join(output_folder, f'{epsilon}')
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
            print(f'Epsilon: {epsilon}')
            print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
            print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))
            
            update_results_csv('PGD', epsilon, PSNRs.avg, SSIMs.avg)
            
            del lr_imgs, hr_imgs, perturbed_data, grad_data, adversarial_sr_imgs
            print("\n")