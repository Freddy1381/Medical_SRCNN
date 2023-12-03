import numpy as np
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
import torch.nn as nn
import torch


def zoo_attack(model, images, labels, alpha=0.01, beta=0.001, iterations=10):
    # images: the original image
    # labels: the target class for the adversarial example
    # alpha: learning rate for the attack
    # beta: finite difference parameter
    # iterations: number of iterations for the attack

    adversarial_image = np.copy(images)  # batch_size x 1 x w x h
    adversarial_image.to(device)

    for _ in range(iterations):
        gradient = torch.zeros_like(adversarial_image)

        # # Approximate the gradient
        # for x in range(adversarial_image.shape[2]):
        #     for y in range(adversarial_image.shape[3]):
        #         original_value = adversarial_image[0, 0, x, y]
        #
        #         # Perturb the pixel value positively
        #         adversarial_image[0, 0, x, y] = original_value + beta
        #         output_plus = model(adversarial_image)
        #
        #         # Perturb the pixel value negatively
        #         adversarial_image[0, 0, x, y] = original_value - beta
        #         output_minus = model(adversarial_image)
        #
        #         # Approximate gradient (central difference)
        #         gradient[0, 0, x, y] = (output_plus - output_minus) / (2 * beta)
        #
        #         # Reset pixel value
        #         adversarial_image[0, 0, x, y] = original_value
        # Approximate the gradient
        original_value = adversarial_image[0, 0, :, :]

        # Perturb the pixel value positively
        adversarial_image[0, 0, :, :] = original_value + beta
        output_plus = model(adversarial_image)

        # Perturb the pixel value negatively
        adversarial_image[0, 0, :, :] = original_value - beta
        output_minus = model(adversarial_image)

        # Approximate gradient (central difference)
        gradient[0, 0, :, :] = (output_plus - output_minus) / (2 * beta)

        # Reset pixel value
        adversarial_image[0, 0, :, :] = original_value

        # Update adversarial image
        adversarial_image = torch.clamp(adversarial_image + alpha * np.sign(gradient),  min=-1, max=1)

    return adversarial_image


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data_folder = "./"
    test_data_names = ['IXI-T1-test']
    example_folder = "./adversarial_examples/PGD"
    output_folder = "./adversarial_outputs/PGD"
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
            lr_imgs = lr_imgs.to(device)  # (batch_size (1), 1, w / 4, h / 4), imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (1), 1, w, h), in [-1, 1]
            hr_imgs_y = convert_image(hr_imgs,
                                      source='[-1, 1]',
                                      target='[0, 255]')
            hr_imgs_np = np.squeeze(hr_imgs_y.cpu().numpy())

            # Call PGD Attack
            perturbed_data = zoo_attack(model, images=lr_imgs, labels=hr_imgs)

            # Forward prop.
            adversarial_sr_imgs = model(perturbed_data)  # (1, 1, w, h), in [-1, 1]
            adversarial_sr_imgs = torch.clamp(adversarial_sr_imgs, min=-1, max=1)

            # Convert tensor to PIL Image for perturbed image
            perturbed_img_pil = convert_image(perturbed_data.cpu().detach().squeeze(0), source='[-1, 1]', target='pil')

            # Save perturbed image
            perturbed_img_path = os.path.join(example_folder, f'perturbed_img_{i}.png')
            perturbed_img_pil.save(perturbed_img_path)

            # Convert tensor to PIL Image for adversarial super-resolved image
            adv_sr_img_pil = convert_image(adversarial_sr_imgs.cpu().detach().squeeze(0), source='[-1, 1]',
                                           target='pil')

            # Save adversarial super-resolved image
            adv_sr_img_path = os.path.join(output_folder, f'adversarial_sr_img_{i}.png')
            adv_sr_img_pil.save(adv_sr_img_path)

            # Calculate PSNR and SSIM
            adv_sr_imgs_y = convert_image(adversarial_sr_imgs,
                                          source='[-1, 1]',
                                          target='[0, 255]')
            adv_sr_imgs_np = np.squeeze(adv_sr_imgs_y.cpu().detach().numpy())

            psnr = peak_signal_noise_ratio(hr_imgs_np,
                                           adv_sr_imgs_np,
                                           data_range=255.)
            ssim = structural_similarity(hr_imgs_np,
                                         adv_sr_imgs_np,
                                         data_range=255.)
            PSNRs.update(psnr, lr_imgs.size(0))
            SSIMs.update(ssim, lr_imgs.size(0))

            if i % 100 == 0:
                print(f"{round((i / len(test_loader) * 100), 2)}% done......")

        # Print average PSNR and SSIM
        print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

    print("\n")