from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
import torch.nn as nn
import torch


# PGD attack code
def pgd_attack(model, image, label, epsilon=8/255, alpha=2/255, steps=10):
    loss = nn.CrossEntropyLoss()
    adv_image = image.clone().detach()

    # Starting at a uniformly random point

    perturb = torch.distributions.uniform.Uniform(-epsilon, epsilon).sample(adv_image.shape)
    adv_image = adv_image.to(device) + perturb.to(device)
    adv_image = torch.clamp(adv_image, min=0, max=255).detach()

    for _ in range(steps):
        adv_image.requires_grad = True
        outputs = model(adv_image)

        # Calculate loss
        cost = loss(outputs, label)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_image, retain_graph=False, create_graph=False)[0]

        adv_image = adv_image.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
        adv_image = torch.clamp(image + delta, min=0, max=255).detach()

    return adv_image


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
    srresnet_checkpoint = "./checkpoint_srresnet_nifti.pth.tar"

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
            lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]
            # Call PGD Attack
            perturbed_data = pgd_attack(model, image=lr_imgs, label=hr_imgs)

            # Forward prop.
            sr_imgs = model(perturbed_data)  # (1, 3, w, h), in [-1, 1]

            # Convert tensor to PIL Image for perturbed image
            perturbed_img_pil = convert_image(perturbed_data.cpu().detach().squeeze(0), source='[-1, 1]', target='pil')

            # Save perturbed image
            perturbed_img_path = os.path.join(example_folder, f'perturbed_img_{i}.png')
            perturbed_img_pil.save(perturbed_img_path)

            # Convert tensor to PIL Image for adversarial super-resolved image
            adv_sr_img_pil = convert_image(sr_imgs.cpu().detach().squeeze(0), source='[-1, 1]', target='pil')

            # Save adversarial super-resolved image
            adv_sr_img_path = os.path.join(output_folder, f'adversarial_sr_img_{i}.png')
            adv_sr_img_pil.save(adv_sr_img_path)

            # Calculate PSNR and SSIM
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                0)  # (w, h), in y-channel
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().detach().numpy(), sr_imgs_y.cpu().detach().numpy(),
                                        data_range=255.)
            ssim = structural_similarity(hr_imgs_y.cpu().detach().numpy(), sr_imgs_y.cpu().detach().numpy(),
                                        data_range=255.)
            PSNRs.update(psnr, lr_imgs.size(0))
            SSIMs.update(ssim, lr_imgs.size(0))

            if i % 10 == 0:
                print(f"{round((i / len(test_loader) * 100), 2)}% done......")

        # Print average PSNR and SSIM
        print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

    print("\n")