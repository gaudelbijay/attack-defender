import torch
import numpy as np
import torchvision
import torch.nn as nn
from model import Unet
from trainer import Trainer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from gaussian_diffusion import GaussianDiffusion


class AttackDetection(object):
    def __init__(self, 
                 image_size=128, 
                 dim=64, 
                 timestamp=1000, 
                 sampling_timesteps=250, 
                 flash_attn=False, 
                 objective='pred_noise',
                 train_root='./data/airsim/raw',
                 test_root='./data/test',
                 batch_size = 16,
                 device = "cuda") -> None:
        
        self.image_size = image_size
        self.dim = dim 
        self.flash_attn = flash_attn 
        self.timestamp = timestamp
        self.sampling_timesteps = sampling_timesteps
        self.objective = objective 
        self.train_root = train_root
        self.test_root = test_root
        self.batch_size = batch_size
        self.device = device
    
    def model_init(self,):
        self.model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            flash_attn = True
            )
        
        self.diffusion = GaussianDiffusion(
        model = self.model,
        image_size = self.image_size,
        timesteps = self.timestamp,           # number of steps
        sampling_timesteps = self.sampling_timesteps,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        objective = self.objective,
        )
        self.trainer = Trainer(self.diffusion, self.train_root)
        self.trainer.load(100) # 100 => last saved model


    def transformation(self,):
        transform = transforms.Compose([
        transforms.Resize(self.image_size), # Resize the input image
        transforms.ToTensor(), # Convert to torch tensor (scales data into [0,1])
        transforms.Lambda(lambda t: (t * 2) - 1), # Scale data between [-1, 1] 
        ])
        return transform

    def dataloader(self,):
        transforms = self.transformation()
        dataset = torchvision.datasets.ImageFolder(self.test_root, transform=transforms)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=6, drop_last=True)
        return dataloader
    
    def fft_batch_images(self, batch_images): # fast fourier transform of batch images
        batch_size, channels, height, width = batch_images.size()
        im = np.float32(batch_images.clone().detach().cpu().numpy()) / 255.0
        
        for i in range(batch_size):
            for j in range(channels):
                pic = im[i, j]
                fft_img = np.fft.fft2(pic)
                fshift = np.fft.fftshift(fft_img)
                magnitude_spectrum = np.abs(fshift)+1
                magnitude_spectrum = np.log(magnitude_spectrum)
                im[i, j] = magnitude_spectrum

        return torch.from_numpy(im).to(self.device)  # Convert back to PyTorch tensor and move to device

    def calculate_magnitude_spectrum(self, batch_images):
        #real_magnitude_spectrum = fft_batch(batch)
        t = torch.tensor([1 for t in batch_images ]).to(self.device)
        predicted_batch = self.diffusion.to(self.device).model_predictions(batch_images, t)[1]
        #predicted_magnitude_spectrum = fft_batch(predicted_batch)
        res = torch.absolute(batch_images - predicted_batch)
        res_fft = self.fft_batch_images(res)
        res_magnitude_spectrum = torch.sum(res_fft, dim = (1, 2,3)) / 3 
        return res_magnitude_spectrum
    

    def calculate_binary_accuracy(self, threshold):
        correct = 0
        total = 0
        ground_truth_labels = []
        predicted_labels = []
        prob_scores = []
        with torch.no_grad():
            dataloader = self.dataloader()
            for batch_images, batch_labels in dataloader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                scores = self.calculate_magnitude_spectrum(batch_images)
                predicted = torch.where(scores < threshold, torch.tensor(0), torch.tensor(1))  # Apply threshold for classification

                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)
                
                predicted_labels.extend(predicted.cpu().numpy())
                ground_truth_labels.extend(batch_labels.cpu().numpy())
                prob_scores.extend(scores.cpu().numpy())
                
        accuracy = correct / total
        return accuracy*100, predicted_labels, ground_truth_labels, prob_scores


if __name__ == '__main__':
    attack_detection = AttackDetection()
    attack_detection.model_init()
    attack_detection.calculate_binary_accuracy(43)