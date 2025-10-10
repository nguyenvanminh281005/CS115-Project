import torch
from torchvision.models import inception_v3
from torchmetrics.image.inception import InceptionScore
from diffusers import DDPMPipeline, StyleGAN2Pipeline
import torchvision.transforms as transforms
from PIL import Image

# 1. Hàm tiền xử lý ảnh cho InceptionV3
def preprocess_images(images):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 yêu cầu kích thước này
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return torch.stack([transform(img) for img in images])

# 2. Hàm tính Inception Score
def compute_inception_score(images, num_splits=10):
    model = inception_v3(pretrained=True, transform_input=False).eval()
    model.to("cuda")
    metric = InceptionScore(feature_extractor=model, num_splits=num_splits).to("cuda")
    return metric(torch.stack(images))

# 3. Sinh ảnh từ DDPM
def generate_ddpm_images(num_images):
    ddpm = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to("cuda")
    return [transforms.ToTensor()(ddpm(batch_size=1).images[0]) for _ in range(num_images)]

# 4. Sinh ảnh từ GAN (StyleGAN2)
def generate_gan_images(num_images):
    gan = StyleGAN2Pipeline.from_pretrained("AK391/StyleGAN2-ffhq-1024").to("cuda")
    return [transforms.ToTensor()(gan().images[0]) for _ in range(num_images)]

# 5. Sinh ảnh từ VAE
def generate_vae_images(num_images):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda")
    noise = torch.randn((num_images, 3, 64, 64)).to("cuda")
    return [transforms.ToTensor()(vae.decode(noise)[0].cpu()) for _ in range(num_images)]

# 6. So sánh các mô hình
def compare_models_inception_score(num_images=50):
    print("Generating images...")
    ddpm_images = generate_ddpm_images(num_images)
    gan_images = generate_gan_images(num_images)
    vae_images = generate_vae_images(num_images)

    print("Preprocessing images...")
    ddpm_images = preprocess_images(ddpm_images).to("cuda")
    gan_images = preprocess_images(gan_images).to("cuda")
    vae_images = preprocess_images(vae_images).to("cuda")

    print("Computing Inception Score...")
    ddpm_score = compute_inception_score(ddpm_images)
    gan_score = compute_inception_score(gan_images)
    vae_score = compute_inception_score(vae_images)

    print(f"Inception Score for DDPM: {ddpm_score}")
    print(f"Inception Score for GAN: {gan_score}")
    print(f"Inception Score for VAE: {vae_score}")

# Chạy so sánh
compare_models_inception_score(num_images=50)
