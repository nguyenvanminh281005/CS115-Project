from tensorflow.keras.models import load_model
from vaes import encoder, decoder, Sampling, latent_dim, VAE
import numpy as np
import matplotlib.pyplot as plt

# Load VAE model from file
vae_model = load_model(
    'vae.h5',
    custom_objects={
        'Sampling': Sampling,
        'encoder': encoder,
        'decoder': decoder,
        'VAE': VAE,
    },
    compile=False
)

# Define the latent space size
latent_dim = 2  # Replace 2 with the actual latent dimension of your model

# Generate random latent vectors
num_images = 10  # Number of images to generate
random_latent_vectors = np.random.normal(size=(num_images, latent_dim))

# Use the decoder to generate images from the latent vectors
generated_images = decoder.predict(random_latent_vectors)

# Plot the generated images
plt.figure(figsize=(15, 3))
for i in range(num_images):
    ax = plt.subplot(1, num_images, i + 1)
    plt.imshow(generated_images[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()

# Save the generated images to a file
plt.savefig('vae2.png')
