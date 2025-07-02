import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter
from clize import run
from log import logger


def generate(*, width: 'w' = 600, height: 'e' = 600, output_filename: 'o' = 'image-600x600.bin') -> None: # noqa F601
    """
    Generates a grayscale image with some simple shapes and adds speckle-like noise,
    mimicking basic ultrasound image characteristics.

    :param width: of the generated image
    :param height: of the generated image
    :param output_filename: of the generated image
    """
    data = np.zeros((height, width), dtype=np.float32)

    # Add a bright circle (simulating a cyst or vessel)
    center_y, center_x = height // 2, width // 2
    radius = min(width, height) // 5
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    data[dist_from_center < radius] = 200  # Bright intensity

    # Add a darker rectangle (simulating a shadow or different tissue)
    data[height//4:height//3, width//3:width*2//3] = 50

    # Add some basic gradient (simulating depth attenuation)
    for row in range(height):
        data[row, :] += (row / height) * 50  # Darker at bottom

    # Add speckle noise (Gaussian noise squared and scaled)
    gaussian_noise = np.random.normal(loc=0, scale=20.0, size=(height, width))
    speckle_amplitude = gaussian_noise**2
    speckle_amplitude = speckle_amplitude / speckle_amplitude.max() * 50  # Scale noise contribution
    data += speckle_amplitude

    # Ensure values are within 0-255 and convert to uint8
    data = np.clip(data, 0, 255).astype(np.uint8)

    data.tofile(output_filename)
    logger.success(f"Generated {width}x{height} (approx {data.nbytes / (1024*1024):.2f} MB) "
                   f"structured pixel data to {output_filename}")

    if width * height < 1000000:
        plt.imshow(data, cmap='gray', vmin=0, vmax=255)
        plt.title("Structured Ultrasound-like Image (Simulated)")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    run(generate)
