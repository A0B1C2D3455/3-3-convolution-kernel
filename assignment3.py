import cv2
import numpy as np

# Full path to the image
image = cv2.imread(r'C:\Users\Sharvari\Desktop\python\sample.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("❌ Could not find or open 'sample.png' at the given path!")

# Define convolution kernels
blur_kernel = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

edge_kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

# Convolution function
def custom_convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)

# Apply each kernel
blurred = custom_convolve(image, blur_kernel)
sharpened = custom_convolve(image, sharpen_kernel)
edges = custom_convolve(image, edge_kernel)

# Save output images in the same folder
cv2.imwrite(r'C:\Users\Sharvari\Desktop\python\blur_output.png', blurred)
cv2.imwrite(r'C:\Users\Sharvari\Desktop\python\sharpen_output.png', sharpened)
cv2.imwrite(r'C:\Users\Sharvari\Desktop\python\edge_output.png', edges)

print("✅ Done! Check your Desktop/python folder for:")
print(" - blur_output.png")
print(" - sharpen_output.png")
print(" - edge_output.png")


