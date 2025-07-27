import cv2
import os
import numpy as np

# Paths
input_folder = 'raw_dataset/Gates raw/'                  # Original images
output_folder = 'sharpened_dataset/Gates sharpened/'           # Final result

os.makedirs(output_folder, exist_ok=True)

# Contrast enhancer
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Sharpening kernel
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Step 1: Apply CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Step 2: Sharpen
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

        # Save
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, sharpened)

        print(f"Enhanced + sharpened: {filename}")
