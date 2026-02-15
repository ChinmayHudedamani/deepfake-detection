import os
import cv2

REAL_PATH = r"C:\Users\chinm\OneDrive\Desktop\Projects\deepfake_detection\data\real"
FAKE_PATH = r"C:\Users\chinm\OneDrive\Desktop\Projects\deepfake_detection\data\fake"

OUTPUT_REAL = r"C:\Users\chinm\OneDrive\Desktop\Projects\deepfake_detection\data\processed\real"
OUTPUT_FAKE = r"C:\Users\chinm\OneDrive\Desktop\Projects\deepfake_detection\data\processed\fake"
IMG_SIZE = 224  # Standard size for CNN models

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)

        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(file_path)

            if img is None:
                continue

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Save processed image
            save_path = os.path.join(output_folder, f"img_{count}.jpg")
            cv2.imwrite(save_path, img)

            count += 1

    print(f"Processed {count} images from {input_folder}")


print("Processing REAL images...")
process_images(REAL_PATH, OUTPUT_REAL)

print("Processing FAKE images...")
process_images(FAKE_PATH, OUTPUT_FAKE)

print("Image preprocessing completed ✅")
