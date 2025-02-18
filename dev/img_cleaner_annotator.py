import os
import sys
import shutil
import cv2
import argparse
import numpy as np

def main():
    # Usage: python image_sorter.py /path/to/input_folder /path/to/output_folder 3
    # python dev\img_sorter.py "C:\Users\au761367\OneDrive - Aarhus universitet\Datasets\detection\cormorant\austria\Fotos Kormoran\Cormorant" data\classif\cormorants 2

    parser = argparse.ArgumentParser(description='Manually sort images into categories.')
    parser.add_argument('input_folder', help='Path to folder with images')
    parser.add_argument('output_folder', help='Path to output folder (will create one subfolder per class)')
    parser.add_argument('num_classes', type=int, help='Number of classes (must be between 2 and 9)')
    parser.add_argument("-k", "--keep_original", default=False,  action='store_true', dest='keep_original',
        help="Whether to keep the original images in the input folder. Careful if doing this and the annotation is interrupted, the current tool does not allow to restart the annotation where it has been stopped.") 
    args = parser.parse_args()

    if not (2 <= args.num_classes <= 9):
        print("Error: num_classes must be between 2 and 9")
        sys.exit(1)

    # Create subdirectories for each class (named as "1", "2", etc.)
    for i in range(1, args.num_classes + 1):
        class_dir = os.path.join(args.output_folder, str(i))
        os.makedirs(class_dir, exist_ok=True)

    # Gather images from input folder
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    images = sorted([
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if f.lower().endswith(valid_exts)
    ])

    if not images:
        print("No images found in input folder.")
        sys.exit(1)

    print(f"Loaded {len(images)} images.")
    print(f"Press keys [1-{args.num_classes}] to sort images, 'c' to cancel, or 'q' to quit.")

    sorted_images = []

    id = 0
    while id < len(images):
        image_path = images[id]
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}. Skipping.")
            continue

        # Display the image in a window.
        cv2.imshow('Image Sorter', img)
        key = cv2.waitKey(0) & 0xFF  # Wait for a key press

        # Quit if 'q' is pressed.
        if key == ord('q'):
            print("Quitting.")
            break

        elif key == ord('c') and id > 0:
            image_path, dest_path = sorted_images.pop()
            print(f"Cancelling last sorting: {dest_path} is moved back to {image_path}.")
            id -= 1
            shutil.move(dest_path, image_path)

        # Check if the pressed key is one of the allowed class keys.
        elif ord('1') <= key <= ord(str(args.num_classes)):
            class_key = chr(key)
            # dest_dir = os.path.join(args.output_folder, class_key)
            dest_path = os.path.join(args.output_folder, class_key, image_name)
            try:
                if args.keep_original:
                    shutil.copyfile(image_path, dest_path)
                    print(f"Copied '{os.path.basename(image_path)}' to folder '{image_path}'.")
                else:
                    shutil.move(image_path, dest_path)
                    print(f"Moved '{os.path.basename(image_path)}' to folder '{dest_path}'.")
                sorted_images.append((image_path, dest_path))
                id += 1
            except Exception as e:
                print(f"Error moving file {image_path}: {e}")
        else:
            print("Invalid key pressed.")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
