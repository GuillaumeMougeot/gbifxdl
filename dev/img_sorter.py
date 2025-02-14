import os
import sys
import shutil
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description='Manually sort images into categories.')
    parser.add_argument('input_folder', help='Path to folder with images')
    parser.add_argument('output_folder', help='Path to output folder (will create one subfolder per class)')
    parser.add_argument('num_classes', type=int, help='Number of classes (must be between 2 and 9)')
    args = parser.parse_args()

    if not (2 <= args.num_classes <= 9):
        print("Error: num_classes must be between 2 and 9")
        sys.exit(1)

    # Create subdirectories for each class (named as "1", "2", etc.)
    for i in range(1, args.num_classes + 1):
        class_dir = os.path.join(args.output_folder, str(i))
        os.makedirs(class_dir, exist_ok=True)

    # Gather images from input folder (you can add or remove extensions)
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
    print(f"Press keys [1-{args.num_classes}] to sort images, or 'q' to quit.")

    for image_path in images:
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

        # Check if the pressed key is one of the allowed class keys.
        if ord('1') <= key <= ord(str(args.num_classes)):
            class_key = chr(key)
            dest_dir = os.path.join(args.output_folder, class_key)
            try:
                shutil.move(image_path, dest_dir)
                print(f"Moved '{os.path.basename(image_path)}' to folder '{dest_dir}'.")
            except Exception as e:
                print(f"Error moving file {image_path}: {e}")
        else:
            print("Invalid key pressed. Skipping image.")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
