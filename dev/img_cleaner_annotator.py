import os
import sys
import shutil
import cv2
import argparse
import numpy as np

def main_v1():
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

def main():
    # Usage examples:
    #   python image_sorter.py /path/to/input_folder /path/to/output_folder 3
    #   python image_sorter.py "C:\path\to\images" data\sorted_images 2 --keep_original

    parser = argparse.ArgumentParser(description='Manually sort images into categories with recovery.')
    parser.add_argument('input_folder', help='Path to folder with images')
    parser.add_argument('output_folder', help='Path to output folder (will create one subfolder per class)')
    parser.add_argument('num_classes', type=int, help='Number of classes (must be between 2 and 9)')
    parser.add_argument("-k", "--keep_original", default=False, action='store_true', dest='keep_original',
                        help="Keep original images in the input folder. (If using this, recovery is needed to avoid re-annotating images.)")
    args = parser.parse_args()

    if not (2 <= args.num_classes <= 9):
        print("Error: num_classes must be between 2 and 9")
        sys.exit(1)

    # Create one subfolder per class (named "1", "2", etc.)
    for i in range(1, args.num_classes + 1):
        os.makedirs(os.path.join(args.output_folder, str(i)), exist_ok=True)

    # Define progress file path (for recovery)
    progress_file = os.path.join(args.output_folder, "progress.txt")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as pf:
            # Each line is the base name of an image that has already been sorted.
            progress_lines = [line.strip() for line in pf if line.strip()]
        print(f"Resuming annotation. {len(progress_lines)} images were already processed and will be skipped.")
    else:
        progress_lines = []

    # Gather images from input folder
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    images = sorted([
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if f.lower().endswith(valid_exts)
    ])

    # In keep_original mode, the sorted images remain in the input folder,
    # so filter out those that were already processed.
    if args.keep_original:
        images = [img for img in images if os.path.basename(img) not in progress_lines]

    if not images:
        print("No images to sort.")
        sys.exit(1)

    print(f"Loaded {len(images)} images to sort.")
    print(f"Press keys [1-{args.num_classes}] to sort images, 'c' to cancel the last sorting (current session only), or 'q' to quit.")

    # This list is used only for undo in the current session.
    # Each element is a tuple: (original_path, dest_path, category)
    sorted_images = []

    id = 0
    while id < len(images):
        image_path = images[id]
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}. Skipping.")
            id += 1
            continue

        cv2.imshow('Image Sorter', img)
        key = cv2.waitKey(0) & 0xFF

        # Quit if 'q' is pressed.
        if key == ord('q'):
            print("Quitting.")
            break

        # Undo the last sorting (only if there is a move in the current session)
        elif key == ord('c'):
            if sorted_images:
                last_original, last_dest, last_cat = sorted_images.pop()
                print(f"Cancelling last sorting: moving '{os.path.basename(last_dest)}' back to its original location.")
                try:
                    shutil.move(last_dest, last_original)
                except Exception as e:
                    print(f"Error undoing move: {e}")
                # Remove the last entry from the progress file.
                if os.path.exists(progress_file):
                    with open(progress_file, 'r') as pf:
                        lines = pf.read().splitlines()
                    if lines:
                        lines.pop()
                    with open(progress_file, 'w') as pf:
                        pf.write("\n".join(lines) + ("\n" if lines else ""))
                # Go back one image (so the undone image is reprocessed)
                id = max(0, id - 1)
            else:
                print("Nothing to undo from the current session.")
            continue

        # If the pressed key is one of the allowed class keys.
        elif ord('1') <= key <= ord('1') + args.num_classes - 1:
            category = chr(key)
            dest_path = os.path.join(args.output_folder, category, image_name)
            try:
                if args.keep_original:
                    shutil.copyfile(image_path, dest_path)
                    print(f"Copied '{image_name}' to folder '{dest_path}'.")
                else:
                    shutil.move(image_path, dest_path)
                    print(f"Moved '{image_name}' to folder '{dest_path}'.")
                sorted_images.append((image_path, dest_path, category))
                # Append the sorted image name to the progress file.
                with open(progress_file, 'a') as pf:
                    pf.write(image_name + "\n")
                id += 1
            except Exception as e:
                print(f"Error processing file {image_path}: {e}")
        else:
            print("Invalid key pressed. Please press a valid key.")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
