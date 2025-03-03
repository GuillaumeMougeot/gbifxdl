import os
import sys
import shutil
import cv2
import argparse


def get_screen_resolution():
    """ Get the screen resolution to resize large images accordingly. """
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()

def resize_image(img, max_width, max_height):
    """ Resize the image to fit within max_width and max_height while keeping
    aspect ratio. """
    h, w = img.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def main():
    parser = argparse.ArgumentParser(
        description='Manually sort images into categories with recovery.')
    parser.add_argument(
        'input_folder', help='Path to folder with images')
    parser.add_argument(
        'output_folder',
        help='Path to output folder (will create one subfolder per class)')
    parser.add_argument(
        'num_classes',
        type=int,
        help='Number of classes (must be between 2 and 9)')
    parser.add_argument(
        "-k",
        "--keep_original",
        default=False,
        action='store_true',
        dest='keep_original',
        help=("Keep original images in the input folder. (If using this, "
              "recovery is needed to avoid re-annotating images.)"))
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
            progress_lines = [line.strip() for line in pf if line.strip()]
        print((f"Resuming annotation. {len(progress_lines)} images were "
               "already processed and will be skipped."))
    else:
        progress_lines = []

    # Gather images from input folder
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    images = sorted([
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if f.lower().endswith(valid_exts)
    ])

    if args.keep_original:
        images = [
            img for img in images if os.path.basename(img) not in progress_lines
        ]

    if not images:
        print("No images to sort.")
        sys.exit(1)

    print(f"Loaded {len(images)} images to sort.")
    print((f"Press keys [1-{args.num_classes}] to sort images, 'c' to cancel "
          "the last sorting, 'f' for fullscreen, or 'q' to quit."))

    sorted_images = []
    screen_width, screen_height = get_screen_resolution()
    fullscreen = False

    id = 0
    while id < len(images):
        image_path = images[id]
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}. Skipping.")
            id += 1
            continue

        img = resize_image(img, screen_width - 100, screen_height - 100)

        window_name = 'Image Sorter'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow manual resizing
        cv2.imshow(window_name, img)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("Quitting.")
            break

        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(
                    window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(
                    window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        elif key == ord('c'):
            if sorted_images:
                last_original, last_dest, last_cat = sorted_images.pop()
                print(("Cancelling last sorting: moving "
                       f"'{os.path.basename(last_dest)}' "
                       "back to its original location."))
                try:
                    shutil.move(last_dest, last_original)
                except Exception as e:
                    print(f"Error undoing move: {e}")
                if os.path.exists(progress_file):
                    with open(progress_file, 'r') as pf:
                        lines = pf.read().splitlines()
                    if lines:
                        lines.pop()
                    with open(progress_file, 'w') as pf:
                        pf.write("\n".join(lines) + ("\n" if lines else ""))
                id = max(0, id - 1)
            else:
                print("Nothing to undo from the current session.")
            continue

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