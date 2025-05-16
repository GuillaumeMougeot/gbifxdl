try:
    from flat_bug.predictor import Predictor
except ImportError:
    print("Warning: flat-bug is not installed, please install it before using it. Aborting...")
    exit()
import torch
from PIL import Image
import os
from pathlib import Path
import wget

__all__ = ["Cropper"]

ERDA_MODEL_ZOO_TEMPLATE = "https://anon.erda.au.dk/share_redirect/C1nJdS1jtA/{}"


class Cropper:
    def __init__(
        self,
        cropper_model_path,
        device="cpu",
        dtype=torch.float16,
    ):

        # If the model is not locally, then attempts to download from remote
        if not os.path.exists(cropper_model_path):
            model_path = Path(cropper_model_path)
            model_name = model_path.name
            model_folder = model_path.parent
            os.makedirs(model_folder, exist_ok=True)
            model_url = ERDA_MODEL_ZOO_TEMPLATE.format(model_name)
            print(
                f"Cropping model not found locally, attempting to download it from server at {model_url}"
            )
            cropper_model_path = wget.download(model_url, out=str(model_folder))

        self.cropper = Predictor(
            model=cropper_model_path, device=device, dtype=torch.float16
        )

    def _change_ext(self, path, ext=".png"):
        return str(Path(path).with_suffix(ext))

    def run(self, img_path) -> torch.Tensor:
        with Image.open(img_path) as img:
            img_size = img.size
        scale_before = min(1, 1000 / min(img_size))
        try:
            pred = self.cropper.pyramid_predictions(
                img_path, "", single_scale=True, scale_before=scale_before
            )
        except Exception as e:
            # print(f"Error {e} while cropping image.")
            return
        if len(pred) == 0:
            # print("Empty pred.")
            return
        else:
            crop_weights = (
                pred.confs
                * (pred.boxes[:, 2] - pred.boxes[:, 0] > 0)
                * (pred.boxes[:, 3] - pred.boxes[:, 1] > 0)
            )
            selected_crop = crop_weights.argmax()
            bbox = pred.boxes[selected_crop]
            # x1, y1, x2, y2 = bbox.float()
            bbox = bbox.round().long().tolist()

            # crop = pred.save_crops(mask=True)[selected_crop]
            crop = pred.crops[selected_crop]
            mask = pred.crop_masks[selected_crop]
            output_path = self._change_ext(img_path)
            pred._save_1_crop(crop, mask, output_path)
            return str(Path(output_path).name)


# def save_crop(bbox_crop):
#     # path = join(ctfb_big_image_folder, str(row["speciesKey"]), str(row["filename"]))
#     # bbox_crop = get_crop(cropper, path)
#     if bbox_crop is None:
#         return False
#     else:
#         bbox, crop = bbox_crop
#         out_dir = join(ctfb_small_image_folder, str(row["speciesKey"]))
#         out_path = join(out_dir, str(row["filename"]))
#         os.makedirs(out_dir, exist_ok=True)
#         crop = crop.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
#         im = Image.fromarray(crop)
#         im.save(out_path)
#         return True


def test():
    cropper = Cropper(
        "data/classif/mini/fb_xprize_medium.pt",
        device="cuda:0",
    )

    img_path = "test_img1.jpeg"
    crop = cropper.run(img_path)
    print(crop)


if __name__ == "__main__":
    test()
