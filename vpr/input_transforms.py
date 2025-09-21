import torch
import torchvision.transforms as v2


to_tensor = v2.ToTensor()

def make_resize(resize_size: int = 224):
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    return resize

class Normalize:
    imagenet_normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    dinov3_normalize = v2.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )

class CenterCropToSquare:
    def __call__(self, img):
        # Get width and height
        width, height = img.size
        # Determine the size of the square crop
        crop_size = min(width, height)
        # Use torchvision's functional center_crop
        return v2.functional.center_crop(img, crop_size)

center_crop = CenterCropToSquare()


def make_compose_transform(new_size: int = 224, mean_std: str="imagenet", crop: bool=False):
    print('-'*100)
    print(new_size)
    resize = make_resize(new_size)
    if mean_std == "imagenet":
        normalize = Normalize.imagenet_normalize
    elif mean_std == "dinov3":
        normalize = Normalize.dinov3_normalize
    else:
        raise RuntimeError("Invalid mean_std")
    
    if crop:
        return v2.Compose([center_crop, resize, to_tensor,normalize])
    else:
        return v2.Compose([resize, to_tensor, normalize])
