import os
from typing import Dict
import albumentations as albu
from albumentations import *
from transformers import AutoTokenizer

#basically for text
def load_tokenizer(source, pretrained_model_name_or_path, cache_dir, **kwargs):
    if source == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(
                os.path.join(cache_dir, f'models--{pretrained_model_name_or_path.replace("/", "--")}')),
            **kwargs,
        )
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = tokenizer.cls_token_id
    else:
        raise KeyError(f"Not supported tokenizer source: {source}")

    return tokenizer

# for images

#data augmentation: need standard augmentation for dominant classes, and increase data size for weaker classes

def get_density_augmentation(img_size = 1344):
    base_Augmentation = [albu.Resize(height=img_size, width=img_size),
                        albu.HorizontalFlip(p=0.5),
                        albu.Normalize(mean = (0.485, 0.465, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2(),]
    #mild texture change for dominant class
    dominant_pipeline = albu.Compose([
        albu.Resize(height=img_size, width=img_size),
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit= 0.1, contrast_limit=0.1, p=0.5),
        albu.Normalize(mean = (0.485, 0.465, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),])
    #heavy texture augmentation
    #jitter, noise and blur to force robustness
    
    rare_pipeline = albu.Compose([
        albu.Resize(height=img_size, width=img_size),
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit= 0.1, contrast_limit=0.1, p=0.5),
        albu.OneOf([
                albu.GaussianBlur(blur_limit=(3,5), p=0.5),
                albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5,1.0), p = 0.5),
                ], p = 0.4),
        albu.GaussNoise(std_range=(0.2,0.44), p = 0.4),    
        albu.Normalize(mean = (0.485, 0.465, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),])
    
    validation_pipeline = albu.Compose(base_Augmentation)
    
    return {"common": dominant_pipeline, "rare": rare_pipeline, "valid": validation_pipeline}


def load_transform(split: str = "train", transform_config: Dict = None):
    img_size = 1344 # force swin resolution
    tfms = get_density_augmentation()
    if split == "valid" or split == "test":
        return tfms["valid"]
    return tfms["common"]

        