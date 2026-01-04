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

def load_transform(split: str = "train", transform_config: Dict = None):
    img_size = 1344 # force swin resolution
    if split == "train":
        return albu.Compose([
            
            #resize
            albu.Resize(height=img_size, width=img_size),
            
            #flip horizontal, flipping vertically does not make sense medically due to breast sagging
            #flip with a probability of 0.5
            albu.HorizontalFlip(p=0.5),
            
            #texture augmentation
            albu.RandomBrightnessContrast(brightness_limit= 0.1, contrast_limit=0.1, p=0.5),
            albu.OneOf([
                albu.GaussianBlur(blur_limit=(3,5), p=0.5),
                albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5,1.0), p = 0.5),
                
            ], p = 0.4),
            
            albu.GaussNoise(std_range=(0.2,0.44), p = 0.4),
            
            albu.Normalize(mean = (0.485, 0.465, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        #valid / test/ aug
        return albu.Compose([
            albu.Resize(height=img_size, width=img_size),
            albu.Normalize(mean= (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    #this is contradicts my propsal, i need to implement a density aware augmentation
    """assert split in {"train", "valid", "test", "aug"}
    transforms = transform_config[split]
    if split == "train":
        if (transforms["Resize"]["size_h"] == 512 or transforms["Resize"]["size_h"] == 224) and (
                transforms["Resize"]["size_w"] == 512 or transforms["Resize"]["size_w"] == 224):
            return Compose([
                Resize(width=transforms["Resize"]["size_h"], height=transforms["Resize"]["size_w"]),
                HorizontalFlip(),
                VerticalFlip(),
                Affine(
                    rotate=transforms["transform"]["affine_transform_degree"],
                    translate_percent=transforms["transform"]["affine_translate_percent"],
                    scale=transforms["transform"]["affine_scale"],
                    shear=transforms["transform"]["affine_shear"]
                ),
                ElasticTransform(
                    alpha=transforms["transform"]["elastic_transform_alpha"],
                    sigma=transforms["transform"]["elastic_transform_sigma"]
                )
            ], p=transforms["transform"]["p"]
            )
        else:
            return Compose([
                HorizontalFlip(),
                VerticalFlip(),
                Affine(
                    rotate=transforms["transform"]["affine_transform_degree"],
                    translate_percent=transforms["transform"]["affine_translate_percent"],
                    scale=transforms["transform"]["affine_scale"],
                    shear=transforms["transform"]["affine_shear"]
                ),
                ElasticTransform(
                    alpha=transforms["transform"]["elastic_transform_alpha"],
                    sigma=transforms["transform"]["elastic_transform_sigma"]
                )
            ], p=transforms["transform"]["p"]
            )
    elif split == "valid":
        if transforms["Resize"]["size_h"] == 512 and transforms["Resize"]["size_w"] == 512:
            return Compose([
                Resize(width=transforms["Resize"]["size_h"], height=transforms["Resize"]["size_w"])
            ])
            """
