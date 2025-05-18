import argparse
import os

import torch
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="PE-Core-B16-224", choices=pe.CLIP.available_configs())
args = parser.parse_args()

model = pe.CLIP.from_config(args.config, pretrained=True)

preprocess = transforms.get_image_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

cwd = os.getcwd()
path = os.path.join(cwd, args.config + ".package.pt")
if os.path.exists(path):
    print(f"Removing existing package at {path}")
    os.remove(path)

with torch.package.PackageExporter(path) as exporter:
    exporter.intern([
        "core.vision_encoder.pe",
        "core.vision_encoder.transforms",
        "core.vision_encoder.tokenizer",
        "core.vision_encoder.rope",
        "core.vision_encoder.config",
    ])
    exporter.extern([
        "numpy",
        "einops",
        "timm.layers",
        "huggingface_hub",
        "torchvision.transforms",
        "ftfy",
        "regex",
    ])
    exporter.save_pickle("model", "model.pkl", model)
    exporter.save_pickle("get_image_transform", "get_image_transform.pkl", transforms.get_image_transform)
    exporter.save_pickle("get_text_tokenizer", "get_text_tokenizer.pkl", transforms.get_text_tokenizer)

print(f"Package saved to {path}")
