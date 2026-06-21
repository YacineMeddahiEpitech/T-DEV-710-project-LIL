"""
generate_data.py — Batch augmented image creation (data_creation criterion).

Reads images from datasets/train/{NORMAL,PNEUMONIA} and generates N augmented
copies per original image, saving them to datasets/augmented/{NORMAL,PNEUMONIA}.

Augmentations applied per image:
  - Random horizontal flip
  - Random rotation (±20°)
  - Random brightness / contrast adjustment
  - Random zoom (crop + resize, 0.85–1.0 scale)
  - Random translation (±10% of image size)

Usage:
    python generate_data.py --copies 3 --size 64
    python generate_data.py --copies 5 --size 128 --source train
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SRC_BASE  = Path("./datasets")
DEST_BASE = Path("./datasets/augmented")


def augment_image(img: Image.Image, seed: int) -> Image.Image:
    """Apply a random combination of augmentations to a PIL Image."""
    rng = random.Random(seed)
    w, h = img.size

    # 1. Horizontal flip (50%)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Rotation ±20°
    angle = rng.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    # 3. Brightness (0.7 – 1.3)
    factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 4. Contrast (0.8 – 1.2)
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 5. Zoom (crop to 85–100% then resize back)
    scale = rng.uniform(0.85, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    left  = rng.randint(0, w - new_w)
    upper = rng.randint(0, h - new_h)
    img   = img.crop((left, upper, left + new_w, upper + new_h)).resize((w, h), Image.BILINEAR)

    # 6. Translation (shift ±10%)
    tx = int(rng.uniform(-0.10, 0.10) * w)
    ty = int(rng.uniform(-0.10, 0.10) * h)
    img = img.transform(img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty),
                        resample=Image.BILINEAR, fillcolor=0)

    return img


def generate_for_class(src_folder: Path, dest_folder: Path, copies: int, target_size: int):
    dest_folder.mkdir(parents=True, exist_ok=True)
    files = list(src_folder.glob("*.jpeg")) + list(src_folder.glob("*.jpg")) + list(src_folder.glob("*.png"))

    if not files:
        print(f"  No images found in {src_folder}")
        return 0

    count = 0
    for f in files:
        try:
            img = Image.open(f).convert("L").resize((target_size, target_size))
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")
            continue

        stem = f.stem
        for i in range(copies):
            seed = hash(stem) ^ i
            aug  = augment_image(img, seed)
            out  = dest_folder / f"{stem}_aug{i:03d}.jpeg"
            aug.save(out, "JPEG", quality=90)
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Batch augmented image generator")
    parser.add_argument("--copies",  type=int, default=3,    help="Augmented copies per original image (default: 3)")
    parser.add_argument("--size",    type=int, default=64,   help="Output image size in pixels (default: 64)")
    parser.add_argument("--source",  type=str, default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Batch Image Augmentation Generator")
    print(f"  Source  : datasets/{args.source}/{{NORMAL,PNEUMONIA}}")
    print(f"  Dest    : datasets/augmented/{{NORMAL,PNEUMONIA}}")
    print(f"  Copies  : {args.copies} per original image")
    print(f"  Size    : {args.size}×{args.size} px")
    print(f"{'='*60}\n")

    total = 0
    for cls in ["NORMAL", "PNEUMONIA"]:
        src  = SRC_BASE / args.source / cls
        dest = DEST_BASE / cls
        n = generate_for_class(src, dest, args.copies, args.size)
        print(f"  {cls:<12}: {n} augmented images saved → {dest}")
        total += n

    print(f"\nDone. Total augmented images created: {total}")
    print(f"  (original dataset size × {args.copies + 1} = ~{total + total // args.copies} total images)")


if __name__ == "__main__":
    main()
