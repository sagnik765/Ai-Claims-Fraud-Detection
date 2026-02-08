from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

from PIL import Image, ImageDraw


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _norm(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def _make_image(row: Dict[str, str], size: int = 128) -> Image.Image:
    claim_status = int(float(row.get("claim_status", 0) or 0))
    base = (200, 60, 60) if claim_status == 1 else (60, 110, 200)
    img = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(img)

    vehicle_age = _safe_float(row.get("vehicle_age", "0"))
    customer_age = _safe_float(row.get("customer_age", "0"))
    subscription_length = _safe_float(row.get("subscription_length", "0"))

    bars = [
        _norm(vehicle_age, 20.0),
        _norm(customer_age - 18.0, 70.0),
        _norm(subscription_length, 365.0),
    ]

    bar_w = size // 6
    gap = size // 10
    for i, value in enumerate(bars):
        x0 = gap + i * (bar_w + gap)
        x1 = x0 + bar_w
        y1 = size - 10
        y0 = int(y1 - value * (size - 20))
        color = (30, 30, 30) if claim_status == 1 else (240, 240, 240)
        draw.rectangle([x0, y0, x1, y1], fill=color)

    region_code = int(_safe_float(row.get("region_code", "0")))
    for j in range(region_code % 5):
        y = 10 + j * 8
        draw.line([(10, y), (size - 10, y)], fill=(255, 255, 255), width=1)

    if claim_status == 1:
        draw.ellipse([size - 40, 10, size - 10, 40], outline=(255, 255, 255), width=3)

    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic claim images")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--image-dir", required=True, help="Directory to store images")
    parser.add_argument("--limit", type=int, default=1000, help="Number of images to generate")
    args = parser.parse_args()

    os.makedirs(args.image_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames: List[str] = list(reader.fieldnames or [])
        if "image_paths" not in fieldnames:
            fieldnames.append("image_paths")

        with open(args.output, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for idx, row in enumerate(reader):
                if idx < args.limit:
                    img = _make_image(row)
                    filename = f"claim_{idx:06d}.png"
                    img_path = os.path.join(args.image_dir, filename)
                    img.save(img_path)
                    row["image_paths"] = os.path.relpath(img_path, os.getcwd())
                else:
                    row["image_paths"] = row.get("image_paths", "")
                writer.writerow(row)


if __name__ == "__main__":
    main()
