import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dquantize_3dvae import use_quantized_3dvae
from quantize_vae import use_quantized_vae
from show_img import (
    decode_latents_2d,
    decode_latents_3d,
    frame_indices,
    normalize_to_uint8,
)
from token_redundancy_pipeline import reconstruct_distilled_latents


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def load_payload(path):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported artifact format: {path}")
    return payload


def pick_index_by_class(labels: torch.Tensor, class_id: int, rank: int) -> int:
    matches = (labels.long() == int(class_id)).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        raise ValueError(f"No sample with class_id={class_id}.")
    if rank >= matches.numel():
        raise ValueError(f"class_id={class_id} only has {matches.numel()} samples; rank={rank} is invalid.")
    return int(matches[rank].item())


def denormalize_pixel_video(video: torch.Tensor) -> torch.Tensor:
    video = video.detach().cpu().float()
    # Pixel-space baselines are stored in dataset-normalized space. If the values
    # already look like [0, 1], keep them unchanged for compatibility.
    if float(video.min()) < -0.05 or float(video.max()) > 1.05:
        video = video.unsqueeze(0) * IMAGENET_STD + IMAGENET_MEAN
        video = video[0]
    return video.clamp(0, 1)


def decode_ours_video(payload, sample_idx: int, device: str, batch_size: int) -> torch.Tensor:
    if "videos" in payload or "compressed_videos" in payload:
        latents = reconstruct_distilled_latents(payload, SimpleNamespace())
    elif "latents" in payload:
        latents = payload["latents"].float()
    else:
        raise ValueError("Ours artifact must contain compressed latent videos or latents.")

    latent_video = latents[sample_idx: sample_idx + 1]
    vae_model = payload.get("vae_model", "2DVAE")
    if vae_model == "3DVAE":
        vae = use_quantized_3dvae().to(device).half().eval()
        decoded = decode_latents_3d(latent_video, vae, device, batch_size, expected_frames=latent_video.shape[1])
    else:
        vae = use_quantized_vae().to(device).eval()
        decoded = decode_latents_2d(latent_video, vae, device, batch_size)
    return decoded[0].cpu().float().clamp(0, 1)


def load_original_video(frame_dir: str) -> torch.Tensor:
    frame_path = Path(frame_dir)
    if not frame_path.exists():
        raise FileNotFoundError(f"Original frame directory does not exist: {frame_dir}")
    image_files = sorted(
        p for p in frame_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_files:
        raise ValueError(f"No image frames found in original frame directory: {frame_dir}")

    frames = []
    for path in image_files:
        image = Image.open(path).convert("RGB")
        frame = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        frames.append(frame)
    return torch.stack(frames, dim=0).clamp(0, 1)


def vae_roundtrip_pixel_video(video: torch.Tensor, vae_model: str, device: str, batch_size: int) -> torch.Tensor:
    """Pass a pixel-space baseline video through the same VAE path as latent methods."""
    video = denormalize_pixel_video(video)
    if vae_model == "3DVAE":
        vae = use_quantized_3dvae().to(device).half().eval()
        batch = video.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device).half()
        batch = batch * 2 - 1
        with torch.no_grad():
            latents = vae.encode(batch).latent_dist.sample()
            decoded = vae.decode(latents).sample.float()
        decoded = ((decoded + 1) / 2).clamp(0, 1).cpu()
        return decoded.permute(0, 2, 1, 3, 4)[0].float()

    vae = use_quantized_vae().to(device).eval()
    flat = video.to(device) * 2 - 1
    decoded = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            batch = flat[start:start + batch_size]
            latents = vae.encode(batch).latent_dist.sample()
            frames = vae.decode(latents).sample.float()
            decoded.append(((frames + 1) / 2).clamp(0, 1).cpu())
    return torch.cat(decoded, dim=0).float()


def load_pixel_video(payload, sample_idx: int) -> torch.Tensor:
    if "images" not in payload:
        raise ValueError("Pixel baseline artifact must contain an 'images' tensor.")
    return denormalize_pixel_video(payload["images"][sample_idx])


def get_class_name(payloads, class_id: int):
    for payload in payloads:
        class_names = payload.get("class_names")
        if class_names is not None and 0 <= class_id < len(class_names):
            return str(class_names[class_id])
    return f"class_{class_id}"


def resize_video_to(video: torch.Tensor, size_hw):
    h, w = int(size_hw[0]), int(size_hw[1])
    if int(video.shape[-2]) == h and int(video.shape[-1]) == w:
        return video
    return torch.nn.functional.interpolate(
        video.float(),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)


def sample_frames_for_grid(video: torch.Tensor, max_frames: int):
    ids = frame_indices(video.shape[0], min(max_frames, video.shape[0]))
    return normalize_to_uint8(video)[ids]


def load_grid_font(font_size):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def make_comparison_grid(rows, labels, out_path, max_frames=8, pad=6, label_w=128, label_font_size=10):
    if not rows:
        raise ValueError("No rows were provided for comparison grid.")

    target_h, target_w = rows[0].shape[-2], rows[0].shape[-1]
    rows = [resize_video_to(row, (target_h, target_w)) for row in rows]
    frame_arrays = [sample_frames_for_grid(row, max_frames) for row in rows]
    h, w, _ = frame_arrays[0][0].shape
    num_cols = min(len(frames) for frames in frame_arrays)
    frame_arrays = [frames[:num_cols] for frames in frame_arrays]
    grid_w = label_w + num_cols * w + (num_cols - 1) * pad
    grid_h = len(rows) * h + (len(rows) - 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = load_grid_font(label_font_size)

    for row_idx, (frames, label) in enumerate(zip(frame_arrays, labels)):
        y = row_idx * (h + pad)
        try:
            text_box = draw.textbbox((0, 0), label, font=label_font)
            text_h = text_box[3] - text_box[1]
        except AttributeError:
            text_h = draw.textsize(label, font=label_font)[1]
        draw.text((6, y + max(0, (h - text_h) // 2)), label, fill=(0, 0, 0), font=label_font)
        for col_idx, frame in enumerate(frames):
            x = label_w + col_idx * (w + pad)
            canvas.paste(Image.fromarray(frame), (x, y))

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Compare generated frames from Ours and a pixel-space baseline.")
    parser.add_argument("--ours_artifact", required=True, help="ImportanceHOSVD synthetic_data.pt.")
    parser.add_argument("--baseline_artifact", required=True, help="Pixel baseline synthetic_data.pt, e.g. DM.")
    parser.add_argument("--lvdd_artifact", default=None, help="Optional LVDD synthetic_data.pt.")
    parser.add_argument("--original_dir", default=None, help="Optional original video frame directory.")
    parser.add_argument("--baseline_name", default="DM", help="Name shown in the comparison figure.")
    parser.add_argument("--lvdd_name", default="LVDD", help="Name shown for the LVDD row.")
    parser.add_argument("--class_id", type=int, default=0, help="Class id to compare.")
    parser.add_argument("--rank", type=int, default=0, help="Sample rank within the selected class.")
    parser.add_argument(
        "--ours_sample_idx",
        type=int,
        default=None,
        help="Optional exact Ours sample index. If set, class_id is inferred from this sample.",
    )
    parser.add_argument(
        "--lvdd_sample_idx",
        type=int,
        default=None,
        help="Optional exact LVDD sample index. If unset, the script uses class_id/rank.",
    )
    parser.add_argument(
        "--baseline_vae_roundtrip",
        action="store_true",
        help="Encode and decode the pixel baseline through the same VAE before visualization.",
    )
    parser.add_argument(
        "--vae_model",
        default=None,
        choices=["2DVAE", "3DVAE"],
        help="VAE used for baseline round-trip. Defaults to the Ours artifact VAE.",
    )
    parser.add_argument("--max_frames", type=int, default=8, help="Number of frames shown.")
    parser.add_argument("--batch_size", type=int, default=8, help="VAE decode batch size for Ours.")
    parser.add_argument("--device", default="cpu", help="cuda/cpu for decoding Ours.")
    parser.add_argument("--label_font_size", type=int, default=10, help="Font size for row labels in the output grid.")
    parser.add_argument("--output", default="./paper_ours_visualizations/ours_vs_dm_frames.png")
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)
    ours_payload = load_payload(args.ours_artifact)
    baseline_payload = load_payload(args.baseline_artifact)
    lvdd_payload = load_payload(args.lvdd_artifact) if args.lvdd_artifact else None

    ours_labels = ours_payload["labels"].long()
    baseline_labels = baseline_payload["labels"].long()
    lvdd_labels = lvdd_payload["labels"].long() if lvdd_payload is not None else None
    if args.ours_sample_idx is not None:
        if args.ours_sample_idx < 0 or args.ours_sample_idx >= len(ours_labels):
            raise ValueError(f"ours_sample_idx={args.ours_sample_idx} is outside [0, {len(ours_labels) - 1}].")
        ours_idx = int(args.ours_sample_idx)
        class_id = int(ours_labels[ours_idx].item())
        baseline_rank = min(args.rank, int((baseline_labels == class_id).sum().item()) - 1)
        baseline_idx = pick_index_by_class(baseline_labels, class_id, baseline_rank)
    else:
        class_id = int(args.class_id)
        ours_idx = pick_index_by_class(ours_labels, class_id, args.rank)
        baseline_idx = pick_index_by_class(baseline_labels, class_id, min(args.rank, int((baseline_labels == class_id).sum().item()) - 1))

    if lvdd_payload is not None:
        if args.lvdd_sample_idx is not None:
            if args.lvdd_sample_idx < 0 or args.lvdd_sample_idx >= len(lvdd_labels):
                raise ValueError(f"lvdd_sample_idx={args.lvdd_sample_idx} is outside [0, {len(lvdd_labels) - 1}].")
            lvdd_idx = int(args.lvdd_sample_idx)
        else:
            lvdd_rank = min(args.rank, int((lvdd_labels == class_id).sum().item()) - 1)
            lvdd_idx = pick_index_by_class(lvdd_labels, class_id, lvdd_rank)
    else:
        lvdd_idx = None

    class_name = get_class_name([payload for payload in [ours_payload, baseline_payload, lvdd_payload] if payload is not None], class_id)
    print(
        f"Ours sample index: {ours_idx}; baseline sample index: {baseline_idx}; "
        f"class_id={class_id}; class_name={class_name}"
    )
    if lvdd_idx is not None:
        print(f"LVDD sample index: {lvdd_idx}")

    ours_video = decode_ours_video(ours_payload, ours_idx, args.device, args.batch_size)
    baseline_video = load_pixel_video(baseline_payload, baseline_idx)
    if args.baseline_vae_roundtrip:
        vae_model = args.vae_model or ours_payload.get("vae_model", "2DVAE")
        baseline_video = vae_roundtrip_pixel_video(baseline_payload["images"][baseline_idx], vae_model, args.device, args.batch_size)
        baseline_suffix = f"{args.baseline_name} + VAE"
    else:
        baseline_suffix = args.baseline_name

    rows = []
    labels = []
    if args.original_dir:
        rows.append(load_original_video(args.original_dir))
        labels.append("Original")
    rows.append(baseline_video)
    labels.append(baseline_suffix)
    if lvdd_payload is not None:
        rows.append(decode_ours_video(lvdd_payload, lvdd_idx, args.device, args.batch_size))
        labels.append(args.lvdd_name)
    rows.append(ours_video)
    labels.append("Ours")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_comparison_grid(
        rows=rows,
        labels=labels,
        max_frames=args.max_frames,
        out_path=out_path,
        label_font_size=args.label_font_size,
    )
    print(f"Saved comparison figure to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
