import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

from dquantize_3dvae import use_quantized_3dvae
from models import vae_models
from quantize_vae import use_quantized_vae
from show_img import normalize_to_uint8
from token_redundancy_pipeline import reconstruct_distilled_latents


def choose_frame_ids(num_frames: int, max_frames: int):
    count = min(num_frames, max_frames)
    return np.linspace(0, num_frames - 1, count).round().astype(int).tolist()


def decode_2dvae(latents, vae, device, batch_size):
    flat = rearrange(latents, "b t c h w -> (b t) c h w")
    decoded = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            batch = flat[start:start + batch_size].to(device)
            frames = vae.decode(batch).sample.float()
            decoded.append(((frames + 1) / 2).clamp(0, 1).cpu())
    decoded = torch.cat(decoded, dim=0)
    return rearrange(decoded, "(b t) c h w -> b t c h w", b=latents.shape[0])


def decode_3dvae(latents, vae, device, batch_size, expected_frames):
    batches = rearrange(latents, "b t c h w -> b c t h w")
    decoded = []
    use_half = device.startswith("cuda")
    with torch.no_grad():
        for start in range(0, batches.shape[0], batch_size):
            batch = batches[start:start + batch_size].to(device)
            if use_half:
                batch = batch.half()
            frames = vae.decode(batch).sample.float()
            decoded.append(((frames + 1) / 2).clamp(0, 1).cpu())
    decoded = torch.cat(decoded, dim=0)
    decoded = rearrange(decoded, "b c t h w -> b t c h w").float()
    if decoded.shape[1] < expected_frames:
        pad = decoded[:, -1:].repeat(1, expected_frames - decoded.shape[1], 1, 1, 1)
        decoded = torch.cat([decoded, pad], dim=1)
    return decoded[:, :expected_frames]


def patch_xformers_attention_for_cpu():
    """3DVAE uses xformers attention by default; CPU visualization needs a fallback."""

    def attention_2d(self, h_):
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q, k, v = [rearrange(x, "b c h w -> b (h w) c").unsqueeze(1) for x in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).squeeze(1)
        return rearrange(out, "b (h w) c -> b c h w", b=b, h=h, w=w, c=c)

    def attention_t(self, h_):
        h_ = self.norm_t(h_)
        q = self.q_t(h_).unsqueeze(1)
        k = self.k_t(h_).unsqueeze(1)
        v = self.v_t(h_).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).squeeze(1)
        return self.proj_out_t(out)

    vae_models.MemoryEfficientAttnBlock.attention = attention_2d
    vae_models.MemoryEfficientAttnVideoBlock.attention = attention_2d
    vae_models.MemoryEfficientAttnVideoBlock.attention_t = attention_t


def save_frames(video, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = normalize_to_uint8(video)
    for frame_idx, frame in enumerate(frames):
        Image.fromarray(frame).save(out_dir / f"frame_{frame_idx:02d}.png")


def make_contact_sheet(videos, row_labels, path, max_frames=6, label_width=170):
    if len(videos) == 0:
        return []
    frame_ids = choose_frame_ids(videos[0].shape[0], max_frames)
    first = normalize_to_uint8(videos[0][frame_ids])
    h, w = first.shape[1], first.shape[2]
    frame_gap = 6
    row_gap = 8
    width = label_width + len(frame_ids) * w + (len(frame_ids) - 1) * frame_gap
    height = len(videos) * h + (len(videos) - 1) * row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = None
        small_font = None

    for row, video in enumerate(videos):
        y = row * (h + row_gap)
        draw.text((8, y + max(0, h // 2 - 12)), row_labels[row], fill=(0, 0, 0), font=font)
        frames = normalize_to_uint8(video[frame_ids])
        for col, frame in enumerate(frames):
            x = label_width + col * (w + frame_gap)
            canvas.paste(Image.fromarray(frame), (x, y))
            if row == 0:
                draw.text((x + 4, y + 4), f"t={frame_ids[col]}", fill=(255, 255, 255), font=small_font)

    canvas.save(path)
    canvas.save(path.with_suffix(".tiff"), dpi=(300, 300))
    return frame_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode selected classes from an ImportanceHOSVD/summary-DPP distilled latent artifact."
    )
    parser.add_argument("--artifact", required=True, help="Path to synthetic_data.pt.")
    parser.add_argument("--output_dir", required=True, help="Directory for decoded visualizations.")
    parser.add_argument("--class_id", type=int, default=0, help="Class to decode when --all_classes is not set.")
    parser.add_argument("--all_classes", action="store_true", help="Decode every class in the artifact.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap per class.")
    parser.add_argument("--batch_size", type=int, default=4, help="Decode batch size.")
    parser.add_argument("--contact_frames", type=int, default=6, help="Number of frames per row in contact sheets.")
    parser.add_argument("--rows_per_page", type=int, default=24, help="Rows per contact-sheet page.")
    parser.add_argument("--save_frames", action="store_true", help="Also save all decoded frames per sample.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)

    artifact_path = Path(args.artifact)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    payload = torch.load(artifact_path, map_location="cpu")
    labels = payload["labels"].long().cpu()
    class_ids = labels.unique().tolist() if args.all_classes else [args.class_id]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    vae_model = payload.get("vae_model", "2DVAE")
    print(f"Artifact: {artifact_path}")
    print(f"VAE: {vae_model}; decode device: {device}")
    if vae_model == "3DVAE" and not str(device).startswith("cuda"):
        print("CPU 3DVAE decode detected; using PyTorch attention fallback instead of xformers.")
        patch_xformers_attention_for_cpu()
    print("Reconstructing latent videos from artifact...")
    latents = reconstruct_distilled_latents(payload, SimpleNamespace()).float()
    print(f"Reconstructed latent tensor: {tuple(latents.shape)}")

    if vae_model == "3DVAE":
        vae = use_quantized_3dvae().to(device).eval()
        decode_fn = lambda batch: decode_3dvae(batch, vae, device, args.batch_size, expected_frames=16)
    else:
        vae = use_quantized_vae().to(device).eval()
        decode_fn = lambda batch: decode_2dvae(batch, vae, device, args.batch_size)

    summary = {
        "artifact": str(artifact_path),
        "vae_model": vae_model,
        "decode_device": device,
        "classes": {},
    }

    for class_id in class_ids:
        indices = (labels == int(class_id)).nonzero(as_tuple=True)[0].tolist()
        if args.max_samples is not None:
            indices = indices[:args.max_samples]
        class_dir = output_root / f"class_{int(class_id):03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"Decoding class {class_id}: {len(indices)} sample(s)")

        decoded_videos = []
        row_labels = []
        for local_start in range(0, len(indices), args.batch_size):
            chunk_indices = indices[local_start:local_start + args.batch_size]
            batch = latents[chunk_indices]
            decoded = decode_fn(batch)
            for offset, video in enumerate(decoded):
                global_index = chunk_indices[offset]
                local_index = local_start + offset
                decoded_videos.append(video.cpu().float().clamp(0, 1))
                row_labels.append(f"sample {local_index:02d}")
                if args.save_frames:
                    save_frames(video, class_dir / f"sample_{local_index:02d}_idx_{global_index:04d}")

        pages = []
        for page_id, start in enumerate(range(0, len(decoded_videos), args.rows_per_page)):
            end = min(start + args.rows_per_page, len(decoded_videos))
            page_path = class_dir / f"contact_page_{page_id:02d}.png"
            frame_ids = make_contact_sheet(
                decoded_videos[start:end],
                row_labels[start:end],
                page_path,
                max_frames=args.contact_frames,
            )
            pages.append(str(page_path))

        summary["classes"][str(class_id)] = {
            "num_samples": len(indices),
            "indices": indices,
            "contact_pages": pages,
            "frames_used_in_contact_sheet": frame_ids if indices else [],
        }

    with open(output_root / "decode_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
