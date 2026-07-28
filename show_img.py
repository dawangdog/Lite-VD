import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange

from dquantize_3dvae import use_quantized_3dvae
from models import vae_models
from quantize_vae import use_quantized_vae
from token_redundancy_pipeline import reconstruct_distilled_latents
from utils import get_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find_latest_artifact(root: str) -> str:
    candidates = sorted(
        Path(root).glob("**/LongVideoToken_*_ImportanceHOSVD_*/**/synthetic_data.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No ImportanceHOSVD synthetic_data.pt found under {root}. "
            "Please pass --artifact explicitly."
        )
    return str(candidates[-1])


def load_class_names(csv_path: str):
    if not csv_path or not os.path.exists(csv_path):
        return None

    class_names = []
    seen = set()
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[1].strip()
            if name and name not in seen:
                seen.add(name)
                class_names.append(name)
    return sorted(class_names)


def infer_dataset_csv(artifact_path: str):
    path = artifact_path.lower()
    if "hmdb51" in path:
        return "./distill_utils/data/HMDB51/hmdb51_splits1.csv"
    return "./distill_utils/data/UCF101/ucf50_splits1.csv"


def infer_dataset_name(artifact_path: str):
    path = artifact_path.lower()
    if "hmdb51" in path:
        return "HMDB51"
    if "miniucf101" in path:
        return "miniUCF101"
    if "ucf101" in path:
        return "UCF101"
    return None


def normalize_to_uint8(frames: torch.Tensor) -> np.ndarray:
    frames = frames.detach().cpu().float().clamp(0, 1)
    frames = (frames * 255.0).round().clamp(0, 255).byte()
    return frames.permute(0, 2, 3, 1).numpy()


def denormalize_dataset_video(video: torch.Tensor) -> torch.Tensor:
    video = video.detach().cpu().float()
    if float(video.min()) < -0.05 or float(video.max()) > 1.05:
        video = video * IMAGENET_STD + IMAGENET_MEAN
    return video.clamp(0, 1)


def decode_latents_2d(latent_videos: torch.Tensor, vae, device: str, batch_size: int):
    flat_latents = rearrange(latent_videos, "b t c h w -> (b t) c h w")
    decoded = []
    with torch.no_grad():
        for start in range(0, flat_latents.shape[0], batch_size):
            batch = flat_latents[start:start + batch_size].to(device)
            frames = vae.decode(batch).sample.float()
            frames = ((frames + 1) / 2).clamp(0, 1)
            decoded.append(frames.cpu())
    decoded = torch.cat(decoded, dim=0)
    return rearrange(decoded, "(b t) c h w -> b t c h w", b=latent_videos.shape[0])


def decode_latents_3d(latent_videos: torch.Tensor, vae, device: str, batch_size: int, expected_frames: int):
    latent_batches = rearrange(latent_videos, "b t c h w -> b c t h w")
    decoded = []
    use_half = str(device).startswith("cuda")
    with torch.no_grad():
        for start in range(0, latent_batches.shape[0], batch_size):
            batch = latent_batches[start:start + batch_size].to(device)
            if use_half:
                batch = batch.half()
            frames = vae.decode(batch).sample.float()
            frames = ((frames + 1) / 2).clamp(0, 1)
            decoded.append(frames.cpu())
    decoded = torch.cat(decoded, dim=0)
    decoded = rearrange(decoded, "b c t h w -> b t c h w").float()
    if decoded.shape[1] < expected_frames:
        pad = decoded[:, -1:].repeat(1, expected_frames - decoded.shape[1], 1, 1, 1)
        decoded = torch.cat([decoded, pad], dim=1)
    return decoded[:, :expected_frames]


def patch_xformers_attention_for_cpu():
    """Use PyTorch attention when visualizing 3DVAE artifacts without CUDA."""

    def attention_2d(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q, k, v = [rearrange(x, "b c h w -> b (h w) c").unsqueeze(1) for x in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).squeeze(1)
        return rearrange(out, "b (h w) c -> b c h w", b=b, h=h, w=w, c=c)

    def attention_t(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm_t(h_)
        q = self.q_t(h_).unsqueeze(1)
        k = self.k_t(h_).unsqueeze(1)
        v = self.v_t(h_).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).squeeze(1)
        return self.proj_out_t(out)

    vae_models.MemoryEfficientAttnBlock.attention = attention_2d
    vae_models.MemoryEfficientAttnVideoBlock.attention = attention_2d
    vae_models.MemoryEfficientAttnVideoBlock.attention_t = attention_t


def build_token_mask(indices: torch.Tensor, shape):
    t, _c, h, w = shape
    mask = torch.zeros(t * h * w, dtype=torch.float32)
    if indices is not None and indices.numel() > 0:
        mask[indices.long().cpu()] = 1.0
    return mask.view(t, h, w)


def resize_mask(mask: torch.Tensor, out_hw):
    mask = mask.unsqueeze(1)
    resized = F.interpolate(mask, size=out_hw, mode="nearest")
    return resized[:, 0]


def heatmap_color(values: np.ndarray, color=(255, 48, 32)):
    values = np.clip(values, 0.0, 1.0)
    heat = np.zeros((*values.shape, 3), dtype=np.uint8)
    heat[..., 0] = (values * color[0]).astype(np.uint8)
    heat[..., 1] = (values * color[1]).astype(np.uint8)
    heat[..., 2] = (values * color[2]).astype(np.uint8)
    return heat


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color, alpha: float):
    color_img = np.zeros_like(frame, dtype=np.float32)
    color_img[..., 0] = color[0]
    color_img[..., 1] = color[1]
    color_img[..., 2] = color[2]
    mask = mask[..., None].astype(np.float32)
    blended = frame.astype(np.float32) * (1 - alpha * mask) + color_img * (alpha * mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_visible_content_mask(frames_np: np.ndarray, threshold: int) -> np.ndarray:
    """Suppress VAE/video letterbox padding when projecting latent masks to pixels."""
    max_rgb = frames_np.astype(np.float32).max(axis=-1)
    return (max_rgb > float(threshold)).astype(np.float32)


def apply_content_mask(mask: np.ndarray, content_mask: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return mask
    return mask * content_mask


def make_grid(frames, pad=6, label=None):
    if len(frames) == 0:
        raise ValueError("No frames were provided to make_grid.")
    h, w, _ = frames[0].shape
    label_h = 24 if label else 0
    grid_h = h + label_h
    grid_w = len(frames) * w + (len(frames) - 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if label:
        draw.text((4, 4), label, fill=(0, 0, 0))
    for idx, frame in enumerate(frames):
        x = idx * (w + pad)
        canvas.paste(Image.fromarray(frame), (x, label_h))
    return canvas


def load_grid_font(font_size):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def make_method_frame_grid(rows, row_labels, chosen_frames, out_path, pad=6, label_w=132, label_h=22, label_font_size=20):
    frame_arrays = [[normalize_to_uint8(row)[idx] for idx in chosen_frames] for row in rows]
    h, w, _ = frame_arrays[0][0].shape
    grid_w = label_w + len(chosen_frames) * w + (len(chosen_frames) - 1) * pad
    grid_h = label_h + len(rows) * h + (len(rows) - 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = load_grid_font(label_font_size)
    header_font = load_grid_font(max(10, int(label_font_size * 0.8)))

    for col_idx, frame_id in enumerate(chosen_frames):
        x = label_w + col_idx * (w + pad)
        draw.text((x + 3, 4), f"t={frame_id}", fill=(70, 70, 70), font=header_font)

    for row_idx, (frames, label) in enumerate(zip(frame_arrays, row_labels)):
        y = label_h + row_idx * (h + pad)
        try:
            text_box = draw.textbbox((0, 0), label, font=label_font)
            text_h = text_box[3] - text_box[1]
        except AttributeError:
            text_h = draw.textsize(label, font=label_font)[1]
        draw.text((6, y + max(0, (h - text_h) // 2)), label, fill=(0, 0, 0), font=label_font)
        for col_idx, frame in enumerate(frames):
            x = label_w + col_idx * (w + pad)
            canvas.paste(Image.fromarray(frame), (x, y))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    canvas.save(out_path.with_suffix(".tiff"), dpi=(300, 300))
    return out_path


def save_frames(frames_np, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for idx, frame in enumerate(frames_np):
        Image.fromarray(frame).save(os.path.join(out_dir, f"frame_{idx:02d}.png"))


def load_distill_state(artifact_path: str, artifact):
    state_path = Path(artifact_path).with_name("distill_state.pt")
    if state_path.exists():
        try:
            return torch.load(state_path, map_location="cpu")
        except Exception as exc:
            print(f"Warning: failed to load distill_state.pt ({exc}).")
    return None


def pick_sample_ids(labels: torch.Tensor, args):
    if args.sample_ids:
        return [int(x.strip()) for x in args.sample_ids.split(",") if x.strip()]

    if args.class_ids:
        ids = []
        for class_id in [int(x.strip()) for x in args.class_ids.split(",") if x.strip()]:
            class_indices = (labels == class_id).nonzero(as_tuple=True)[0]
            ids.extend(class_indices[:args.samples_per_class].tolist())
        return ids

    ids = []
    for class_id in labels.unique().tolist()[:args.num_classes]:
        class_indices = (labels == class_id).nonzero(as_tuple=True)[0]
        ids.extend(class_indices[:args.samples_per_class].tolist())
    return ids[:args.max_samples]


def get_class_label(label_id: int, class_names):
    if class_names is not None and 0 <= label_id < len(class_names):
        return f"{label_id}_{class_names[label_id]}"
    return f"class_{label_id}"


def frame_indices(total_frames: int, max_frames: int):
    if total_frames <= max_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, max_frames).round().astype(int).tolist()


def visualize_sample(sample_id, artifact, decoded_video, importance_map, class_names, args):
    record = artifact["videos"][sample_id] if "videos" in artifact else None
    if args.flip_output:
        decoded_video = torch.flip(decoded_video, dims=[-1])
        if importance_map is not None:
            importance_map = torch.flip(importance_map, dims=[-1])
    labels = artifact["labels"].long()
    label_id = int(labels[sample_id].item())
    class_label = get_class_label(label_id, class_names)
    sample_dir = Path(args.output_dir) / f"sample_{sample_id:04d}_{class_label}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    frames_np = normalize_to_uint8(decoded_video)
    chosen = frame_indices(frames_np.shape[0], args.max_frames)
    frame_hw = (frames_np.shape[1], frames_np.shape[2])

    save_frames(frames_np, sample_dir / "frames")
    make_grid([frames_np[i] for i in chosen], label=f"decoded frames | sample={sample_id} | {class_label}").save(
        sample_dir / "decoded_grid.png"
    )

    metadata = {
        "sample_id": int(sample_id),
        "label_id": label_id,
        "label": class_label,
        "num_frames": int(frames_np.shape[0]),
    }

    content_mask = build_visible_content_mask(frames_np, args.content_threshold)
    if args.suppress_black_borders:
        metadata["black_border_suppression"] = {
            "enabled": True,
            "content_threshold": int(args.content_threshold),
            "visible_pixel_ratio": float(content_mask.mean()),
        }
    else:
        metadata["black_border_suppression"] = {"enabled": False}

    if record is not None:
        shape = tuple(record["shape"])
        high_mask = resize_mask(build_token_mask(record["high_precision"]["indices"], shape), frame_hw)
        medium_mask = resize_mask(build_token_mask(record["medium_precision"]["indices"], shape), frame_hw)
        high_np = apply_content_mask(high_mask.numpy(), content_mask, args.suppress_black_borders)
        medium_np = apply_content_mask(medium_mask.numpy(), content_mask, args.suppress_black_borders)

        high_overlays = [overlay_mask(frames_np[i], high_np[i], color=(255, 32, 32), alpha=args.alpha) for i in chosen]
        medium_overlays = [overlay_mask(frames_np[i], medium_np[i], color=(255, 180, 20), alpha=args.alpha) for i in chosen]
        both_overlays = []
        for i in chosen:
            frame = overlay_mask(frames_np[i], medium_np[i], color=(255, 180, 20), alpha=args.alpha * 0.8)
            frame = overlay_mask(frame, high_np[i], color=(255, 32, 32), alpha=args.alpha)
            both_overlays.append(frame)

        make_grid(high_overlays, label="high precision token mask (red)").save(sample_dir / "high_token_overlay.png")
        make_grid(medium_overlays, label="medium precision token mask (orange)").save(sample_dir / "medium_token_overlay.png")
        make_grid(both_overlays, label="high + medium token mask").save(sample_dir / "precision_overlay.png")

        metadata.update(
            {
                "high_precision_tokens": int(record["high_precision"]["indices"].numel()),
                "medium_precision_tokens": int(record["medium_precision"]["indices"].numel()),
                "base_type": record["base"].get("type", "unknown"),
                "ranks": record["base"].get("ranks", None),
            }
        )

    if importance_map is not None:
        importance = resize_mask(importance_map.float(), frame_hw)
        imp = apply_content_mask(importance.numpy(), content_mask, args.suppress_black_borders)
        imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
        heatmaps = []
        overlays = []
        for i in chosen:
            heat = heatmap_color(imp[i], color=(255, 64, 0))
            heatmaps.append(heat)
            overlays.append(np.clip(frames_np[i] * 0.55 + heat * 0.45, 0, 255).astype(np.uint8))
        make_grid(heatmaps, label="token importance heatmap").save(sample_dir / "importance_heatmap.png")
        make_grid(overlays, label="importance overlay").save(sample_dir / "importance_overlay.png")
        metadata["importance_mean"] = float(importance_map.float().mean().item())

    with open(sample_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return metadata


def visualize_pixel_artifact(artifact_path: str, artifact, class_names, args):
    if "images" not in artifact or "labels" not in artifact:
        raise ValueError("Pixel artifact must contain `images` and `labels`.")

    videos = artifact["images"].float()
    labels = artifact["labels"].long()
    sample_ids = pick_sample_ids(labels, args)
    valid_sample_ids = [idx for idx in sample_ids if 0 <= idx < videos.shape[0]]
    if len(valid_sample_ids) < len(sample_ids):
        invalid = sorted(set(sample_ids) - set(valid_sample_ids))
        print(f"Skipping invalid sample ids: {invalid}")
    if not valid_sample_ids:
        raise ValueError("No valid pixel samples were selected for visualization.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"Pixel artifact: {tuple(videos.shape)}; visualized subset: {len(valid_sample_ids)}; "
        f"sample ids: {valid_sample_ids}"
    )

    summaries = []
    for sample_id in valid_sample_ids:
        decoded_video = denormalize_dataset_video(videos[sample_id])
        summaries.append(visualize_sample(sample_id, artifact, decoded_video, None, class_names, args))

    with open(Path(args.output_dir) / "visualization_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "artifact": artifact_path,
                "format": artifact.get("format", "pixel_artifact"),
                "method": artifact.get("method", "pixel"),
                "num_visualized": len(summaries),
                "samples": summaries,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved pixel visualizations to: {os.path.abspath(args.output_dir)}")


def load_latent_artifact_for_visualization(path: str):
    payload = torch.load(path, map_location="cpu")
    if "videos" in payload or "compressed_videos" in payload:
        latents = reconstruct_distilled_latents(payload, SimpleNamespace())
    elif "latents" in payload:
        latents = payload["latents"].float()
    else:
        raise ValueError(f"Unsupported latent comparison artifact: {path}")
    return payload, latents.float()


def selected_original_index(artifact, sample_id: int):
    selected = artifact.get("selected_indices")
    if selected is None:
        return None
    if isinstance(selected, list):
        if sample_id >= len(selected):
            return None
        return int(selected[sample_id])
    if isinstance(selected, torch.Tensor):
        if sample_id >= selected.numel():
            return None
        return int(selected.view(-1)[sample_id].item())
    return None


def find_comparison_sample(ours_artifact, comparison_artifact, ours_sample_id: int, allow_class_fallback: bool):
    original_idx = selected_original_index(ours_artifact, ours_sample_id)
    labels = ours_artifact["labels"].long()
    target_label = int(labels[ours_sample_id].item())

    comparison_selected = comparison_artifact.get("selected_indices")
    if comparison_selected is not None and original_idx is not None:
        if isinstance(comparison_selected, list):
            comparison_selected = torch.tensor(comparison_selected, dtype=torch.long)
        comparison_selected = comparison_selected.long().view(-1)
        matches = (comparison_selected == int(original_idx)).nonzero(as_tuple=True)[0]
        if matches.numel() > 0:
            return int(matches[0].item()), int(original_idx), "same_selected_index"

    if not allow_class_fallback:
        raise ValueError(
            "Could not find the same original video in the comparison artifact. "
            "Pass --allow_class_fallback to compare by class instead, or use a comparison artifact "
            "created from the same selected indices/distill_state."
        )

    comparison_labels = comparison_artifact["labels"].long()
    ours_class_positions = (labels == target_label).nonzero(as_tuple=True)[0]
    comparison_class_positions = (comparison_labels == target_label).nonzero(as_tuple=True)[0]
    if comparison_class_positions.numel() == 0:
        raise ValueError(f"Comparison artifact has no sample for class_id={target_label}.")
    rank = (ours_class_positions == int(ours_sample_id)).nonzero(as_tuple=True)[0]
    rank = int(rank[0].item()) if rank.numel() else 0
    rank = min(rank, comparison_class_positions.numel() - 1)
    return int(comparison_class_positions[rank].item()), original_idx, "class_fallback"


def build_importance_overlay(decoded_video: torch.Tensor, importance_map: torch.Tensor, args) -> torch.Tensor:
    frames_np = normalize_to_uint8(decoded_video)
    frame_hw = (frames_np.shape[1], frames_np.shape[2])
    importance = resize_mask(importance_map.float(), frame_hw)
    imp = importance.numpy()

    content_mask = build_visible_content_mask(frames_np, args.content_threshold)
    imp = apply_content_mask(imp, content_mask, args.suppress_black_borders)
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)

    overlays = []
    for i in range(frames_np.shape[0]):
        heat = heatmap_color(imp[i], color=(255, 64, 0))
        overlays.append(np.clip(frames_np[i] * 0.55 + heat * 0.45, 0, 255).astype(np.uint8))
    overlays = np.stack(overlays, axis=0)
    return torch.from_numpy(overlays).permute(0, 3, 1, 2).float() / 255.0


def save_paper_comparison(
    sample_id: int,
    original_video: torch.Tensor,
    comparison_video: torch.Tensor,
    ours_video: torch.Tensor,
    importance_map: torch.Tensor,
    class_label: str,
    match_mode: str,
    comparison_label: str,
    args,
):
    total_frames = min(original_video.shape[0], comparison_video.shape[0], ours_video.shape[0])
    chosen = frame_indices(total_frames, args.max_frames)
    overlay_video = build_importance_overlay(ours_video[:total_frames], importance_map, args)

    sample_dir = Path(args.output_dir) / f"paper_compare_sample_{sample_id:04d}_{class_label}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_dir / args.comparison_output_name
    make_method_frame_grid(
        rows=[
            original_video[:total_frames],
            comparison_video[:total_frames],
            ours_video[:total_frames],
            overlay_video[:total_frames],
        ],
        row_labels=["Original", comparison_label, "Ours", "Importance"],
        chosen_frames=chosen,
        out_path=out_path,
    )
    with open(sample_dir / "paper_comparison_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample_id": int(sample_id),
                "class_label": class_label,
                "match_mode": match_mode,
                "comparison_label": comparison_label,
                "chosen_frames": [int(x) for x in chosen],
                "output": str(out_path),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize ImportanceHOSVD distilled videos and token masks.")
    parser.add_argument("--artifact", type=str, default=None, help="Path to synthetic_data.pt.")
    parser.add_argument("--search_root", type=str, default="./logged_files", help="Root used when --artifact is omitted.")
    parser.add_argument("--output_dir", type=str, default="./paper_ours_visualizations", help="Where visualizations are saved.")
    parser.add_argument("--data_path", type=str, default="./distill_utils/data", help="Dataset root used for original videos.")
    parser.add_argument("--csv_path", type=str, default=None, help="Optional dataset csv for class names.")
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma-separated artifact sample ids, e.g. 0,24,48.")
    parser.add_argument("--class_ids", type=str, default=None, help="Comma-separated class ids to visualize.")
    parser.add_argument("--samples_per_class", type=int, default=1, help="Samples per class when using --class_ids.")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of leading classes used by default.")
    parser.add_argument("--max_samples", type=int, default=6, help="Maximum samples visualized by default.")
    parser.add_argument("--max_frames", type=int, default=8, help="Maximum frames shown in each grid.")
    parser.add_argument("--batch_size", type=int, default=8, help="VAE decode batch size.")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu. Defaults to cuda if available.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay opacity.")
    parser.add_argument(
        "--flip_output",
        action="store_true",
        help="Horizontally flip decoded frames and token maps for visualization only.",
    )
    parser.add_argument(
        "--paper_compare",
        action="store_true",
        help="Also save Original | comparison | Ours | Importance same-video paper comparison grids.",
    )
    parser.add_argument(
        "--comparison_artifact",
        type=str,
        default=None,
        help="UniformHOSVD/LVDD synthetic_data.pt used in --paper_compare mode.",
    )
    parser.add_argument(
        "--comparison_label",
        type=str,
        default="UniformHOSVD",
        help="Row label for the comparison artifact in paper comparison grids.",
    )
    parser.add_argument(
        "--comparison_output_name",
        type=str,
        default="paper_comparison_grid.png",
        help="Output filename inside each paper comparison sample directory.",
    )
    parser.add_argument(
        "--allow_class_fallback",
        action="store_true",
        help="If same selected_indices cannot be matched, fall back to same-class comparison.",
    )
    parser.add_argument(
        "--content_threshold",
        type=int,
        default=25,
        help="Pixel threshold used to detect pure black letterbox/padding regions.",
    )
    parser.add_argument(
        "--no_suppress_black_borders",
        dest="suppress_black_borders",
        action="store_false",
        help="Do not remove pure black padding regions from mask overlays.",
    )
    parser.set_defaults(suppress_black_borders=True)
    args = parser.parse_args()

    original_cwd = Path.cwd()
    if args.artifact:
        artifact_candidate = Path(args.artifact)
        artifact_path = artifact_candidate if artifact_candidate.is_absolute() else original_cwd / artifact_candidate
    else:
        search_root = Path(args.search_root)
        if not search_root.is_absolute():
            cwd_root = original_cwd / search_root
            script_root = SCRIPT_DIR / search_root
            search_root = cwd_root if cwd_root.exists() else script_root
        artifact_path = Path(find_latest_artifact(str(search_root)))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = original_cwd / output_dir
        args.output_dir = str(output_dir)

    # Quantized VAE loaders use repo-relative paths such as ./quantized_vae.
    os.chdir(SCRIPT_DIR)

    artifact_path = os.path.abspath(artifact_path)
    print(f"Loading artifact: {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu")

    csv_path = args.csv_path or infer_dataset_csv(artifact_path)
    class_names = artifact.get("class_names") or load_class_names(csv_path)
    if class_names:
        print(f"Loaded {len(class_names)} class names")

    if isinstance(artifact, torch.Tensor):
        artifact = {
            "format": "pixel_tensor",
            "method": "pixel",
            "images": artifact.float(),
            "labels": torch.arange(artifact.shape[0]).long(),
        }

    if "images" in artifact and "labels" in artifact:
        visualize_pixel_artifact(artifact_path, artifact, class_names, args)
        return

    if "videos" in artifact or "compressed_videos" in artifact:
        vae_model = artifact.get("vae_model", "2DVAE")
        latent_videos = reconstruct_distilled_latents(artifact, SimpleNamespace())
    elif "latents" in artifact:
        vae_model = artifact.get("vae_model", "2DVAE")
        latent_videos = artifact["latents"].float()
    else:
        raise ValueError("Unsupported artifact format. Expected ImportanceHOSVD synthetic_data.pt.")

    labels = artifact["labels"].long()
    state = load_distill_state(artifact_path, artifact)
    selected_importance = None
    if state is not None and "selected_importance" in state:
        selected_importance = state["selected_importance"].float()

    sample_ids = pick_sample_ids(labels, args)
    valid_sample_ids = [idx for idx in sample_ids if 0 <= idx < latent_videos.shape[0]]
    if len(valid_sample_ids) < len(sample_ids):
        invalid = sorted(set(sample_ids) - set(valid_sample_ids))
        print(f"Skipping invalid sample ids: {invalid}")
    if not valid_sample_ids:
        raise ValueError("No valid samples were selected for visualization.")
    latent_subset = latent_videos[valid_sample_ids]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"VAE model: {vae_model}; latent artifact: {tuple(latent_videos.shape)}; "
        f"visualized subset: {tuple(latent_subset.shape)}; device: {device}"
    )
    if vae_model == "3DVAE":
        if not str(device).startswith("cuda"):
            print("CPU 3DVAE visualization detected; using PyTorch attention fallback instead of xformers.")
            patch_xformers_attention_for_cpu()
            vae = use_quantized_3dvae().to(device).eval()
        else:
            vae = use_quantized_3dvae().to(device).half().eval()
        decoded = decode_latents_3d(latent_subset, vae, device, args.batch_size, expected_frames=latent_videos.shape[1])
    else:
        vae = use_quantized_vae().to(device).eval()
        decoded = decode_latents_2d(latent_subset, vae, device, args.batch_size)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Visualizing sample ids: {valid_sample_ids}")

    comparison_artifact = None
    comparison_latents = None
    dst_train = None
    if args.paper_compare:
        if not args.comparison_artifact:
            raise ValueError("--paper_compare requires --comparison_artifact.")
        if selected_importance is None:
            raise ValueError("--paper_compare requires distill_state.pt with selected_importance for importance overlays.")

        comparison_path = Path(args.comparison_artifact)
        if not comparison_path.is_absolute():
            comparison_path = original_cwd / comparison_path
        comparison_artifact, comparison_latents = load_latent_artifact_for_visualization(str(comparison_path))
        comparison_vae_model = comparison_artifact.get("vae_model", vae_model)
        if comparison_vae_model != vae_model:
            raise ValueError(
                f"Comparison artifact uses {comparison_vae_model}, but Ours uses {vae_model}. "
                "Use artifacts created with the same VAE for same-video visualization."
            )

        dataset_name = artifact.get("dataset") or infer_dataset_name(artifact_path)
        if not dataset_name:
            raise ValueError("The Ours artifact does not record a dataset name.")
        _, _, _, _, _, _, dst_train, _, _ = get_dataset(dataset_name, args.data_path, num_workers=0)
        print(f"Loaded original training dataset for paper comparison: {dataset_name}")

    summaries = []
    for local_id, sample_id in enumerate(valid_sample_ids):
        importance_map = selected_importance[sample_id] if selected_importance is not None else None
        summaries.append(visualize_sample(sample_id, artifact, decoded[local_id], importance_map, class_names, args))

        if args.paper_compare:
            comparison_id, original_idx, match_mode = find_comparison_sample(
                artifact,
                comparison_artifact,
                sample_id,
                allow_class_fallback=args.allow_class_fallback,
            )
            if original_idx is None:
                raise ValueError("The Ours artifact has no selected_indices, so same-video comparison is unavailable.")

            original_video = denormalize_dataset_video(dst_train[original_idx][0])
            if vae_model == "3DVAE":
                comparison_decoded = decode_latents_3d(
                    comparison_latents[comparison_id:comparison_id + 1],
                    vae,
                    device,
                    args.batch_size,
                    expected_frames=comparison_latents.shape[1],
                )[0]
            else:
                comparison_decoded = decode_latents_2d(
                    comparison_latents[comparison_id:comparison_id + 1],
                    vae,
                    device,
                    args.batch_size,
                )[0]

            label_id = int(artifact["labels"].long()[sample_id].item())
            class_label = get_class_label(label_id, class_names)
            paper_path = save_paper_comparison(
                sample_id=sample_id,
                original_video=original_video,
                comparison_video=comparison_decoded,
                ours_video=decoded[local_id],
                importance_map=importance_map,
                class_label=class_label,
                match_mode=match_mode,
                comparison_label=args.comparison_label,
                args=args,
            )
            print(
                f"Saved paper comparison for sample={sample_id}, original_idx={original_idx}, "
                f"comparison_sample={comparison_id}, match={match_mode}: {paper_path}"
            )

    with open(Path(args.output_dir) / "visualization_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "artifact": artifact_path,
                "vae_model": vae_model,
                "num_visualized": len(summaries),
                "samples": summaries,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved visualizations to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
