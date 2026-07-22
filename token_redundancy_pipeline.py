import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tensorly as tl
from dppy.finite_dpps import FiniteDPP
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from tensorly.decomposition import tucker


def _minmax_normalize_per_video(value: torch.Tensor) -> torch.Tensor:
    mins = value.reshape(value.shape[0], -1).min(dim=1).values.view(-1, 1, 1, 1)
    maxs = value.reshape(value.shape[0], -1).max(dim=1).values.view(-1, 1, 1, 1)
    return (value - mins) / (maxs - mins + 1e-8)


def _safe_ratio(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def score_token_importance(latents: torch.Tensor, args, return_stats: bool = False):
    device = latents.device
    dtype = latents.dtype

    temporal_delta = torch.zeros(
        latents.shape[0],
        latents.shape[1],
        latents.shape[3],
        latents.shape[4],
        device=device,
        dtype=dtype,
    )
    temporal_delta[:, 1:] = (latents[:, 1:] - latents[:, :-1]).abs().mean(dim=2)

    frame_mean = latents.mean(dim=(3, 4), keepdim=True)
    spatial_contrast = (latents - frame_mean).abs().mean(dim=2)

    latents_2d = latents.reshape(-1, latents.shape[2], latents.shape[3], latents.shape[4])
    local_context = F.avg_pool2d(latents_2d, kernel_size=3, stride=1, padding=1)
    local_residual = (latents_2d - local_context).abs().mean(dim=1)
    local_residual = local_residual.view(latents.shape[0], latents.shape[1], latents.shape[3], latents.shape[4])

    channel_energy = torch.sqrt((latents ** 2).mean(dim=2) + 1e-8)

    norm_temporal = _minmax_normalize_per_video(temporal_delta)
    norm_spatial = _minmax_normalize_per_video(spatial_contrast)
    norm_local = _minmax_normalize_per_video(local_residual)
    norm_energy = _minmax_normalize_per_video(channel_energy)

    score = (
        args.importance_temporal_weight * norm_temporal
        + args.importance_spatial_weight * norm_spatial
        + args.importance_local_weight * norm_local
        + args.importance_energy_weight * norm_energy
    )
    score = score.detach().cpu()

    if not return_stats:
        return score

    stats = {
        "temporal_delta_mean": float(norm_temporal.mean().item()),
        "spatial_contrast_mean": float(norm_spatial.mean().item()),
        "local_residual_mean": float(norm_local.mean().item()),
        "channel_energy_mean": float(norm_energy.mean().item()),
    }
    return score, stats


def build_video_summaries(latents: torch.Tensor, importance_scores: torch.Tensor) -> torch.Tensor:
    summaries = []
    for video, score in zip(latents, importance_scores):
        token_vectors = video.permute(0, 2, 3, 1).reshape(-1, video.shape[1]).float()
        token_weights = score.reshape(-1).float()
        token_weights = token_weights / token_weights.sum().clamp_min(1e-8)

        weighted_mean = (token_vectors * token_weights.unsqueeze(1)).sum(dim=0)
        centered = token_vectors - weighted_mean.unsqueeze(0)
        weighted_std = torch.sqrt((centered.pow(2) * token_weights.unsqueeze(1)).sum(dim=0) + 1e-8)

        frame_scores = score.mean(dim=(1, 2)).float()
        frame_profile = frame_scores / frame_scores.sum().clamp_min(1e-8)
        summary = torch.cat([weighted_mean, weighted_std, frame_profile], dim=0)
        summaries.append(summary)
    return torch.stack(summaries, dim=0)


def _build_dpp_kernel(features: np.ndarray, quality: np.ndarray) -> np.ndarray:
    distances = cdist(features, features, metric="euclidean")
    non_zero = distances[distances > 0]
    sigma2 = float(np.median(non_zero) ** 2) if non_zero.size > 0 else 1.0
    sigma2 = max(sigma2, 1e-8)
    similarity = np.exp(-(distances ** 2) / (2 * sigma2))
    quality = quality.astype(np.float64).reshape(-1)
    kernel = similarity * np.outer(quality, quality)
    return kernel + 1e-6 * np.eye(kernel.shape[0], dtype=np.float64)


def _greedy_diverse_selection(features: np.ndarray, size: int) -> List[int]:
    if len(features) <= size:
        return list(range(len(features)))

    chosen = [int(np.argmax(np.linalg.norm(features, axis=1)))]
    while len(chosen) < size:
        remaining = [i for i in range(len(features)) if i not in chosen]
        min_dist = cdist(features[remaining], features[chosen], metric="euclidean").min(axis=1)
        nxt = remaining[int(np.argmax(min_dist))]
        chosen.append(nxt)
    return chosen


def select_videos_with_summary_dpp(
    summaries: torch.Tensor,
    labels: torch.Tensor,
    ipc: int,
    select_mode: str,
    random_state: int,
) -> List[int]:
    unique_labels = sorted(labels.unique().tolist())
    selected_indices: List[int] = []

    if select_mode == "full":
        return list(range(len(labels)))

    rng = np.random.default_rng(random_state)

    for class_id in unique_labels:
        class_indices = (labels == class_id).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(class_indices) == 0:
            continue

        if select_mode == "random":
            picked = rng.choice(class_indices, size=min(ipc, len(class_indices)), replace=False)
            selected_indices.extend(sorted(picked.tolist()))
            continue

        class_features = summaries[class_indices].cpu().numpy()
        if len(class_indices) <= ipc:
            selected_indices.extend(class_indices.tolist())
            continue

        standardized = StandardScaler().fit_transform(class_features)
        class_quality = np.ones(len(class_indices), dtype=np.float64)
        kernel = _build_dpp_kernel(standardized, class_quality)

        try:
            dpp = FiniteDPP(kernel_type="likelihood", L=kernel)
            dpp.sample_exact_k_dpp(size=ipc)
            chosen_local = list(dpp.list_of_samples[0])
            if len(chosen_local) != ipc:
                raise RuntimeError("DPP returned an unexpected sample size.")
        except Exception:
            chosen_local = _greedy_diverse_selection(standardized, ipc)

        selected_indices.extend(sorted(class_indices[chosen_local].tolist()))

    return selected_indices


def _allocate_token_buckets(importance_map: torch.Tensor, high_ratio: float, medium_ratio: float):
    flat_scores = importance_map.reshape(-1)
    num_tokens = flat_scores.numel()
    high_count = min(num_tokens, math.ceil(num_tokens * _safe_ratio(high_ratio)))
    medium_count = min(num_tokens - high_count, math.ceil(num_tokens * _safe_ratio(medium_ratio)))

    order = torch.argsort(flat_scores, descending=True)
    high_idx = order[:high_count].long()
    medium_idx = order[high_count:high_count + medium_count].long()
    return high_idx, medium_idx


def _tiered_rank_budget(
    selected_importance: torch.Tensor,
    video_shape: Tuple[int, int, int, int],
    compress_ratio: float,
    rank_boost: float,
    high_token_ratio: float,
) -> List[Tuple[int, int, int, int]]:
    num_videos, frames, height, width = selected_importance.shape
    channels = video_shape[1]

    temporal_base = min(frames, max(1, math.floor(frames * _safe_ratio(compress_ratio)) + 1))
    spatial_base = min(height, max(1, math.floor(height * _safe_ratio(compress_ratio)) + 1))

    frame_scores = selected_importance.mean(dim=(2, 3)).float()
    temporal_complexity = frame_scores.mean(dim=1) + frame_scores.std(dim=1, unbiased=False)
    spatial_complexity = selected_importance.float().std(dim=(1, 2, 3), unbiased=False)

    temporal_low_count = min(num_videos, max(0, math.ceil(num_videos * _safe_ratio(rank_boost))))
    spatial_low_count = min(num_videos, max(0, math.ceil(num_videos * _safe_ratio(high_token_ratio) * 0.25)))
    spatial_high_count = min(
        max(0, num_videos - spatial_low_count),
        max(0, round(num_videos * _safe_ratio(high_token_ratio) * _safe_ratio(rank_boost) * 0.2)),
    )

    temporal_order = torch.argsort(temporal_complexity, descending=False)
    spatial_order = torch.argsort(spatial_complexity, descending=False)

    temporal_ranks = torch.full((num_videos,), temporal_base + 1, dtype=torch.long)
    if temporal_low_count > 0:
        temporal_ranks[temporal_order[:temporal_low_count]] = temporal_base

    spatial_ranks = torch.full((num_videos,), spatial_base + 1, dtype=torch.long)
    if spatial_low_count > 0:
        spatial_ranks[spatial_order[:spatial_low_count]] = spatial_base
    if spatial_high_count > 0:
        spatial_ranks[spatial_order[-spatial_high_count:]] = spatial_base + 2

    temporal_ranks = temporal_ranks.clamp_max(frames)
    spatial_ranks = spatial_ranks.clamp_max(height)

    return [
        (int(temporal_ranks[idx].item()), channels, int(spatial_ranks[idx].item()), int(spatial_ranks[idx].item()))
        for idx in range(num_videos)
    ]


def _flatten_video_tokens(video: torch.Tensor) -> torch.Tensor:
    return video.permute(0, 2, 3, 1).reshape(-1, video.shape[1])


def _unflatten_video_tokens(tokens: torch.Tensor, video_shape: Tuple[int, int, int, int]) -> torch.Tensor:
    t, c, h, w = video_shape
    return tokens.view(t, h, w, c).permute(0, 3, 1, 2).contiguous()


def compress_single_video(video: torch.Tensor, importance_map: torch.Tensor, args, ranks: Tuple[int, int, int, int]) -> Dict:
    tl.set_backend("pytorch")

    high_idx, medium_idx = _allocate_token_buckets(
        importance_map,
        args.high_token_ratio,
        args.medium_token_ratio,
    )

    flat_video = _flatten_video_tokens(video.float())
    try:
        core, factors = tucker(video.float(), rank=ranks)
        base_record = {
            "type": "tucker",
            "core": core.cpu().half(),
            "factors": [factor.cpu().half() for factor in factors],
            "ranks": list(ranks),
        }
    except Exception as error:
        base_record = {
            "type": "raw_fp16",
            "tensor": video.cpu().half(),
            "error": str(error),
        }

    return {
        "shape": tuple(video.shape),
        "importance_mean": float(importance_map.mean().item()),
        "high_precision": {
            "indices": high_idx.cpu().to(torch.int32),
            "tokens": flat_video[high_idx].cpu().float(),
            "storage_format": "fp32",
        },
        "medium_precision": {
            "indices": medium_idx.cpu().to(torch.int32),
            "tokens": flat_video[medium_idx].cpu().half(),
            "storage_format": "fp16",
        },
        "base": base_record,
    }


def _uniform_rank_budget(video_shape: Tuple[int, int, int, int], compress_ratio: float) -> Tuple[int, int, int, int]:
    frames, channels, height, width = video_shape
    temporal_rank = min(frames, max(1, math.floor(frames * _safe_ratio(compress_ratio)) + 1))
    spatial_rank = min(height, max(1, math.floor(height * _safe_ratio(compress_ratio)) + 1))
    return temporal_rank, channels, spatial_rank, min(width, spatial_rank)


def compress_single_video_uniform_hosvd(video: torch.Tensor, ranks: Tuple[int, int, int, int]) -> Dict:
    tl.set_backend("pytorch")

    try:
        core, factors = tucker(video.float(), rank=ranks)
        base_record = {
            "type": "tucker",
            "core": core.cpu().half(),
            "factors": [factor.cpu().half() for factor in factors],
            "ranks": list(ranks),
        }
    except Exception as error:
        base_record = {
            "type": "raw_fp16",
            "tensor": video.cpu().half(),
            "error": str(error),
        }

    empty_indices = torch.empty(0, dtype=torch.int32)
    empty_tokens = torch.empty((0, video.shape[1]), dtype=torch.float16)
    return {
        "shape": tuple(video.shape),
        "importance_mean": 0.0,
        "high_precision": {
            "indices": empty_indices,
            "tokens": empty_tokens.float(),
            "storage_format": "none",
        },
        "medium_precision": {
            "indices": empty_indices,
            "tokens": empty_tokens,
            "storage_format": "none",
        },
        "base": base_record,
    }


def compress_distilled_dataset(
    selected_latents: torch.Tensor,
    selected_labels: torch.Tensor,
    selected_importance: torch.Tensor,
    selected_summaries: torch.Tensor,
    args,
    project_name: str,
    run_name: str,
    selected_indices=None,
    class_names=None,
):
    save_dir = os.path.join(args.save_path, project_name, run_name)
    os.makedirs(save_dir, exist_ok=True)

    compressed_videos = []
    if args.method == "UniformHOSVD":
        uniform_ranks = _uniform_rank_budget(tuple(selected_latents.shape[1:]), args.compress_ratio)
        for video in selected_latents:
            compressed_videos.append(compress_single_video_uniform_hosvd(video.cpu(), uniform_ranks))
    else:
        rank_plan = _tiered_rank_budget(
            selected_importance,
            tuple(selected_latents.shape[1:]),
            args.compress_ratio,
            args.rank_boost,
            args.high_token_ratio,
        )
        for video, importance_map, ranks in zip(selected_latents, selected_importance, rank_plan):
            compressed_videos.append(compress_single_video(video.cpu(), importance_map.cpu(), args, ranks))

    dense_reference_bytes = int(selected_latents.numel() * 4)
    artifact = {
        "format": "token_redundancy_distill_v1",
        "method": args.method,
        "vae_model": args.vae_model,
        "storage_mode": args.method.lower(),
        "labels": selected_labels.cpu(),
        "selected_indices": torch.tensor(selected_indices if selected_indices is not None else [], dtype=torch.long),
        "summaries": selected_summaries.cpu().half(),
        "frame_scores": selected_importance.mean(dim=(2, 3)).cpu().half(),
        "dense_reference_bytes": dense_reference_bytes,
        "videos": compressed_videos,
    }

    artifact_path = os.path.join(save_dir, "synthetic_data.pt")
    torch.save(artifact, artifact_path)
    return artifact, artifact_path


def reconstruct_distilled_latents(artifact: Dict, args) -> torch.Tensor:
    tl.set_backend("pytorch")
    reconstructed = []
    video_records = artifact["videos"] if "videos" in artifact else artifact["compressed_videos"]
    for item in video_records:
        video_shape = tuple(item["shape"] if "shape" in item else item["video_shape"])
        base = item["base"]

        base_type = base.get("type", base.get("storage_format", ""))
        if base_type in {"tucker", "fp16_tucker"}:
            core = base["core"].float()
            factors = [factor.float() for factor in base["factors"]]
            video = tl.tucker_to_tensor((core, factors)).float()
        else:
            video = base["tensor"].float()

        flat_video = _flatten_video_tokens(video)

        medium_tokens = item["medium_precision"].get("tokens", item["medium_precision"].get("values"))
        high_tokens = item["high_precision"].get("tokens", item["high_precision"].get("values"))

        if item["medium_precision"]["indices"].numel() > 0:
            flat_video[item["medium_precision"]["indices"].long()] = medium_tokens.float()
        if item["high_precision"]["indices"].numel() > 0:
            flat_video[item["high_precision"]["indices"].long()] = high_tokens.float()

        reconstructed.append(_unflatten_video_tokens(flat_video, video_shape))

    return torch.stack(reconstructed, dim=0)


def build_artifact_report(
    original_latents: torch.Tensor,
    reconstructed_latents: torch.Tensor,
    artifact: Dict,
    artifact_path: str,
    args,
    importance_stats: Dict = None,
    runtime_stats: Dict = None,
    evaluation_stats: Dict = None,
) -> Dict:
    raw_bytes = original_latents.numel() * 4
    compressed_bytes = os.path.getsize(artifact_path)
    video_records = artifact["videos"] if "videos" in artifact else artifact["compressed_videos"]
    high_counts = [record["high_precision"]["indices"].numel() for record in video_records]
    medium_counts = [record["medium_precision"]["indices"].numel() for record in video_records]
    labels = artifact["labels"]
    frame_scores = artifact.get("frame_scores")
    total_original_tokens = int(original_latents.shape[0] * original_latents.shape[1] * original_latents.shape[3] * original_latents.shape[4])
    total_explicit_tokens = int(sum(high_counts) + sum(medium_counts))
    token_keep_ratio = total_explicit_tokens / max(total_original_tokens, 1)
    storage_keep_ratio = compressed_bytes / max(raw_bytes, 1)

    report = {
        "dense_reference_mb": raw_bytes / (1024 ** 2),
        "stored_artifact_mb": compressed_bytes / (1024 ** 2),
        "compression_gain_x": raw_bytes / max(compressed_bytes, 1),
        "storage_compression_ratio": raw_bytes / max(compressed_bytes, 1),
        "storage_keep_ratio": storage_keep_ratio,
        "num_selected_videos": float(original_latents.shape[0]),
        "avg_high_precision_tokens": float(np.mean(high_counts)) if high_counts else 0.0,
        "avg_medium_precision_tokens": float(np.mean(medium_counts)) if medium_counts else 0.0,
        "total_original_tokens": float(total_original_tokens),
        "total_explicit_tokens": float(total_explicit_tokens),
        "token_keep_ratio": float(token_keep_ratio),
        "token_reduction_ratio": float(total_original_tokens / max(total_explicit_tokens, 1)),
        "num_classes": float(labels.unique().numel()),
        "selected_videos": float(original_latents.shape[0]),
        "avg_frame_importance": float(frame_scores.float().mean().item()) if frame_scores is not None else 0.0,
    }

    if importance_stats is not None:
        report.update(importance_stats)
    if runtime_stats is not None:
        report.update(runtime_stats)
    if evaluation_stats is not None:
        report.update(evaluation_stats)

    if "downstream_accuracy_mean" in report:
        report["pareto_x_token_keep_ratio"] = float(report["token_keep_ratio"])
        report["pareto_x_storage_keep_ratio"] = float(report["storage_keep_ratio"])
        report["pareto_y_accuracy"] = float(report["downstream_accuracy_mean"])

    return report
