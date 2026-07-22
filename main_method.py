import argparse
import datetime
import json
import os
import random
import time
import warnings
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import tensorly as tl
from einops import rearrange
from dppy.finite_dpps import FiniteDPP
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorly.decomposition import tucker
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from reparam_module import ReparamModule
from token_redundancy_pipeline import (
    build_artifact_report,
    build_video_summaries,
    compress_distilled_dataset,
    reconstruct_distilled_latents,
    score_token_importance,
    select_videos_with_summary_dpp,
)
from utils import (
    Conv3DNet,
    MultiStaticSharedDataset,
    TensorDataset,
    evaluate_synset,
    get_dataset,
    get_eval_pool,
    get_loops,
    get_network,
    match_loss,
    preload_test_data,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)


PIXEL_SPACE_BASELINE_METHODS = {
    "Random",
    "Herding",
    "Full",
    "DM",
    "MTT",
    "DM+VDSD",
    "MTT+VDSD",
}
LATENT_LVDD_METHODS = {"LVDD_PCA", "LVDD_Tucker"}
TOKEN_VIDEO_MODELS = {"VideoMAE", "TimeSformer"}
RECURRENT_EVAL_MODELS = {
    "VideoConvNetLSTM",
    "VideoConvNetRNN",
    "VideoConvNetGRU",
    "CNNGRU",
    "CNN_GRU",
    "CNNLSTM",
    "CNN_LSTM",
}
METHOD_ALIASES = {
    "PixelRandom": "Random",
    "PixelHerding": "Herding",
    "PixelFull": "Full",
}


def use_eval_data_parallel(args, model_name):
    return bool(args.eval_data_parallel and model_name not in RECURRENT_EVAL_MODELS)


class VAEEncodeWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(batch).latent_dist.sample()


class VAEDecodeWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(batch).sample


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_runtime(args) -> None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.num_visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.multi_gpu = args.num_visible_gpus > 1 and not args.disable_data_parallel
    args.eval_data_parallel = args.multi_gpu and not args.disable_eval_data_parallel
    args.decode_data_parallel = args.multi_gpu and args.enable_decode_data_parallel

    if args.num_visible_gpus > 0:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(args.num_visible_gpus)]
        print(f"Using {args.num_visible_gpus} visible GPU(s): {gpu_names}")
        if args.multi_gpu:
            print("DataParallel is enabled for supported stages.")
        else:
            print("Running in single-GPU mode.")
    else:
        print("Using CPU: CUDA is not available")

    print("CUDNN STATUS:", torch.backends.cudnn.enabled)


def preload_training_data(dst_train, batch_size: int, num_workers: int):
    print("Preloading training dataset into memory...")
    loader = DataLoader(
        dst_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_videos = []
    all_labels = []
    for videos, labels in tqdm(loader):
        all_videos.append(videos)
        all_labels.append(labels)

    return torch.cat(all_videos, dim=0), torch.cat(all_labels, dim=0).long()


def materialize_training_data(dst_train):
    print("Materializing training dataset sample by sample...")
    all_videos = []
    all_labels = []
    for idx in trange(len(dst_train)):
        video, label = dst_train[idx]
        all_videos.append(video.unsqueeze(0))
        all_labels.append(label)
    return torch.cat(all_videos, dim=0), torch.tensor(all_labels).long()


def get_training_labels(dst_train):
    """Read labels without materializing every video frame into memory."""
    if hasattr(dst_train, "targets"):
        return torch.as_tensor(dst_train.targets, dtype=torch.long)
    if hasattr(dst_train, "labels"):
        return torch.as_tensor(dst_train.labels, dtype=torch.long)

    print("Dataset does not expose labels; reading labels sample by sample...")
    labels = []
    for idx in trange(len(dst_train)):
        _, label = dst_train[idx]
        labels.append(label)
    return torch.tensor(labels, dtype=torch.long)


def _tensor_storage_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def build_lvdd_report(
    original_latents: torch.Tensor,
    distilled_latents: torch.Tensor,
    original_labels: torch.Tensor,
    distilled_labels: torch.Tensor,
    args,
    runtime_stats: dict = None,
    evaluation_stats: dict = None,
) -> dict:
    original_bytes = _tensor_storage_bytes(original_latents.cpu()) + _tensor_storage_bytes(original_labels.cpu())
    distilled_bytes = _tensor_storage_bytes(distilled_latents.cpu()) + _tensor_storage_bytes(distilled_labels.cpu())

    total_original_tokens = int(
        original_latents.shape[0] * original_latents.shape[1] * original_latents.shape[3] * original_latents.shape[4]
    )
    total_distilled_tokens = int(
        distilled_latents.shape[0] * distilled_latents.shape[1] * distilled_latents.shape[3] * distilled_latents.shape[4]
    )

    report = {
        "baseline_mode": args.method,
        "selection_space": "latent",
        "lvdd_select_mode": args.lvdd_select_mode,
        "num_selected_videos": float(distilled_latents.shape[0]),
        "selected_videos": float(distilled_latents.shape[0]),
        "dense_reference_mb": original_bytes / (1024 ** 2),
        "stored_artifact_mb": distilled_bytes / (1024 ** 2),
        "compression_gain_x": original_bytes / max(distilled_bytes, 1),
        "storage_compression_ratio": original_bytes / max(distilled_bytes, 1),
        "storage_keep_ratio": distilled_bytes / max(original_bytes, 1),
        "total_original_tokens": float(total_original_tokens),
        "total_explicit_tokens": float(total_distilled_tokens),
        "token_keep_ratio": float(total_distilled_tokens / max(total_original_tokens, 1)),
        "token_reduction_ratio": float(total_original_tokens / max(total_distilled_tokens, 1)),
    }

    if runtime_stats is not None:
        report.update(runtime_stats)
    if evaluation_stats is not None:
        report.update(evaluation_stats)

    if "downstream_accuracy_mean" in report:
        report["pareto_x_token_keep_ratio"] = float(report["token_keep_ratio"])
        report["pareto_x_storage_keep_ratio"] = float(report["storage_keep_ratio"])
        report["pareto_y_accuracy"] = float(report["downstream_accuracy_mean"])

    return report


def resolve_baseline_inner_model(args) -> str:
    distill_model = getattr(args, "distill_model", None)
    if distill_model:
        return distill_model
    if args.model in TOKEN_VIDEO_MODELS:
        return "ConvNet3D"
    return args.model


def resolve_baseline_eval_iters(args, start_it: int = 0):
    if getattr(args, "skip_baseline_inloop_eval", False):
        return []
    eval_stride = max(1, int(args.eval_it))
    eval_iters = np.arange(start_it, args.Iteration + 1, eval_stride).tolist()
    if args.Iteration not in eval_iters:
        eval_iters.append(args.Iteration)

    if args.model in TOKEN_VIDEO_MODELS and len(eval_iters) > 2:
        trimmed = sorted(set([start_it, args.Iteration]))
        print(
            f"Heavy downstream model {args.model} detected. "
            f"Baseline in-loop evaluation is reduced to iterations {trimmed} "
            f"instead of every {eval_stride} steps."
        )
        return trimmed

    return sorted(set(eval_iters))


def select_pixelspace_baseline_videos(train_videos: torch.Tensor, train_labels: torch.Tensor, args):
    method = args.method
    labels_np = train_labels.cpu().numpy()
    unique_labels = sorted(np.unique(labels_np).tolist())
    rng = np.random.default_rng(args.random_state)

    if method == "Full":
        selected_indices = list(range(len(train_labels)))
    elif method == "Random":
        selected_indices = []
        for class_id in unique_labels:
            class_indices = np.where(labels_np == class_id)[0]
            if len(class_indices) == 0:
                continue
            picked = rng.choice(class_indices, size=min(args.ipc, len(class_indices)), replace=False)
            selected_indices.extend(sorted(picked.tolist()))
    elif method == "Herding":
        selected_indices = []
        flattened = train_videos.reshape(train_videos.shape[0], -1).float()
        for class_id in unique_labels:
            class_indices = torch.where(train_labels == class_id)[0]
            if class_indices.numel() == 0:
                continue
            if class_indices.numel() <= args.ipc:
                selected_indices.extend(class_indices.cpu().tolist())
                continue

            class_feats = flattened[class_indices]
            class_mean = class_feats.mean(dim=0)
            chosen_local = []
            available = list(range(class_indices.numel()))
            running_sum = torch.zeros_like(class_mean)

            steps = min(args.ipc, class_indices.numel())
            for _ in range(steps):
                best_pos = None
                best_score = None
                for pos in available:
                    candidate_mean = (running_sum + class_feats[pos]) / (len(chosen_local) + 1)
                    score = torch.norm(class_mean - candidate_mean, p=2).item()
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = pos
                chosen_local.append(best_pos)
                running_sum += class_feats[best_pos]
                available.remove(best_pos)

            selected_indices.extend(class_indices[chosen_local].cpu().tolist())
    else:
        raise NotImplementedError(f"Unsupported pixel-space baseline method: {method}")

    selected_indices = [int(idx) for idx in selected_indices]
    image_syn = train_videos[selected_indices].float()
    label_syn = train_labels[selected_indices].long()
    return selected_indices, image_syn, label_syn


def build_class_index(labels: torch.Tensor, num_classes: int):
    indices_class = [[] for _ in range(num_classes)]
    for i, lab in enumerate(labels.cpu().tolist()):
        indices_class[int(lab)].append(i)
    return indices_class


def get_real_video_sampler(train_videos: torch.Tensor, train_labels: torch.Tensor, num_classes: int, args):
    indices_class = build_class_index(train_labels, num_classes)

    def get_images(c, n):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        if n == 1:
            imgs = train_videos[idx_shuffle[0]].unsqueeze(0)
        else:
            imgs = torch.stack([train_videos[i] for i in idx_shuffle], dim=0)
        return imgs.to(args.device)

    return get_images


def evaluate_pixel_synthetic_tensor(
    image_syn,
    label_syn,
    args,
    channel,
    num_classes,
    im_size,
    model_eval_pool,
    dst_test,
    log_prefix="pixel_distill",
):
    return evaluate_pixelspace_baseline(
        image_syn=image_syn,
        label_syn=label_syn,
        args=args,
        channel=channel,
        num_classes=num_classes,
        im_size=im_size,
        model_eval_pool=model_eval_pool,
        dst_test=dst_test,
    )


def run_dm_baseline(
    train_videos: torch.Tensor,
    train_labels: torch.Tensor,
    args,
    channel,
    num_classes,
    im_size,
    model_eval_pool,
    dst_test,
):
    image_syn = torch.randn(
        size=(num_classes * args.ipc, args.frames, channel, im_size[0], im_size[1]),
        dtype=torch.float,
        requires_grad=True,
        device=args.device,
    )
    label_syn = torch.tensor(
        np.stack([np.ones(args.ipc) * i for i in range(num_classes)]),
        dtype=torch.long,
        requires_grad=False,
        device=args.device,
    ).view(-1)

    get_images = get_real_video_sampler(train_videos, train_labels, num_classes, args)

    if args.init == "real":
        print("Initialize pixel synthetic videos from random real videos")
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print("Initialize pixel synthetic videos from random noise")

    if getattr(args, "save_initial_baseline_artifact", True):
        init_path = save_pixelspace_artifact(
            image_syn.detach(),
            label_syn.detach(),
            args,
            args._artifact_project_name,
            args._artifact_run_name,
            selected_indices=list(range(len(label_syn))),
            class_names=args._class_names,
            artifact_filename="synthetic_data_init.pt",
        )
        print(f"Initial DM pixel-space artifact saved to: {init_path}")

    if getattr(args, "baseline_init_only", False):
        return image_syn.detach().clone(), label_syn.detach().clone()

    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    eval_it_pool = resolve_baseline_eval_iters(args, start_it=0)
    best_ckpt = image_syn.detach().clone()
    best_score = -1.0
    inner_model = resolve_baseline_inner_model(args)
    print(f"DM inner distillation backbone: {inner_model}")

    for it in trange(0, args.Iteration + 1, ncols=60):
        if it in eval_it_pool:
            eval_summary = evaluate_pixel_synthetic_tensor(
                image_syn.detach().clone(),
                label_syn.detach().clone(),
                args,
                channel,
                num_classes,
                im_size,
                model_eval_pool,
                dst_test,
            )
            score = float(eval_summary.get("downstream_accuracy_mean", 0.0))
            if score > best_score:
                best_score = score
                best_ckpt = image_syn.detach().clone()

        net = get_network(inner_model, channel, num_classes, im_size, frames=args.frames).to(args.device)
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

        embed = net.module.embed if isinstance(net, nn.DataParallel) else net.embed

        loss = torch.tensor(0.0, device=args.device)
        for c in range(num_classes):
            img_real = get_images(c, args.batch_real)
            img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                (args.ipc, args.frames, channel, im_size[0], im_size[1])
            )
            output_real = embed(img_real).detach()
            output_syn = embed(img_syn)
            loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        wandb.log({"DM/Loss": float(loss.item() / max(num_classes, 1))})

    if getattr(args, "skip_baseline_inloop_eval", False):
        return image_syn.detach().clone(), label_syn.detach().clone()
    return best_ckpt.detach().clone(), label_syn.detach().clone()


def run_mtt_baseline(
    train_videos: torch.Tensor,
    train_labels: torch.Tensor,
    args,
    channel,
    num_classes,
    im_size,
    model_eval_pool,
    dst_test,
):
    if not args.buffer_path:
        raise ValueError("MTT requires --buffer_path pointing to expert trajectory buffers.")

    image_syn = torch.randn(
        size=(num_classes * args.ipc, args.frames, channel, im_size[0], im_size[1]),
        dtype=torch.float,
        requires_grad=True,
        device=args.device,
    )
    label_syn = torch.tensor(
        np.stack([np.ones(args.ipc) * i for i in range(num_classes)]),
        dtype=torch.long,
        requires_grad=False,
        device=args.device,
    ).view(-1)
    syn_lr = torch.tensor(args.lr_teacher, device=args.device)
    get_images = get_real_video_sampler(train_videos, train_labels, num_classes, args)

    if args.init == "real":
        print("Initialize pixel synthetic videos from random real videos")
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print("Initialize pixel synthetic videos from random noise")

    image_syn = image_syn.detach().requires_grad_(True)
    syn_lr = syn_lr.detach().requires_grad_(args.train_lr)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5) if args.train_lr else None
    criterion = nn.CrossEntropyLoss().to(args.device)

    expert_files = []
    n = 0
    while os.path.exists(os.path.join(args.buffer_path, f"replay_buffer_{n}.pt")):
        expert_files.append(os.path.join(args.buffer_path, f"replay_buffer_{n}.pt"))
        n += 1
    if n == 0:
        raise AssertionError(f"No buffers detected at {args.buffer_path}")

    file_idx = 0
    expert_idx = 0
    random.shuffle(expert_files)
    buffer = torch.load(expert_files[file_idx], map_location="cpu")
    random.shuffle(buffer)

    eval_it_pool = resolve_baseline_eval_iters(args, start_it=0)
    best_ckpt = image_syn.detach().clone()
    best_score = -1.0
    inner_model = resolve_baseline_inner_model(args)
    print(f"MTT inner distillation backbone: {inner_model}")

    for it in trange(0, args.Iteration + 1, ncols=60):
        if it in eval_it_pool:
            args.lr_net = syn_lr.detach()
            eval_summary = evaluate_pixel_synthetic_tensor(
                image_syn.detach().clone(),
                label_syn.detach().clone(),
                args,
                channel,
                num_classes,
                im_size,
                model_eval_pool,
                dst_test,
            )
            score = float(eval_summary.get("downstream_accuracy_mean", 0.0))
            if score > best_score:
                best_score = score
                best_ckpt = image_syn.detach().clone()

        student_net = get_network(inner_model, channel, num_classes, im_size, frames=args.frames, dist=False).to(args.device)
        student_net = ReparamModule(student_net)
        student_net.train()
        num_params = sum(np.prod(p.size()) for p in student_net.parameters())

        expert_trajectory = buffer[expert_idx]
        expert_idx += 1
        if expert_idx == len(buffer):
            expert_idx = 0
            file_idx += 1
            if file_idx == len(expert_files):
                file_idx = 0
                random.shuffle(expert_files)
            buffer = torch.load(expert_files[file_idx], map_location="cpu")
            random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch + args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], dim=0)
        student_params = [
            torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], dim=0).requires_grad_(True)
        ]
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], dim=0)

        syn_images = image_syn
        y_hat = label_syn.to(args.device)
        indices_chunks = []

        for _ in range(args.syn_steps):
            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))
            these_indices = indices_chunks.pop()
            x = syn_images[these_indices]
            this_y = y_hat[these_indices]
            logits = student_net(x, flat_param=student_params[-1])
            ce_loss = criterion(logits, this_y)
            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = F.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist = F.mse_loss(starting_params, target_params, reduction="sum")
        grand_loss = (param_loss / num_params) / ((param_dist / num_params) + 1e-12)

        optimizer_img.zero_grad()
        if args.train_lr:
            optimizer_lr.zero_grad()
        grand_loss.backward()
        optimizer_img.step()
        if args.train_lr:
            optimizer_lr.step()
            syn_lr.data = syn_lr.data.clamp(min=0.001)

        wandb.log({"MTT/Grand_Loss": grand_loss.detach().cpu()})

    return best_ckpt.detach().clone(), label_syn.detach().clone()


def run_lvdd_baseline(
    video_latents: torch.Tensor,
    labels_all: torch.Tensor,
    args,
):
    image_syn = torch.randn(
        size=(labels_all.unique().numel() * args.ipc, video_latents.shape[-4], video_latents.shape[-3], video_latents.shape[-2], video_latents.shape[-1]),
        dtype=torch.float,
        requires_grad=False,
        device=args.device,
    )
    label_syn = torch.tensor(
        np.stack([np.ones(args.ipc) * i for i in range(int(labels_all.unique().numel()))]),
        dtype=torch.long,
        requires_grad=False,
        device=args.device,
    ).view(-1)

    labels_cpu = labels_all.cpu()
    latents_cpu = video_latents.cpu()

    def get_latent_images(c, n):
        class_indices = torch.where(labels_cpu == c)[0].cpu().numpy()
        idx_shuffle = np.random.permutation(class_indices)[:n]
        if n == 1:
            return latents_cpu[idx_shuffle[0]].unsqueeze(0).to(args.device)
        return torch.stack([latents_cpu[i] for i in idx_shuffle], dim=0).to(args.device)

    if args.lvdd_select_mode == "full":
        image_syn = video_latents.detach().clone().to(args.device)
        label_syn = labels_all.detach().clone().to(args.device)
    elif args.lvdd_select_mode == "kmeans":
        num_classes = int(labels_all.unique().numel())
        num_samples = args.ipc
        latent_features_np = latents_cpu.reshape(latents_cpu.shape[0], -1).numpy()
        selected_indices = []
        for class_id in trange(num_classes):
            class_mask = labels_cpu == class_id
            class_latents = latent_features_np[class_mask.numpy()]
            class_indices = np.where(class_mask.numpy())[0]
            if len(class_latents) <= num_samples:
                sampled_indices = class_indices
            else:
                num_subclusters = args.lvdd_num_clusters
                kmeans = KMeans(n_clusters=num_subclusters, random_state=args.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(class_latents)
                cluster_indices_list = [np.where(cluster_labels == cluster)[0] for cluster in range(num_subclusters)]
                base_samples_per_cluster = num_samples // num_subclusters
                remainder = num_samples % num_subclusters
                cluster_sample_counts = [base_samples_per_cluster] * num_subclusters
                for i in range(remainder):
                    cluster_sample_counts[i] += 1
                sampled_indices = []
                deficit = 0
                for cluster_id, cluster_indices in enumerate(cluster_indices_list):
                    num_to_sample = cluster_sample_counts[cluster_id]
                    if len(cluster_indices) >= num_to_sample:
                        cluster_sample = np.random.choice(cluster_indices, num_to_sample, replace=False)
                    else:
                        cluster_sample = cluster_indices
                        deficit += num_to_sample - len(cluster_indices)
                    sampled_indices.extend(cluster_sample)
                if deficit > 0:
                    extra_needed = deficit
                    for cluster_indices in cluster_indices_list:
                        remaining = list(set(cluster_indices) - set(sampled_indices))
                        if len(remaining) > 0:
                            extra_sample = np.random.choice(remaining, min(len(remaining), extra_needed), replace=False)
                            sampled_indices.extend(extra_sample)
                            extra_needed -= len(extra_sample)
                        if extra_needed == 0:
                            break
                sampled_indices = class_indices[sampled_indices]
            selected_indices.extend(sampled_indices)
        image_syn = video_latents[selected_indices].to(args.device)
        label_syn = labels_all[selected_indices].to(args.device)
    elif args.lvdd_select_mode == "DAPS":
        num_samples = args.ipc
        video_all_flat = latents_cpu.reshape(latents_cpu.shape[0], -1).numpy()
        scaler = StandardScaler()
        video_all_flat = scaler.fit_transform(video_all_flat)
        num_classes = int(labels_all.unique().numel())
        selected_indices = []
        for class_id in tqdm(range(num_classes), desc="Processing Classes"):
            class_mask = labels_cpu == class_id
            class_videos = video_all_flat[class_mask.numpy()]
            class_indices = np.where(class_mask.numpy())[0]
            if len(class_videos) < num_samples:
                selected_indices.extend(class_indices)
                continue
            pairwise_distances = cdist(class_videos, class_videos, metric="euclidean")
            similarity_matrix = np.exp(-pairwise_distances)
            dpp = FiniteDPP(kernel_type="likelihood", L=similarity_matrix)
            dpp.sample_exact_k_dpp(size=num_samples)
            selected = list(dpp.list_of_samples[0])
            selected_indices.extend(class_indices[selected])
        image_syn = video_latents[selected_indices].to(args.device)
        label_syn = labels_all[selected_indices].to(args.device)
    else:
        if args.init == "real":
            print("Initialize LVDD latent set from random real latent videos")
            num_classes = int(labels_all.unique().numel())
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_latent_images(c, args.ipc).detach().data
        else:
            print("Initialize LVDD latent set from random noise")

    if args.method == "LVDD_PCA":
        n, t, c, h, w = image_syn.shape
        d = t * c * h * w
        r = int((n * d) / (2 * (n + d)))
        x = image_syn.view(n, -1)
        x_mean = x.mean(dim=0, keepdim=True)
        x_centered = x - x_mean
        u, s, v = torch.linalg.svd(x_centered, full_matrices=False)
        pca_basis = v[:r, :]
        pca_coeffs = u[:, :r] * s[:r]
        x_recon = pca_coeffs @ pca_basis + x_mean
        image_syn = x_recon.view(n, t, c, h, w).to(args.device)
    elif args.method == "LVDD_Tucker":
        tl.set_backend("pytorch")
        n, t, c, h, w = image_syn.shape
        new_t = max(1, int(t * args.compress_ratio))
        new_c = c
        new_h = max(1, int(h * args.compress_ratio))
        new_w = max(1, int(w * args.compress_ratio))
        batch_size = args.lvdd_batch_size
        reconstructed_batches = []
        for start in trange(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = image_syn[start:end]
            ranks = [end - start, new_t, new_c, new_h, new_w]
            core, factors = tucker(batch, rank=ranks)
            batch_reconstructed = tl.tucker_to_tensor((core, factors))
            reconstructed_batches.append(batch_reconstructed)
        image_syn = torch.cat(reconstructed_batches, dim=0).to(args.device)

    return image_syn.float(), label_syn.long()


def evaluate_vdsd_baseline(
    static_syn,
    dynamic_syn,
    hals,
    args,
    channel,
    num_classes,
    im_size,
    model_eval_pool,
    dst_test,
):
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    test_images, test_labels = preload_test_data(dst_test)
    test_dataset = TensorDataset(test_images, test_labels)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    eval_summary = {
        "decode_seconds": 0.0,
        "decode_peak_gpu_memory_mb": 0.0,
    }
    overall_accuracy = []
    overall_training_seconds = []
    overall_peak_memories = []

    for model_eval in model_eval_pool:
        args._active_eval_model = model_eval
        model_eval_data_parallel = use_eval_data_parallel(args, model_eval)
        print(f"Evaluation model: {model_eval}")
        accs_test = []
        train_seconds = []
        peak_memories = []

        for it_eval in range(args.num_eval):
            net_eval = get_network(
                model_eval,
                channel,
                num_classes,
                im_size,
                frames=args.frames,
                dist=model_eval_data_parallel,
                seed=args.random_state + it_eval,
                model_kwargs={
                    "videomae_model_id": args.videomae_model_id,
                    "timesformer_model_id": args.timesformer_model_id,
                    "video_transformer_image_size": args.video_transformer_image_size,
                    "video_transformer_tune_mode": args.video_transformer_tune_mode,
                },
            )
            if not model_eval_data_parallel:
                net_eval = net_eval.to(args.device)

            static_syn_eval = static_syn.detach().clone()
            dynamic_syn_eval = dynamic_syn.detach().clone()
            hal_eval = nn.ModuleList([copy.deepcopy(h) for h in hals])

            if hasattr(args, "syn_lr") and args.syn_lr is not None:
                args.lr_net = args.syn_lr.detach()

            _, _, acc_test, acc_per_cls, eval_details = evaluate_synset(
                it_eval,
                net_eval,
                [static_syn_eval, dynamic_syn_eval, hal_eval],
                None,
                testloader,
                args,
                mode="multi-static",
                test_freq=args.eval_test_freq,
            )
            accs_test.append(acc_test)
            train_seconds.append(eval_details["train_seconds"])
            peak_memories.append(eval_details["peak_gpu_memory_mb"])
            print("acc_per_cls:", acc_per_cls)
            print("acc_test:", acc_test)

        accs_test = np.array(accs_test)
        acc_test_mean = float(np.mean(accs_test))
        acc_test_std = float(np.std(accs_test))
        train_seconds_mean = float(np.mean(train_seconds)) if train_seconds else 0.0
        train_seconds_total = float(np.sum(train_seconds)) if train_seconds else 0.0
        peak_gpu_memory_mb = float(np.max(peak_memories)) if peak_memories else 0.0
        if acc_test_mean > best_acc[model_eval]:
            best_acc[model_eval] = acc_test_mean
            best_std[model_eval] = acc_test_std

        wandb.log({f"Accuracy/{model_eval}": acc_test_mean})
        wandb.log({f"Max_Accuracy/{model_eval}": best_acc[model_eval]})
        wandb.log({f"Std/{model_eval}": acc_test_std})
        wandb.log({f"Max_Std/{model_eval}": best_std[model_eval]})
        wandb.log({f"Runtime/{model_eval}_train_seconds_mean": train_seconds_mean})
        wandb.log({f"Runtime/{model_eval}_train_seconds_total": train_seconds_total})
        wandb.log({f"Memory/{model_eval}_peak_gpu_memory_mb": peak_gpu_memory_mb})

        overall_accuracy.append(acc_test_mean)
        overall_training_seconds.extend(train_seconds)
        overall_peak_memories.extend(peak_memories)
        eval_summary.update(
            {
                f"downstream_accuracy_mean_{model_eval}": acc_test_mean,
                f"downstream_accuracy_std_{model_eval}": acc_test_std,
                f"downstream_training_seconds_mean_{model_eval}": train_seconds_mean,
                f"downstream_training_seconds_total_{model_eval}": train_seconds_total,
                f"downstream_peak_gpu_memory_mb_{model_eval}": peak_gpu_memory_mb,
            }
        )

    eval_summary.update(
        {
            "downstream_accuracy_mean": float(np.mean(overall_accuracy)) if overall_accuracy else 0.0,
            "downstream_accuracy_std": float(np.std(overall_accuracy)) if overall_accuracy else 0.0,
            "downstream_training_seconds_mean": float(np.mean(overall_training_seconds)) if overall_training_seconds else 0.0,
            "downstream_training_seconds_total": float(np.sum(overall_training_seconds)) if overall_training_seconds else 0.0,
            "downstream_peak_gpu_memory_mb": float(np.max(overall_peak_memories)) if overall_peak_memories else 0.0,
        }
    )
    return eval_summary


def run_vdsd_baseline(
    train_videos: torch.Tensor,
    train_labels: torch.Tensor,
    args,
    channel,
    num_classes,
    im_size,
    model_eval_pool,
    dst_test,
    method_name: str,
):
    static_syn = torch.randn(size=(num_classes * args.spc, 3, im_size[0], im_size[1]), dtype=torch.float)
    dynamic_syn = torch.randn(size=(num_classes, args.dpc, args.frames, 1, im_size[0], im_size[1]), dtype=torch.float)
    hals = nn.ModuleList([Conv3DNet() for _ in range(args.n_hal)])
    syn_lr = torch.tensor(args.lr_teacher)

    if args.path_static is not None:
        static_syn = torch.load(args.path_static, map_location="cpu")["image"]

    static_syn = static_syn.detach().to(args.device).requires_grad_(False) if args.no_train_static else static_syn.detach().to(args.device).requires_grad_(True)
    dynamic_syn = dynamic_syn.detach().to(args.device).requires_grad_(True)
    hals = hals.to(args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(args.train_lr)
    args.syn_lr = syn_lr

    optimizer_static = None if args.no_train_static else torch.optim.SGD([static_syn], lr=args.lr_static, momentum=0.95)
    optimizer_dynamic = torch.optim.SGD([dynamic_syn], lr=args.lr_dynamic, momentum=0.95)
    optimizer_hals = torch.optim.SGD(hals.parameters(), lr=args.lr_hal, momentum=0.95)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.9) if args.train_lr else None
    criterion = nn.CrossEntropyLoss().to(args.device)

    labels_cpu = train_labels.cpu()
    indices_class = build_class_index(train_labels, num_classes)

    def get_images(c, n):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        if n == 1:
            imgs = train_videos[idx_shuffle[0]].unsqueeze(0)
        else:
            imgs = torch.stack([train_videos[i] for i in idx_shuffle], dim=0)
        return imgs.to(args.device)

    eval_it_pool = resolve_baseline_eval_iters(args, start_it=args.startIt)
    best_static = static_syn.detach().clone()
    best_dynamic = dynamic_syn.detach().clone()
    best_hals = nn.ModuleList([copy.deepcopy(h) for h in hals])
    best_score = -1.0
    inner_model = resolve_baseline_inner_model(args)
    print(f"{method_name} inner distillation backbone: {inner_model}")

    if getattr(args, "save_initial_baseline_artifact", True):
        init_image_syn, init_label_syn = synthesize_vdsd_tensor(
            static_syn,
            dynamic_syn,
            hals,
            args,
            num_classes,
            channel,
            im_size,
        )
        init_path = save_pixelspace_artifact(
            init_image_syn,
            init_label_syn,
            args,
            args._artifact_project_name,
            args._artifact_run_name,
            selected_indices=list(range(len(init_label_syn))),
            class_names=args._class_names,
            artifact_filename="synthetic_data_init.pt",
        )
        print(f"Initial {method_name} pixel-space artifact saved to: {init_path}")

    if getattr(args, "baseline_init_only", False):
        return static_syn.detach().clone(), dynamic_syn.detach().clone(), nn.ModuleList([copy.deepcopy(h) for h in hals])

    if method_name == "MTT+VDSD":
        expert_dir = args.buffer_path
        if not expert_dir:
            raise ValueError("MTT+VDSD requires --buffer_path.")
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, f"replay_buffer_{n}.pt")):
            expert_files.append(os.path.join(expert_dir, f"replay_buffer_{n}.pt"))
            n += 1
        if n == 0:
            raise AssertionError(f"No buffers detected at {expert_dir}")
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        buffer = torch.load(expert_files[file_idx], map_location="cpu")
        random.shuffle(buffer)

    for it in trange(0, args.Iteration + 1):
        wandb.log({"Progress": it})

        if it in eval_it_pool:
            eval_summary = evaluate_vdsd_baseline(
                static_syn,
                dynamic_syn,
                hals,
                args,
                channel,
                num_classes,
                im_size,
                model_eval_pool,
                dst_test,
            )
            score = float(eval_summary.get("downstream_accuracy_mean", 0.0))
            if score > best_score:
                best_score = score
                best_static = static_syn.detach().clone()
                best_dynamic = dynamic_syn.detach().clone()
                best_hals = nn.ModuleList([copy.deepcopy(h) for h in hals])

        if method_name == "MTT+VDSD":
            student_net = get_network(inner_model, channel, num_classes, im_size, frames=args.frames, dist=False).to(args.device)
            student_net = ReparamModule(student_net)
            student_net.train()
            num_params = sum(np.prod(p.size()) for p in student_net.parameters())

            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                buffer = torch.load(expert_files[file_idx], map_location="cpu")
                random.shuffle(buffer)

            start_epoch = np.random.randint(0, args.max_start_epoch)
            starting_params = expert_trajectory[start_epoch]
            target_params = expert_trajectory[start_epoch + args.expert_epochs]
            target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
            student_params = [
                torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)
            ]
            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

            syn_static = static_syn
            syn_dynamic = dynamic_syn
            indices_chunks = []
            for _ in range(args.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(num_classes * args.vpc, device=args.device)
                    indices_chunks = list(torch.split(indices, args.batch_syn))
                these_indices = indices_chunks.pop()
                label = these_indices // args.vpc
                idx = these_indices % args.vpc
                if args.dpc >= 2 * args.vpc:
                    dynamic_idx = 2 * idx + torch.randint(2, (these_indices.shape[0],), device=args.device)
                else:
                    dynamic_idx = torch.randint(args.dpc, (these_indices.shape[0],), device=args.device)
                static_idx = args.spc * label + 2 * idx + torch.randint(2, (these_indices.shape[0],), device=args.device)
                hal = hals[0]
                static = syn_static[static_idx]
                dynamic = syn_dynamic[label, dynamic_idx]
                x = hal(static, dynamic)
                this_y = label.long()
                logits = student_net(x, flat_param=student_params[-1])
                loss = criterion(logits, this_y)
                grad = torch.autograd.grad(loss, student_params[-1], create_graph=True)[0]
                student_params.append(student_params[-1] - syn_lr * grad)

            param_loss = F.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist = F.mse_loss(starting_params, target_params, reduction="sum")
            grand_loss = (param_loss / num_params) / ((param_dist / num_params) + 1e-12)

            if not args.no_train_static:
                optimizer_static.zero_grad()
            optimizer_dynamic.zero_grad()
            optimizer_hals.zero_grad()
            if args.train_lr:
                optimizer_lr.zero_grad()
            grand_loss.backward()
            if not args.no_train_static:
                optimizer_static.step()
            optimizer_dynamic.step()
            optimizer_hals.step()
            if args.train_lr:
                optimizer_lr.step()
                syn_lr.data = syn_lr.data.clamp(min=0.001)
            wandb.log({"MTT+VDSD/Grand_Loss": grand_loss.detach().cpu()})
        else:
            net = get_network(inner_model, channel, num_classes, im_size, frames=args.frames).to(args.device)
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False
            embed = net.module.embed if isinstance(net, nn.DataParallel) else net.embed

            label = torch.tensor(
                np.stack([np.ones(args.vpc) * i for i in range(num_classes)]),
                dtype=torch.long,
                requires_grad=False,
                device=args.device,
            ).view(-1)
            ran = torch.arange(0, num_classes * args.vpc, device=args.device)
            idx = ran % args.vpc
            if args.dpc >= 2 * args.vpc:
                dynamic_idx = 2 * idx + torch.randint(2, (num_classes * args.vpc,), device=args.device)
            else:
                dynamic_idx = torch.randint(args.dpc, (num_classes * args.vpc,), device=args.device)
            static_idx = args.spc * label + 2 * idx + torch.randint(2, (num_classes * args.vpc,), device=args.device)
            hal = hals[0]
            static = static_syn[static_idx]
            dynamic = dynamic_syn[label, dynamic_idx]
            image_syn = hal(static, dynamic)

            loss = torch.tensor(0.0, device=args.device)
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c * args.vpc:(c + 1) * args.vpc].reshape((args.vpc, args.frames, channel, im_size[0], im_size[1]))
                output_real = embed(img_real).detach()
                output_syn = embed(img_syn)
                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

            if not args.no_train_static:
                optimizer_static.zero_grad()
            optimizer_dynamic.zero_grad()
            optimizer_hals.zero_grad()
            if args.train_lr:
                optimizer_lr.zero_grad()
            loss.backward()
            if not args.no_train_static:
                optimizer_static.step()
            optimizer_dynamic.step()
            optimizer_hals.step()
            if args.train_lr:
                optimizer_lr.step()
                syn_lr.data = syn_lr.data.clamp(min=0.001)
            wandb.log({"DM+VDSD/Loss": float(loss.item() / max(num_classes, 1))})

    if getattr(args, "skip_baseline_inloop_eval", False):
        return static_syn.detach().clone(), dynamic_syn.detach().clone(), nn.ModuleList([copy.deepcopy(h) for h in hals])
    return best_static, best_dynamic, best_hals


def build_vae(args):
    if args.vae_model == "2DVAE":
        from quantize_vae import use_quantized_vae

        vae = use_quantized_vae().to(args.device)
        img_mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)
    else:
        from dquantize_3dvae import use_quantized_3dvae

        vae = use_quantized_3dvae().to(args.device)
        img_mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1, 1)

    vae.requires_grad_(False)
    vae.eval()
    return vae, img_mean, img_std


def build_parallel_vae_ops(vae, args):
    encoder = VAEEncodeWrapper(vae).to(args.device)
    decoder = VAEDecodeWrapper(vae).to(args.device)

    if args.multi_gpu:
        device_ids = list(range(args.num_visible_gpus))
        encoder = nn.DataParallel(encoder, device_ids=device_ids)
        if args.decode_data_parallel:
            decoder = nn.DataParallel(decoder, device_ids=device_ids)
        else:
            # Keep decode on a single GPU by default. Quantized diffusers decoding can be unstable
            # when replicated, so we gate it behind an explicit flag.
            print("Decoder stays on a single GPU. Use --enable_decode_data_parallel to override.")

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def resolve_latent_path(args):
    if args.latent_file:
        return args.latent_file

    if not args.latent_cache_dir:
        return None

    latent_cache_dir = os.path.join(
        args.latent_cache_dir,
        args.dataset,
        args.vae_model,
        f"frames_{args.frames}",
    )
    os.makedirs(latent_cache_dir, exist_ok=True)
    return os.path.join(latent_cache_dir, "latents.pt")


def resolve_latent_meta_path(latent_path: str) -> str:
    latent_dir = os.path.dirname(latent_path)
    return os.path.join(latent_dir, "latents_meta.json")


def reset_peak_gpu_memory(args) -> None:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_memory_mb(args) -> float:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0.0


def encode_videos_to_latents(
    source_videos: torch.Tensor,
    vae_encoder,
    img_mean: torch.Tensor,
    img_std: torch.Tensor,
    args,
) -> torch.Tensor:
    num_videos = source_videos.shape[0]
    latents = []

    print(f"\nEncoding {num_videos} videos into latent space with {args.vae_model}...")
    for start in trange(0, num_videos, args.encode_batch_size):
        batch = source_videos[start:start + args.encode_batch_size].to(args.device)

        if args.vae_model == "2DVAE":
            batch = rearrange(batch, "b t c h w -> (b t) c h w")
        else:
            batch = rearrange(batch, "b t c h w -> b c t h w").half()

        batch = batch * img_std + img_mean
        batch = batch * 2 - 1

        with torch.no_grad():
            encoded = vae_encoder(batch).cpu()
        latents.append(encoded)

    latents = torch.cat(latents, dim=0)
    if args.vae_model == "2DVAE":
        latents = rearrange(latents, "(b t) c h w -> b t c h w", b=num_videos)
    else:
        latents = rearrange(latents, "b c t h w -> b t c h w").float()

    print("Encoded latent tensor shape:", tuple(latents.shape))
    return latents


def encode_dataset_to_latents(
    source_dataset,
    vae_encoder,
    img_mean: torch.Tensor,
    img_std: torch.Tensor,
    args,
) -> torch.Tensor:
    loader = DataLoader(
        source_dataset,
        batch_size=args.encode_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    latents = []
    num_videos = len(source_dataset)
    if num_videos == 0:
        raise RuntimeError(
            "No videos are available for latent encoding. "
            f"For {args.dataset} with FRAMES={args.frames}, the dataset loader found 0 samples. "
            "Please re-extract frames with the requested frame count, or rerun with a matching FRAMES value."
        )

    print(f"\nStreaming and encoding {num_videos} videos into latent space with {args.vae_model}...")
    for batch, _ in tqdm(loader):
        batch = batch.to(args.device)

        if args.vae_model == "2DVAE":
            batch = rearrange(batch, "b t c h w -> (b t) c h w")
        else:
            batch = rearrange(batch, "b t c h w -> b c t h w").half()

        batch = batch * img_std + img_mean
        batch = batch * 2 - 1

        with torch.no_grad():
            encoded = vae_encoder(batch).cpu()
        latents.append(encoded)

    latents = torch.cat(latents, dim=0)
    if args.vae_model == "2DVAE":
        latents = rearrange(latents, "(b t) c h w -> b t c h w", b=num_videos)
    else:
        latents = rearrange(latents, "b c t h w -> b t c h w").float()

    print("Encoded latent tensor shape:", tuple(latents.shape))
    return latents


def load_or_create_latents(
    train_source,
    vae_encoder,
    img_mean: torch.Tensor,
    img_std: torch.Tensor,
    args,
) -> torch.Tensor:
    latent_path = resolve_latent_path(args)
    start_time = time.time()

    if latent_path and os.path.exists(latent_path):
        print(f"Loading precomputed latents from {latent_path}")
        latents = torch.load(latent_path, map_location="cpu")
        stats = {
            "latent_prepare_mode": "cache_load",
            "latent_prepare_seconds": float(time.time() - start_time),
            "latent_prepare_peak_gpu_memory_mb": 0.0,
        }
        return latents, stats

    reset_peak_gpu_memory(args)
    if isinstance(train_source, torch.Tensor):
        latents = encode_videos_to_latents(train_source, vae_encoder, img_mean, img_std, args)
    else:
        latents = encode_dataset_to_latents(train_source, vae_encoder, img_mean, img_std, args)
    peak_gpu_memory_mb = get_peak_gpu_memory_mb(args)

    if latent_path:
        os.makedirs(os.path.dirname(latent_path), exist_ok=True)
        torch.save(latents, latent_path)
        meta_path = resolve_latent_meta_path(latent_path)
        metadata = {
            "dataset": args.dataset,
            "vae_model": args.vae_model,
            "frames": int(args.frames),
            "num_videos": int(latents.shape[0]),
            "latent_shape": list(latents.shape),
            "dtype": str(latents.dtype),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved latent cache to {latent_path}")
        print(f"Saved latent metadata to {meta_path}")

    stats = {
        "latent_prepare_mode": "encoded",
        "latent_prepare_seconds": float(time.time() - start_time),
        "latent_prepare_peak_gpu_memory_mb": float(peak_gpu_memory_mb),
    }
    return latents, stats


def load_selected_indices(selection_path: str, total_videos: int) -> list:
    if not os.path.exists(selection_path):
        raise FileNotFoundError(f"Selected indices file not found: {selection_path}")

    if selection_path.endswith(".json"):
        with open(selection_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = torch.load(selection_path, map_location="cpu")

    if isinstance(payload, dict):
        if "selected_indices" not in payload:
            raise KeyError(f"'selected_indices' not found in: {selection_path}")
        selected = payload["selected_indices"]
    else:
        selected = payload

    if isinstance(selected, torch.Tensor):
        selected = selected.tolist()

    selected = [int(idx) for idx in selected]
    if len(selected) == 0:
        raise ValueError(f"No selected indices found in: {selection_path}")

    invalid = [idx for idx in selected if idx < 0 or idx >= total_videos]
    if invalid:
        raise ValueError(
            f"Found {len(invalid)} invalid selected indices in {selection_path}. "
            f"Valid range is [0, {total_videos - 1}]."
        )

    return selected


def load_existing_artifact(artifact_path: str) -> dict:
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

    artifact = torch.load(artifact_path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise ValueError(f"Artifact at {artifact_path} is not a dictionary payload.")

    required_keys = ["labels", "videos"]
    missing = [key for key in required_keys if key not in artifact]
    if missing:
        raise KeyError(f"Artifact {artifact_path} is missing keys: {missing}")

    return artifact


def load_existing_report(artifact_path: str):
    report_path = os.path.join(os.path.dirname(artifact_path), "distill_report.json")
    if not os.path.exists(report_path):
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_distill_state(
    state_path: str,
    args,
    selected_indices,
    selected_labels: torch.Tensor,
    selected_importance: torch.Tensor,
    selected_summaries: torch.Tensor,
    importance_stats: dict,
):
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    payload = {
        "format": "token_redundancy_distill_state_v1",
        "dataset": args.dataset,
        "vae_model": args.vae_model,
        "method": args.method,
        "selected_indices": torch.tensor(selected_indices, dtype=torch.long),
        "selected_labels": selected_labels.cpu().long(),
        "selected_importance": selected_importance.cpu().half(),
        "selected_summaries": selected_summaries.cpu().half(),
        "importance_stats": importance_stats,
    }
    torch.save(payload, state_path)


def load_distill_state(state_path: str, total_videos: int):
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Distill state file not found: {state_path}")

    payload = torch.load(state_path, map_location="cpu")
    required_keys = [
        "selected_indices",
        "selected_labels",
        "selected_importance",
        "selected_summaries",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise KeyError(f"Distill state {state_path} is missing keys: {missing}")

    selected_indices = payload["selected_indices"]
    if isinstance(selected_indices, torch.Tensor):
        selected_indices = selected_indices.tolist()
    selected_indices = [int(idx) for idx in selected_indices]

    invalid = [idx for idx in selected_indices if idx < 0 or idx >= total_videos]
    if invalid:
        raise ValueError(
            f"Found {len(invalid)} invalid selected indices in {state_path}. "
            f"Valid range is [0, {total_videos - 1}]."
        )

    return {
        "selected_indices": selected_indices,
        "selected_labels": payload["selected_labels"].long(),
        "selected_importance": payload["selected_importance"].float(),
        "selected_summaries": payload["selected_summaries"].float(),
        "importance_stats": payload.get("importance_stats"),
    }


def save_pixelspace_artifact(
    image_syn: torch.Tensor,
    label_syn: torch.Tensor,
    args,
    project_name: str,
    run_name: str,
    selected_indices=None,
    class_names=None,
    artifact_filename: str = "synthetic_data.pt",
):
    save_dir = os.path.join(args.save_path, project_name, run_name)
    os.makedirs(save_dir, exist_ok=True)
    artifact_path = os.path.join(save_dir, artifact_filename)
    storage_dtype = getattr(args, "pixel_artifact_dtype", "float32")
    if storage_dtype == "float16":
        images_to_save = image_syn.detach().cpu().half()
    elif storage_dtype == "float32":
        images_to_save = image_syn.detach().cpu().float()
    else:
        raise ValueError(f"Unsupported pixel_artifact_dtype: {storage_dtype}")

    payload = {
        "format": "pixelspace_distill_artifact_v1",
        "dataset": args.dataset,
        "method": args.method,
        "init": getattr(args, "init", None),
        "frames": int(args.frames),
        "pixel_artifact_dtype": storage_dtype,
        "images": images_to_save,
        "labels": label_syn.detach().cpu().long(),
        "selected_indices": torch.tensor(selected_indices or [], dtype=torch.long),
        "class_names": class_names,
    }
    torch.save(payload, artifact_path)
    return artifact_path


def synthesize_vdsd_tensor(static_syn, dynamic_syn, hals, args, num_classes, channel, im_size):
    """Materialize VDSD static/dynamic memories into ordinary video tensors."""
    videos = []
    labels = []
    hal = hals[0]
    was_training = hal.training
    hal.eval()

    with torch.no_grad():
        for c in range(num_classes):
            for idx in range(args.vpc):
                if args.spc >= 2 * args.vpc:
                    static_idx = args.spc * c + 2 * idx
                else:
                    static_idx = args.spc * c + (idx % args.spc)

                if args.dpc >= 2 * args.vpc:
                    dynamic_idx = 2 * idx
                else:
                    dynamic_idx = idx % args.dpc

                static = static_syn[static_idx].unsqueeze(0)
                dynamic = dynamic_syn[c, dynamic_idx].unsqueeze(0)
                video = hal(static, dynamic)[0]
                videos.append(video.detach().cpu())
                labels.append(c)

    if was_training:
        hal.train()

    image_syn = torch.stack(videos, dim=0).view(-1, args.frames, channel, im_size[0], im_size[1])
    label_syn = torch.tensor(labels, dtype=torch.long)
    return image_syn, label_syn


def init_wandb(project_name: str, run_name: str, args):
    config = vars(args).copy()
    try:
        run = wandb.init(
            sync_tensorboard=False,
            project=project_name,
            job_type="CleanRepo",
            config=config,
            name=run_name,
        )
    except Exception as exc:
        print(f"Wandb init failed ({exc}). Falling back to disabled mode.")
        run = wandb.init(
            sync_tensorboard=False,
            project=project_name,
            job_type="CleanRepo",
            config=config,
            name=run_name,
            mode="disabled",
        )
    return run


def decode_latent_videos(latent_videos: torch.Tensor, vae_decoder, img_mean, img_std, args) -> torch.Tensor:
    if args.vae_model == "2DVAE":
        batch_count = latent_videos.shape[0]
        flat_latents = rearrange(latent_videos, "b t c h w -> (b t) c h w")
    else:
        batch_count = latent_videos.shape[0]
        flat_latents = rearrange(latent_videos, "b t c h w -> b c t h w")

    reconstructed = []
    print("Decoding distilled latents back to pixel space...")
    for start in trange(0, len(flat_latents), args.encode_batch_size):
        batch = flat_latents[start:start + args.encode_batch_size].to(args.device)
        with torch.no_grad():
            decoded_batch = vae_decoder(batch).float()
        decoded_batch = ((decoded_batch + 1) / 2).clamp(0, 1)
        decoded_batch = (decoded_batch - img_mean) / img_std
        reconstructed.append(decoded_batch.cpu())

    reconstructed = torch.cat(reconstructed, dim=0)
    if args.vae_model == "2DVAE":
        reconstructed = rearrange(reconstructed, "(b t) c h w -> b t c h w", b=batch_count)
    else:
        reconstructed = rearrange(reconstructed, "b c t h w -> b t c h w").float()
        expected_frames = int(args.frames)
        if reconstructed.shape[1] < expected_frames:
            missing_frames = expected_frames - reconstructed.shape[1]
            last_frame = reconstructed[:, -1:].contiguous()
            padding = last_frame.repeat(1, missing_frames, 1, 1, 1)
            reconstructed = torch.cat([reconstructed, padding], dim=1)

    return reconstructed


def evaluate_distilled_set(
    image_syn,
    label_syn,
    vae_decoder,
    img_mean,
    img_std,
    args,
    channel,
    num_classes,
    im_size,
    model_eval_pool,
    dst_test,
):
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    print("\nEvaluating reconstructed distilled set...")
    test_images, test_labels = preload_test_data(dst_test)
    test_dataset = TensorDataset(test_images, test_labels)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    reset_peak_gpu_memory(args)
    decode_start = time.time()
    decoded_syn = decode_latent_videos(image_syn, vae_decoder, img_mean, img_std, args)
    decode_seconds = float(time.time() - decode_start)
    decode_peak_gpu_memory_mb = get_peak_gpu_memory_mb(args)

    eval_summary = {
        "decode_seconds": decode_seconds,
        "decode_peak_gpu_memory_mb": decode_peak_gpu_memory_mb,
    }
    overall_accuracy = []
    overall_training_seconds = []
    overall_peak_memories = []

    for model_eval in model_eval_pool:
        args._active_eval_model = model_eval
        model_eval_data_parallel = use_eval_data_parallel(args, model_eval)
        print(f"Evaluation model: {model_eval}")
        if model_eval_data_parallel:
            print(f"Evaluation network mode: DataParallel over {args.num_visible_gpus} GPU(s)")
        elif args.device.startswith("cuda"):
            print("Evaluation network mode: single GPU")
        else:
            print("Evaluation network mode: CPU")
        accs_test = []
        train_seconds = []
        peak_memories = []

        for it_eval in range(args.num_eval):
            net_eval = get_network(
                model_eval,
                channel,
                num_classes,
                im_size,
                frames=args.frames,
                dist=model_eval_data_parallel,
                seed=args.random_state + it_eval,
                model_kwargs={
                    "videomae_model_id": args.videomae_model_id,
                    "timesformer_model_id": args.timesformer_model_id,
                    "video_transformer_image_size": args.video_transformer_image_size,
                    "video_transformer_tune_mode": args.video_transformer_tune_mode,
                },
            )
            if not model_eval_data_parallel:
                net_eval = net_eval.to(args.device)
            image_syn_eval = decoded_syn.detach().clone()
            label_syn_eval = label_syn.detach().clone()

            _, _, acc_test, acc_per_cls, eval_details = evaluate_synset(
                it_eval,
                net_eval,
                image_syn_eval,
                label_syn_eval,
                testloader,
                args,
                mode="none",
                test_freq=args.eval_test_freq,
            )
            accs_test.append(acc_test)
            train_seconds.append(eval_details["train_seconds"])
            peak_memories.append(eval_details["peak_gpu_memory_mb"])
            print("acc_per_cls:", acc_per_cls)
            print("acc_test:", acc_test)

        accs_test = np.array(accs_test)
        acc_test_mean = float(np.mean(accs_test))
        acc_test_std = float(np.std(accs_test))
        train_seconds_mean = float(np.mean(train_seconds)) if train_seconds else 0.0
        train_seconds_total = float(np.sum(train_seconds)) if train_seconds else 0.0
        peak_gpu_memory_mb = float(np.max(peak_memories)) if peak_memories else 0.0
        if acc_test_mean > best_acc[model_eval]:
            best_acc[model_eval] = acc_test_mean
            best_std[model_eval] = acc_test_std

        print(
            "Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------"
            % (len(accs_test), model_eval, acc_test_mean, acc_test_std)
        )

        wandb.log({f"Accuracy/{model_eval}": acc_test_mean})
        wandb.log({f"Max_Accuracy/{model_eval}": best_acc[model_eval]})
        wandb.log({f"Std/{model_eval}": acc_test_std})
        wandb.log({f"Max_Std/{model_eval}": best_std[model_eval]})
        wandb.log({f"Runtime/{model_eval}_train_seconds_mean": train_seconds_mean})
        wandb.log({f"Runtime/{model_eval}_train_seconds_total": train_seconds_total})
        wandb.log({f"Memory/{model_eval}_peak_gpu_memory_mb": peak_gpu_memory_mb})

        overall_accuracy.append(acc_test_mean)
        overall_training_seconds.extend(train_seconds)
        overall_peak_memories.extend(peak_memories)
        eval_summary.update(
            {
                f"downstream_accuracy_mean_{model_eval}": acc_test_mean,
                f"downstream_accuracy_std_{model_eval}": acc_test_std,
                f"downstream_training_seconds_mean_{model_eval}": train_seconds_mean,
                f"downstream_training_seconds_total_{model_eval}": train_seconds_total,
                f"downstream_peak_gpu_memory_mb_{model_eval}": peak_gpu_memory_mb,
            }
        )

    wandb.log({"Runtime/decode_seconds": decode_seconds})
    wandb.log({"Memory/decode_peak_gpu_memory_mb": decode_peak_gpu_memory_mb})
    eval_summary.update(
        {
            "downstream_accuracy_mean": float(np.mean(overall_accuracy)) if overall_accuracy else 0.0,
            "downstream_accuracy_std": float(np.std(overall_accuracy)) if overall_accuracy else 0.0,
            "downstream_training_seconds_mean": float(np.mean(overall_training_seconds)) if overall_training_seconds else 0.0,
            "downstream_training_seconds_total": float(np.sum(overall_training_seconds)) if overall_training_seconds else 0.0,
            "downstream_peak_gpu_memory_mb": float(np.max(overall_peak_memories)) if overall_peak_memories else 0.0,
        }
    )
    return eval_summary


def evaluate_full_dataset_baseline(
    video_latents,
    labels_all,
    vae_decoder,
    img_mean,
    img_std,
    args,
    channel,
    num_classes,
    im_size,
    model_eval_pool,
    dst_test,
):
    print("\nRunning full-data downstream baseline without distillation or compression...")
    print("Full latent tensor shape:", tuple(video_latents.shape))
    print("Full label tensor shape:", tuple(labels_all.shape))

    eval_summary = evaluate_distilled_set(
        video_latents.float(),
        labels_all.long(),
        vae_decoder,
        img_mean,
        img_std,
        args,
        channel,
        num_classes,
        im_size,
        model_eval_pool,
        dst_test,
    )

    report = {
        "baseline_mode": "full_data_no_distill_no_compress",
        "num_selected_videos": float(video_latents.shape[0]),
        "selected_videos": float(video_latents.shape[0]),
        "token_keep_ratio": 1.0,
        "token_reduction_ratio": 1.0,
        "storage_keep_ratio": 1.0,
        "storage_compression_ratio": 1.0,
        "compression_gain_x": 1.0,
    }
    report.update(eval_summary)
    return report


def evaluate_pixelspace_baseline(
    image_syn,
    label_syn,
    args,
    channel,
    num_classes,
    im_size,
    model_eval_pool,
    dst_test,
):
    print("\nEvaluating pixel-space baseline set...")
    print("Selected pixel-video tensor shape:", tuple(image_syn.shape))
    print("Selected labels shape:", tuple(label_syn.shape))

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    test_images, test_labels = preload_test_data(dst_test)
    test_dataset = TensorDataset(test_images, test_labels)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    eval_summary = {
        "decode_seconds": 0.0,
        "decode_peak_gpu_memory_mb": 0.0,
    }
    overall_accuracy = []
    overall_training_seconds = []
    overall_peak_memories = []

    for model_eval in model_eval_pool:
        args._active_eval_model = model_eval
        model_eval_data_parallel = use_eval_data_parallel(args, model_eval)
        print(f"Evaluation model: {model_eval}")
        if model_eval_data_parallel:
            print(f"Evaluation network mode: DataParallel over {args.num_visible_gpus} GPU(s)")
        elif args.device.startswith("cuda"):
            print("Evaluation network mode: single GPU")
        else:
            print("Evaluation network mode: CPU")

        accs_test = []
        train_seconds = []
        peak_memories = []

        for it_eval in range(args.num_eval):
            net_eval = get_network(
                model_eval,
                channel,
                num_classes,
                im_size,
                frames=args.frames,
                dist=model_eval_data_parallel,
                seed=args.random_state + it_eval,
                model_kwargs={
                    "videomae_model_id": args.videomae_model_id,
                    "timesformer_model_id": args.timesformer_model_id,
                    "video_transformer_image_size": args.video_transformer_image_size,
                    "video_transformer_tune_mode": args.video_transformer_tune_mode,
                },
            )
            if not model_eval_data_parallel:
                net_eval = net_eval.to(args.device)

            image_syn_eval = image_syn.detach().clone()
            label_syn_eval = label_syn.detach().clone()

            _, _, acc_test, acc_per_cls, eval_details = evaluate_synset(
                it_eval,
                net_eval,
                image_syn_eval,
                label_syn_eval,
                testloader,
                args,
                mode="none",
                test_freq=args.eval_test_freq,
            )
            accs_test.append(acc_test)
            train_seconds.append(eval_details["train_seconds"])
            peak_memories.append(eval_details["peak_gpu_memory_mb"])
            print("acc_per_cls:", acc_per_cls)
            print("acc_test:", acc_test)

        accs_test = np.array(accs_test)
        acc_test_mean = float(np.mean(accs_test))
        acc_test_std = float(np.std(accs_test))
        train_seconds_mean = float(np.mean(train_seconds)) if train_seconds else 0.0
        train_seconds_total = float(np.sum(train_seconds)) if train_seconds else 0.0
        peak_gpu_memory_mb = float(np.max(peak_memories)) if peak_memories else 0.0
        if acc_test_mean > best_acc[model_eval]:
            best_acc[model_eval] = acc_test_mean
            best_std[model_eval] = acc_test_std

        print(
            "Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------"
            % (len(accs_test), model_eval, acc_test_mean, acc_test_std)
        )

        wandb.log({f"Accuracy/{model_eval}": acc_test_mean})
        wandb.log({f"Max_Accuracy/{model_eval}": best_acc[model_eval]})
        wandb.log({f"Std/{model_eval}": acc_test_std})
        wandb.log({f"Max_Std/{model_eval}": best_std[model_eval]})
        wandb.log({f"Runtime/{model_eval}_train_seconds_mean": train_seconds_mean})
        wandb.log({f"Runtime/{model_eval}_train_seconds_total": train_seconds_total})
        wandb.log({f"Memory/{model_eval}_peak_gpu_memory_mb": peak_gpu_memory_mb})

        overall_accuracy.append(acc_test_mean)
        overall_training_seconds.extend(train_seconds)
        overall_peak_memories.extend(peak_memories)
        eval_summary.update(
            {
                f"downstream_accuracy_mean_{model_eval}": acc_test_mean,
                f"downstream_accuracy_std_{model_eval}": acc_test_std,
                f"downstream_training_seconds_mean_{model_eval}": train_seconds_mean,
                f"downstream_training_seconds_total_{model_eval}": train_seconds_total,
                f"downstream_peak_gpu_memory_mb_{model_eval}": peak_gpu_memory_mb,
            }
        )

    eval_summary.update(
        {
            "downstream_accuracy_mean": float(np.mean(overall_accuracy)) if overall_accuracy else 0.0,
            "downstream_accuracy_std": float(np.std(overall_accuracy)) if overall_accuracy else 0.0,
            "downstream_training_seconds_mean": float(np.mean(overall_training_seconds)) if overall_training_seconds else 0.0,
            "downstream_training_seconds_total": float(np.sum(overall_training_seconds)) if overall_training_seconds else 0.0,
            "downstream_peak_gpu_memory_mb": float(np.max(overall_peak_memories)) if overall_peak_memories else 0.0,
        }
    )
    return eval_summary


def main(args):
    args.method = METHOD_ALIASES.get(args.method, args.method)
    set_seed(args.random_state)
    detect_runtime(args)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = get_dataset(
        args.dataset, args.data_path, num_workers=args.num_workers, frames=args.frames
    )

    if args.eval_models:
        model_eval_pool = [item.strip() for item in args.eval_models.split(",") if item.strip()]
    else:
        model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    print("Eval mode is,", args.eval_mode)
    print("Evaluation model pool:", model_eval_pool)

    vae, img_mean, img_std = build_vae(args)
    vae_encoder, vae_decoder = build_parallel_vae_ops(vae, args)

    project_name = f"LongVideoToken_{args.dataset}_{args.method}_{args.vae_model}"
    run_name = f"{args.dataset}_ipc{args.ipc}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    wandb_run = init_wandb(project_name, run_name, args)
    args._artifact_project_name = project_name
    args._artifact_run_name = wandb_run.name
    args._class_names = class_names

    if args.artifact_file:
        print(f"\nReusing prebuilt distilled artifact from {args.artifact_file}")
        artifact = load_existing_artifact(args.artifact_file)
        artifact_report = load_existing_report(args.artifact_file)
        image_syn = reconstruct_distilled_latents(artifact, args)
        label_syn = artifact["labels"].long()

        if artifact_report is not None:
            print(f"Artifact report: {artifact_report}")
            wandb.log(artifact_report)

        evaluate_distilled_set(
            image_syn,
            label_syn,
            vae_decoder,
            img_mean,
            img_std,
            args,
            channel,
            num_classes,
            im_size,
            model_eval_pool,
            dst_test,
        )
        wandb.finish()
        return

    if args.method in PIXEL_SPACE_BASELINE_METHODS:
        if args.preload:
            train_videos, train_labels = preload_training_data(dst_train, args.preload_batch_size, args.num_workers)
            dst_train = torch.utils.data.TensorDataset(train_videos, train_labels)
        else:
            train_videos, train_labels = materialize_training_data(dst_train)
        train_source_for_latents = train_videos
    else:
        train_videos = None
        train_labels = get_training_labels(dst_train)
        if args.preload:
            train_videos, train_labels = preload_training_data(dst_train, args.preload_batch_size, args.num_workers)
            dst_train = torch.utils.data.TensorDataset(train_videos, train_labels)
            train_source_for_latents = train_videos
        else:
            print("Using dataset streaming for latent methods; training videos are not materialized in CPU memory.")
            train_source_for_latents = dst_train

    if args.batch_syn is None:
        args.batch_syn = max(1, int(args.ipc * num_classes))

    if args.method in PIXEL_SPACE_BASELINE_METHODS:
        if args.method in {"Random", "Herding", "Full"}:
            selected_indices, image_syn, label_syn = select_pixelspace_baseline_videos(train_videos, train_labels, args)
        elif args.method == "DM":
            image_syn, label_syn = run_dm_baseline(
                train_videos,
                train_labels,
                args,
                channel,
                num_classes,
                im_size,
                model_eval_pool,
                dst_test,
            )
            selected_indices = list(range(len(label_syn)))
            if args.baseline_init_only:
                print("Baseline init-only mode: saved initial DM artifact and skipped distillation/evaluation.")
                wandb.finish()
                return
        elif args.method == "MTT":
            image_syn, label_syn = run_mtt_baseline(
                train_videos,
                train_labels,
                args,
                channel,
                num_classes,
                im_size,
                model_eval_pool,
                dst_test,
            )
            selected_indices = list(range(len(label_syn)))
        elif args.method in {"DM+VDSD", "MTT+VDSD"}:
            static_syn, dynamic_syn, hals = run_vdsd_baseline(
                train_videos,
                train_labels,
                args,
                channel,
                num_classes,
                im_size,
                model_eval_pool,
                dst_test,
                method_name=args.method,
            )
            if args.baseline_init_only:
                print(f"Baseline init-only mode: saved initial {args.method} artifact and skipped distillation/evaluation.")
                wandb.finish()
                return
            eval_summary = evaluate_vdsd_baseline(
                static_syn,
                dynamic_syn,
                hals,
                args,
                channel,
                num_classes,
                im_size,
                model_eval_pool,
                dst_test,
            )
            report = {
                "baseline_mode": args.method,
                "selection_space": "pixel_vdsd",
                "num_selected_videos": float(num_classes * args.vpc),
                "selected_videos": float(num_classes * args.vpc),
                "token_keep_ratio": 1.0,
                "token_reduction_ratio": 1.0,
                "storage_keep_ratio": 1.0,
                "storage_compression_ratio": 1.0,
                "compression_gain_x": 1.0,
            }
            report.update(eval_summary)
            image_syn, label_syn = synthesize_vdsd_tensor(
                static_syn,
                dynamic_syn,
                hals,
                args,
                num_classes,
                channel,
                im_size,
            )
            artifact_path = save_pixelspace_artifact(
                image_syn,
                label_syn,
                args,
                project_name,
                wandb_run.name,
                selected_indices=list(range(len(label_syn))),
                class_names=class_names,
            )
            report.update(
                {
                    "stored_artifact_mb": os.path.getsize(artifact_path) / (1024 ** 2),
                    "dense_fp32_reference_mb": image_syn.numel() * 4 / (1024 ** 2),
                    "pixel_artifact_dtype": args.pixel_artifact_dtype,
                    "init": args.init,
                }
            )
            report_path = os.path.join(os.path.dirname(artifact_path), "distill_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"VDSD pixel-space artifact saved to: {artifact_path}")
            print(f"VDSD report saved to: {report_path}")
            print(f"VDSD baseline report: {report}")
            wandb.log(report)
            wandb.finish()
            return
        else:
            raise NotImplementedError(f"Unsupported pixel-space baseline method: {args.method}")

        eval_summary = evaluate_pixelspace_baseline(
            image_syn,
            label_syn,
            args,
            channel,
            num_classes,
            im_size,
            model_eval_pool,
            dst_test,
        )
        report = {
            "baseline_mode": args.method,
            "selection_space": "pixel",
            "num_selected_videos": float(len(selected_indices)),
            "selected_videos": float(len(selected_indices)),
            "token_keep_ratio": 1.0,
            "token_reduction_ratio": 1.0,
            "storage_keep_ratio": 1.0,
            "storage_compression_ratio": 1.0,
            "compression_gain_x": 1.0,
        }
        report.update(eval_summary)
        artifact_path = save_pixelspace_artifact(
            image_syn,
            label_syn,
            args,
            project_name,
            wandb_run.name,
            selected_indices=selected_indices,
            class_names=class_names,
        )
        report.update(
            {
                "stored_artifact_mb": os.path.getsize(artifact_path) / (1024 ** 2),
                "dense_fp32_reference_mb": image_syn.numel() * 4 / (1024 ** 2),
                "pixel_artifact_dtype": args.pixel_artifact_dtype,
                "init": args.init,
            }
        )
        report_path = os.path.join(os.path.dirname(artifact_path), "distill_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Pixel-space artifact saved to: {artifact_path}")
        print(f"Pixel-space report saved to: {report_path}")
        print(f"Pixel-space baseline report: {report}")
        wandb.log(report)
        wandb.finish()
        return

    video_latents, latent_runtime_stats = load_or_create_latents(train_source_for_latents, vae_encoder, img_mean, img_std, args)
    labels_all = train_labels.long()

    if args.method in LATENT_LVDD_METHODS:
        image_syn, label_syn = run_lvdd_baseline(video_latents, labels_all, args)
        eval_summary = evaluate_distilled_set(
            image_syn,
            label_syn,
            vae_decoder,
            img_mean,
            img_std,
            args,
            channel,
            num_classes,
            im_size,
            model_eval_pool,
            dst_test,
        )
        report = build_lvdd_report(
            original_latents=video_latents,
            distilled_latents=image_syn,
            original_labels=labels_all,
            distilled_labels=label_syn,
            args=args,
            runtime_stats=latent_runtime_stats,
            evaluation_stats=eval_summary,
        )
        print(f"LVDD-style baseline report: {report}")
        wandb.log(report)
        wandb.finish()
        return

    if args.full_data_baseline:
        report = evaluate_full_dataset_baseline(
            video_latents,
            labels_all,
            vae_decoder,
            img_mean,
            img_std,
            args,
            channel,
            num_classes,
            im_size,
            model_eval_pool,
            dst_test,
        )
        report.update(latent_runtime_stats)
        print(f"Full-data baseline report: {report}")
        wandb.log(report)
        wandb.finish()
        return

    save_dir = os.path.join(args.save_path, project_name, wandb_run.name)
    os.makedirs(save_dir, exist_ok=True)

    if args.distill_state_file:
        print(f"\nReusing distillation state from {args.distill_state_file}")
        distill_state = load_distill_state(args.distill_state_file, total_videos=len(video_latents))
        selected_indices = distill_state["selected_indices"]
        image_syn = video_latents[selected_indices].float()
        label_syn = labels_all[selected_indices].long()
        selected_importance = distill_state["selected_importance"]
        selected_summaries = distill_state["selected_summaries"]
        importance_stats = distill_state["importance_stats"] or {}

        if not torch.equal(label_syn.cpu(), distill_state["selected_labels"].cpu()):
            raise ValueError(
                "Selected labels from distill_state do not match the current dataset order. "
                "Please verify the dataset split and latent cache are the same as the original run."
            )

        print("Selected latent tensor shape:", tuple(image_syn.shape))
        print("Selected label tensor shape:", tuple(label_syn.shape))
        print(f"\n[4/4] Importance-aware compression with {args.method}...")
        artifact, artifact_path = compress_distilled_dataset(
            image_syn,
            label_syn,
            selected_importance,
            selected_summaries,
            args,
            project_name,
            wandb_run.name,
            selected_indices=selected_indices,
            class_names=class_names,
        )

        reconstructed_latents = reconstruct_distilled_latents(artifact, args)
        eval_summary = evaluate_distilled_set(
            reconstructed_latents,
            label_syn,
            vae_decoder,
            img_mean,
            img_std,
            args,
            channel,
            num_classes,
            im_size,
            model_eval_pool,
            dst_test,
        )

        report = build_artifact_report(
            image_syn,
            reconstructed_latents,
            artifact,
            artifact_path,
            args,
            importance_stats=importance_stats,
            runtime_stats=latent_runtime_stats,
            evaluation_stats=eval_summary,
        )

        report_path = os.path.join(os.path.dirname(artifact_path), "distill_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"Artifact saved to: {artifact_path}")
        print(f"Distillation report: {report}")
        print(f"Report saved to: {report_path}")
        wandb.log(report)
        wandb.finish()
        return

    print("\n[1/4] Token importance scoring...")
    importance_scores, importance_stats = score_token_importance(video_latents, args, return_stats=True)
    print("Importance score shape:", tuple(importance_scores.shape))

    print("\n[2/4] Weighted summary construction...")
    video_summaries = build_video_summaries(video_latents, importance_scores)
    print("Summary shape:", tuple(video_summaries.shape))

    print(f"\n[3/4] Dataset distillation in summary space with {args.select_mode}...")
    if args.selected_indices_file:
        print(f"Reusing selected indices from {args.selected_indices_file}")
        selected_indices = load_selected_indices(args.selected_indices_file, total_videos=len(video_latents))
    else:
        selected_indices = select_videos_with_summary_dpp(
            video_summaries,
            labels_all,
            ipc=args.ipc,
            select_mode=args.select_mode,
            random_state=args.random_state,
        )

    image_syn = video_latents[selected_indices].float()
    label_syn = labels_all[selected_indices].long()
    selected_importance = importance_scores[selected_indices]
    selected_summaries = video_summaries[selected_indices]
    print("Selected latent tensor shape:", tuple(image_syn.shape))
    print("Selected label tensor shape:", tuple(label_syn.shape))

    distill_state_path = os.path.join(save_dir, "distill_state.pt")
    save_distill_state(
        distill_state_path,
        args,
        selected_indices,
        label_syn,
        selected_importance,
        selected_summaries,
        importance_stats,
    )
    print(f"Distillation state saved to: {distill_state_path}")

    print(f"\n[4/4] Importance-aware compression with {args.method}...")
    artifact, artifact_path = compress_distilled_dataset(
        image_syn,
        label_syn,
        selected_importance,
        selected_summaries,
        args,
        project_name,
        wandb_run.name,
        selected_indices=selected_indices,
        class_names=class_names,
    )

    reconstructed_latents = reconstruct_distilled_latents(artifact, args)
    if args.skip_eval_after_distill:
        print("Skipping downstream evaluation after distillation; artifact/report will be saved for visualization.")
        eval_summary = None
    else:
        eval_summary = evaluate_distilled_set(
            reconstructed_latents,
            label_syn,
            vae_decoder,
            img_mean,
            img_std,
            args,
            channel,
            num_classes,
            im_size,
            model_eval_pool,
            dst_test,
        )

    report = build_artifact_report(
        image_syn,
        reconstructed_latents,
        artifact,
        artifact_path,
        args,
        importance_stats=importance_stats,
        runtime_stats=latent_runtime_stats,
        evaluation_stats=eval_summary,
    )

    report_path = os.path.join(os.path.dirname(artifact_path), "distill_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Artifact saved to: {artifact_path}")
    print(f"Distillation report: {report}")
    print(f"Report saved to: {report_path}")
    wandb.log(report)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long video token redundancy distillation")
    parser.add_argument("--dataset", type=str, default="miniUCF101", help="dataset")
    parser.add_argument(
        "--method",
        type=str,
        default="ImportanceHOSVD",
        help="ImportanceHOSVD / UniformHOSVD / Random / Herding / Full / DM / MTT / DM+VDSD / MTT+VDSD / LVDD_PCA / LVDD_Tucker",
    )
    parser.add_argument("--model", type=str, default="ConvNet3D", help="model")
    parser.add_argument(
        "--eval_models",
        type=str,
        default="",
        help="comma-separated downstream architectures for cross-architecture evaluation, e.g. VideoMAE,TimeSformer,ConvNet3D,CNNGRU,CNNLSTM",
    )
    parser.add_argument(
        "--distill_model",
        type=str,
        default=None,
        help="optional inner backbone for DM/MTT/VDSD optimization; defaults to ConvNet3D when downstream model is VideoMAE/TimeSformer",
    )
    parser.add_argument("--ipc", type=int, default=1, help="image(s) per class")
    parser.add_argument("--eval_mode", type=str, default="SS", help="evaluation mode")
    parser.add_argument("--num_eval", type=int, default=1, help="how many networks to evaluate on")
    parser.add_argument("--epoch_eval_train", type=int, default=500, help="epochs to train a model with synthetic data")
    parser.add_argument(
        "--eval_test_freq",
        type=int,
        default=100,
        help="evaluate test accuracy every N epochs during downstream training; final epoch is always evaluated",
    )
    parser.add_argument("--lr_net", type=float, default=0.01, help="learning rate for network")
    parser.add_argument("--lr_img", type=float, default=1.0, help="learning rate for synthetic videos in pixel-space DM/MTT")
    parser.add_argument("--lr_lr", type=float, default=1e-5, help="learning rate for synthetic learning rate in MTT")
    parser.add_argument("--lr_teacher", type=float, default=0.001, help="synthetic learning rate used by MTT")
    parser.add_argument("--lr_static", type=float, default=100.0, help="learning rate for VDSD static memory")
    parser.add_argument("--lr_dynamic", type=float, default=0.01, help="learning rate for VDSD dynamic memory")
    parser.add_argument("--lr_hal", type=float, default=0.01, help="learning rate for VDSD hallucinator")
    parser.add_argument("--batch_train", type=int, default=256, help="batch size for training networks")
    parser.add_argument("--batch_real", type=int, default=256, help="batch size of real videos for pixel-space DM")
    parser.add_argument("--test_batch_size", type=int, default=32, help="batch size for testing")
    parser.add_argument("--batch_syn", type=int, default=None, help="batch size for synthetic data")
    parser.add_argument("--Iteration", type=int, default=1000, help="number of distillation iterations for DM/MTT")
    parser.add_argument("--eval_it", type=int, default=50, help="how often to evaluate synthetic videos during DM/MTT")
    parser.add_argument(
        "--skip_baseline_inloop_eval",
        action="store_true",
        help="skip expensive in-loop downstream evaluation for DM/MTT/VDSD and evaluate only the final distilled set",
    )
    parser.add_argument(
        "--skip_eval_after_distill",
        action="store_true",
        help="save the distilled artifact/report and skip downstream evaluation; useful for visualization-only runs",
    )
    parser.add_argument(
        "--baseline_init_only",
        action="store_true",
        help="save the pre-optimization DM/VDSD distilled artifact and exit before distillation/evaluation",
    )
    parser.add_argument(
        "--no_save_initial_baseline_artifact",
        dest="save_initial_baseline_artifact",
        action="store_false",
        help="do not save synthetic_data_init.pt before DM/MTT/VDSD optimization",
    )
    parser.set_defaults(save_initial_baseline_artifact=True)
    parser.add_argument("--data_path", type=str, default="distill_utils/data", help="dataset path")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--preload", action="store_true", help="preload dataset")
    parser.add_argument("--preload_batch_size", type=int, default=32, help="preload batch size")
    parser.add_argument("--save_path", type=str, default="./logged_files", help="path to save")
    parser.add_argument("--frames", type=int, default=16, help="frames per sample")
    parser.add_argument("--random_state", type=int, default=42, help="random seed")
    parser.add_argument("--init", type=str, default="noise", choices=["noise", "real", "real-all"], help="initialization strategy for pixel-space synthetic videos")
    parser.add_argument(
        "--pixel_artifact_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="dtype used when saving pixel-space baseline artifacts",
    )
    parser.add_argument("--spc", type=int, default=10, help="static memory per class for VDSD")
    parser.add_argument("--dpc", type=int, default=1, help="dynamic memory per class for VDSD")
    parser.add_argument("--vpc", type=int, default=5, help="videos per class synthesized by VDSD")
    parser.add_argument("--n_hal", type=int, default=1, help="number of VDSD hallucinators")
    parser.add_argument("--startIt", type=int, default=0, help="start iteration for VDSD / DM / MTT evaluation schedule")
    parser.add_argument("--no_train_static", action="store_true", help="freeze VDSD static memory")
    parser.add_argument("--path_static", type=str, default=None, help="optional path to pretrained VDSD static memory")
    parser.add_argument("--expert_epochs", type=int, default=3, help="number of expert epochs used by MTT")
    parser.add_argument("--syn_steps", type=int, default=64, help="number of synthetic optimization steps per MTT iteration")
    parser.add_argument("--max_start_epoch", type=int, default=25, help="maximum expert start epoch for MTT")
    parser.add_argument("--dis_metric", type=str, default="ours", help="distance metric used by DM/MTT utilities")
    parser.add_argument("--buffer_path", type=str, default=None, help="path to replay buffers for MTT")
    parser.add_argument("--train_lr", action="store_true", help="optimize synthetic learning rate in MTT")
    parser.add_argument(
        "--full_data_baseline",
        action="store_true",
        help="skip distillation/compression and evaluate the full training set directly on the downstream model",
    )
    parser.add_argument("--vae_model", type=str, default="2DVAE", help="VAE model used for encoding and decoding")
    parser.add_argument("--compress_ratio", type=float, default=0.75, help="compression ratio used in HOSVD")
    parser.add_argument("--rank_boost", type=float, default=0.2, help="extra rank budget for important videos")
    parser.add_argument("--lvdd_num_clusters", type=int, default=10, help="number of clusters used by LVDD KMeans selection")
    parser.add_argument(
        "--lvdd_select_mode",
        type=str,
        default="random",
        choices=["full", "kmeans", "random", "DAPS"],
        help="LVDD latent-video selection mode",
    )
    parser.add_argument("--lvdd_batch_size", type=int, default=400, help="batch size for LVDD Tucker decomposition")
    parser.add_argument(
        "--select_mode",
        type=str,
        default="summary_dpp",
        choices=["summary_dpp", "random", "full"],
        help="video selection strategy",
    )
    parser.add_argument("--latent_file", type=str, default=None, help="path to precomputed latent tensor")
    parser.add_argument("--encode_batch_size", type=int, default=8, help="encoding/decoding batch size")
    parser.add_argument("--save_latent_cache", action="store_true", help="deprecated: latent cache is now auto-saved when latent_cache_dir is set")
    parser.add_argument("--latent_cache_dir", type=str, default="./latent_cache", help="root directory for automatic latent caching")
    parser.add_argument("--importance_temporal_weight", type=float, default=0.45, help="temporal importance weight")
    parser.add_argument("--importance_spatial_weight", type=float, default=0.15, help="spatial importance weight")
    parser.add_argument("--importance_local_weight", type=float, default=0.25, help="local residual importance weight")
    parser.add_argument("--importance_energy_weight", type=float, default=0.15, help="channel energy importance weight")
    parser.add_argument("--high_token_ratio", type=float, default=0.10, help="ratio of high-importance tokens")
    parser.add_argument("--medium_token_ratio", type=float, default=0.25, help="ratio of medium-importance tokens")
    parser.add_argument(
        "--videomae_model_id",
        type=str,
        default="MCG-NJU/videomae-base",
        help="pretrained VideoMAE checkpoint name or local path",
    )
    parser.add_argument(
        "--timesformer_model_id",
        type=str,
        default="facebook/timesformer-base-finetuned-k400",
        help="pretrained TimeSformer checkpoint name or local path",
    )
    parser.add_argument(
        "--video_transformer_image_size",
        type=int,
        default=224,
        help="input spatial size used for VideoMAE/TimeSformer",
    )
    parser.add_argument(
        "--video_transformer_tune_mode",
        type=str,
        default="linear_probe",
        choices=["linear_probe", "full_finetune"],
        help="whether to train only the classifier head or finetune the whole transformer",
    )
    parser.add_argument(
        "--video_transformer_lr_linear_probe",
        type=float,
        default=1e-3,
        help="learning rate used when VideoMAE/TimeSformer runs in linear_probe mode",
    )
    parser.add_argument(
        "--video_transformer_lr_finetune",
        type=float,
        default=5e-5,
        help="learning rate used when VideoMAE/TimeSformer runs in full_finetune mode",
    )
    parser.add_argument(
        "--video_transformer_weight_decay",
        type=float,
        default=0.05,
        help="weight decay used by AdamW for VideoMAE/TimeSformer evaluation",
    )
    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="disable automatic mixed precision for token-based evaluation models on CUDA",
    )
    parser.add_argument(
        "--disable_data_parallel",
        action="store_true",
        help="force single-GPU execution even when multiple visible GPUs are available",
    )
    parser.add_argument(
        "--disable_eval_data_parallel",
        action="store_true",
        help="disable multi-GPU DataParallel for evaluation networks",
    )
    parser.add_argument(
        "--enable_decode_data_parallel",
        action="store_true",
        help="enable DataParallel for VAE decoding when multiple visible GPUs are available",
    )
    parser.add_argument(
        "--selected_indices_file",
        type=str,
        default=None,
        help="path to a synthetic_data.pt or json file containing selected_indices to reuse",
    )
    parser.add_argument(
        "--artifact_file",
        type=str,
        default=None,
        help="path to an existing synthetic_data.pt artifact to reuse directly for downstream evaluation",
    )
    parser.add_argument(
        "--distill_state_file",
        type=str,
        default=None,
        help="path to a distill_state.pt file to reuse selected indices, summaries, and importance maps",
    )
    args = parser.parse_args()
    args.enable_amp = not args.disable_amp

    main(args)
