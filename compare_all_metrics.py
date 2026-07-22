import os
import csv
import cv2
import numpy as np
from collections import defaultdict
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DATA_DIR = "result_data"
OUTPUT_CSV = "metrics_results.csv"

VALID_SUFFIXES = ["Original", "LVDD", "Ours"]
VALID_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Can't read the image:{path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_to_match(img, ref):
    if img.shape[:2] != ref.shape[:2]:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
    return img


def calc_mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def calc_metrics(ref, pred):
    pred = resize_to_match(pred, ref)

    mse = calc_mse(ref, pred)
    psnr = peak_signal_noise_ratio(ref, pred, data_range=255)
    ssim = structural_similarity(ref, pred, channel_axis=2, data_range=255)

    return mse, psnr, ssim


def scan_files(data_dir):
    grouped = defaultdict(dict)

    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)

        if not os.path.isfile(full_path):
            continue

        stem, ext = os.path.splitext(fname)
        if ext.lower() not in VALID_EXTS:
            continue

        matched = False
        for suffix in VALID_SUFFIXES:
            token = "_" + suffix
            if stem.endswith(token):
                case_name = stem[:-len(token)]
                grouped[case_name][suffix] = full_path
                matched = True
                break

        if not matched:
            print(f"[Skip] File name does not follow the rules:{fname}")

    return grouped


def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory does not exist:{DATA_DIR}")

    grouped_files = scan_files(DATA_DIR)

    if not grouped_files:
        print("No image files matching the naming rules were found.")
        return

    results = []

    print("=" * 80)
    print("Starting to calculate image difference metrics")
    print("=" * 80)

    for case_name in sorted(grouped_files.keys()):
        files = grouped_files[case_name]

        if "Original" not in files:
            print(f"[skip] {case_name} lack Original")
            continue
        if "LVDD" not in files:
            print(f"[skip] {case_name} lack LVDD")
            continue
        if "Ours" not in files:
            print(f"[skip] {case_name} lack Ours")
            continue

        original = read_image(files["Original"])
        lvdd = read_image(files["LVDD"])
        ours = read_image(files["Ours"])

        mse_lvdd, psnr_lvdd, ssim_lvdd = calc_metrics(original, lvdd)
        mse_ours, psnr_ours, ssim_ours = calc_metrics(original, ours)

        delta_mse = mse_ours - mse_lvdd
        delta_psnr = psnr_ours - psnr_lvdd
        delta_ssim = ssim_ours - ssim_lvdd

        print(f"\n[{case_name}]")
        print(f"LVDD vs Original -> MSE: {mse_lvdd:.4f}, PSNR: {psnr_lvdd:.4f}, SSIM: {ssim_lvdd:.6f}")
        print(f"Ours vs Original -> MSE: {mse_ours:.4f}, PSNR: {psnr_ours:.4f}, SSIM: {ssim_ours:.6f}")
        print(f"Ours - LVDD      -> ΔMSE: {delta_mse:.4f}, ΔPSNR: {delta_psnr:.4f}, ΔSSIM: {delta_ssim:.6f}")

        results.append({
            "case": case_name,
            "lvdd_mse": mse_lvdd,
            "lvdd_psnr": psnr_lvdd,
            "lvdd_ssim": ssim_lvdd,
            "ours_mse": mse_ours,
            "ours_psnr": psnr_ours,
            "ours_ssim": ssim_ours,
            "delta_mse": delta_mse,
            "delta_psnr": delta_psnr,
            "delta_ssim": delta_ssim,
        })

    if not results:
        print("No complete sample to calculate.")
        return

    avg_lvdd_mse = np.mean([r["lvdd_mse"] for r in results])
    avg_lvdd_psnr = np.mean([r["lvdd_psnr"] for r in results])
    avg_lvdd_ssim = np.mean([r["lvdd_ssim"] for r in results])

    avg_ours_mse = np.mean([r["ours_mse"] for r in results])
    avg_ours_psnr = np.mean([r["ours_psnr"] for r in results])
    avg_ours_ssim = np.mean([r["ours_ssim"] for r in results])

    avg_delta_mse = np.mean([r["delta_mse"] for r in results])
    avg_delta_psnr = np.mean([r["delta_psnr"] for r in results])
    avg_delta_ssim = np.mean([r["delta_ssim"] for r in results])

    print("\n" + "=" * 80)
    print("Average result")
    print("=" * 80)
    print(f"LVDD average -> MSE: {avg_lvdd_mse:.4f}, PSNR: {avg_lvdd_psnr:.4f}, SSIM: {avg_lvdd_ssim:.6f}")
    print(f"Ours average -> MSE: {avg_ours_mse:.4f}, PSNR: {avg_ours_psnr:.4f}, SSIM: {avg_ours_ssim:.6f}")
    print(f"average improvment   -> ΔMSE: {avg_delta_mse:.4f}, ΔPSNR: {avg_delta_psnr:.4f}, ΔSSIM: {avg_delta_ssim:.6f}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Case",
            "LVDD_MSE", "LVDD_PSNR", "LVDD_SSIM",
            "Ours_MSE", "Ours_PSNR", "Ours_SSIM",
            "Delta_MSE(Ours-LVDD)", "Delta_PSNR(Ours-LVDD)", "Delta_SSIM(Ours-LVDD)"
        ])

        for r in results:
            writer.writerow([
                r["case"],
                f"{r['lvdd_mse']:.6f}",
                f"{r['lvdd_psnr']:.6f}",
                f"{r['lvdd_ssim']:.6f}",
                f"{r['ours_mse']:.6f}",
                f"{r['ours_psnr']:.6f}",
                f"{r['ours_ssim']:.6f}",
                f"{r['delta_mse']:.6f}",
                f"{r['delta_psnr']:.6f}",
                f"{r['delta_ssim']:.6f}",
            ])

        writer.writerow([])
        writer.writerow([
            "Average",
            f"{avg_lvdd_mse:.6f}",
            f"{avg_lvdd_psnr:.6f}",
            f"{avg_lvdd_ssim:.6f}",
            f"{avg_ours_mse:.6f}",
            f"{avg_ours_psnr:.6f}",
            f"{avg_ours_ssim:.6f}",
            f"{avg_delta_mse:.6f}",
            f"{avg_delta_psnr:.6f}",
            f"{avg_delta_ssim:.6f}",
        ])

    print(f"\n The result has been saved to:{OUTPUT_CSV}")


if __name__ == "__main__":
    main()