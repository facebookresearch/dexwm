import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import concurrent.futures
import tempfile
import argparse
import glob
import cv2
from tqdm import tqdm


def is_random_noise_by_histogram(frame, threshold=0.001):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    # Normalize the histogram
    hist_normalized = hist / np.sum(hist)
    # Check for uniformity (e.g., if many bins have similar frequencies)
    # A simple check could be to see if the standard deviation of the histogram is low
    # np.std(hist_normalized): 0.02280575968325138
    if np.std(hist_normalized) < threshold:
        return True
    return False


def is_random_noise_by_edges(
    frame, low_threshold=50, high_threshold=150, edge_pixel_ratio_threshold=0.01
):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, low_threshold, high_threshold)
    # Calculate the ratio of edge pixels to total pixels
    edge_pixel_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    # edge_pixel_ratio: 0.05940451388888889
    if edge_pixel_ratio < edge_pixel_ratio_threshold:
        return True
    return False


def is_corrupted_frame(img, expected_shape):
    errors = []
    if np.isnan(img).any():
        errors.append("NaN values")
    if np.isinf(img).any():
        errors.append("Inf values")
    if img.shape != expected_shape:
        errors.append(f"Shape mismatch: {img.shape}")
    if np.count_nonzero(img) < img.size * 0.5:
        errors.append("Too many zero pixels")
    if np.sum(img) < 4e6:
        errors.append("Very low pixel sum")
    if is_random_noise_by_histogram(img):
        errors.append(
            "Relatively flat or uniform distribution of pixel values across the entire intensity range"
        )
    if is_random_noise_by_edges(img):
        errors.append("Lack of detectable edges and textures")
    return errors


def check_demo_integrity_worker(args):

    h5_file_path, group_path, save_dir = args
    result = {"group": group_path, "corrupted": False, "reasons": set()}
    all_pixels = []

    mean_diff_threshold = 250  # You can adjust this
    max_allowed_bad_frames = 3
    bad_frame_count = 0

    try:
        with h5py.File(h5_file_path, "r") as f:
            if group_path not in f:
                result["corrupted"] = True
                result["reasons"].add("Group path missing")
                return result

            dset = f[group_path]
            expected_shape = dset.shape[1:]
            num_frames = dset.shape[0]

            base_demo_path = "/".join(group_path.split("/")[:-2])
            joint_path = f"{base_demo_path}/obs/robot0_joint_pos"

            if joint_path not in f:
                result["corrupted"] = True
                result["reasons"].add("Joint position data missing")
                return result

            joint_dset = f[joint_path][:, :7]
            if joint_dset.shape[0] != num_frames:
                result["corrupted"] = True
                result["reasons"].add("Joint data length mismatch")
                return result

            # Check abrupt joint changes
            for i in range(num_frames - 1):
                diff = np.abs(joint_dset[i + 1] - joint_dset[i])
                if np.any(diff > 0.1):
                    result["corrupted"] = True
                    result["reasons"].add(f"Abrupt joint jump at frame {i}")
                    break

            prev_frame = None

            for frame_index in range(num_frames):
                img = dset[frame_index]
                all_pixels.append(img.ravel())

                # --- Static checks ---
                errors = is_corrupted_frame(img, expected_shape)
                if errors:
                    result["corrupted"] = True
                    result["reasons"].update(errors)

                # --- Mean diff check ---
                if prev_frame is not None:
                    diff = np.linalg.norm(
                        img.astype(np.float32) - prev_frame.astype(np.float32), axis=2
                    )
                    mean_diff = diff.mean()
                    if mean_diff > mean_diff_threshold:
                        bad_frame_count += 1
                        result["reasons"].add(f"High mean frame diff: {mean_diff:.2f}")
                prev_frame = img

            # Final corruption flag: allow up to 3 bad frames
            if bad_frame_count > max_allowed_bad_frames:
                result["corrupted"] = True
            else:
                result["reasons"].clear()  # If it was just minor, don't report
                result["corrupted"] = False

    except Exception as e:
        result["corrupted"] = True
        result["reasons"].add(f"Exception: {str(e)}")
        return result

    # Save histogram if corrupted
    if result["corrupted"]:
        try:
            all_pixels_concat = np.concatenate(all_pixels)
            plt.figure(figsize=(10, 8))
            plt.hist(all_pixels_concat, bins=256, range=(0, 255))
            plt.title(f"Histogram - {group_path}")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            fname = re.sub(r"[^\w\-_\. ]", "_", group_path.strip("/")) + "_hist.png"
            plt.savefig(os.path.join(save_dir, fname))
            plt.close()
        except Exception as e:
            result["reasons"].add(f"Histogram error: {str(e)}")

    return result


def scan_all_demos_parallel(
    h5_file_path,
    root="data",
    save_dir="./check_results_all",
    summary_file="corruption_summary.txt",
    max_workers=8,
    clean_h5_path="cleaned_data.hdf5",
):
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, summary_file)

    # Step 1: Find all demos
    with h5py.File(h5_file_path, "r") as f:
        demo_keys = [k for k in f[root].keys() if k.startswith("demo_")]
        print(f"Found {len(demo_keys)} demos.")

    # Step 2: Run integrity check in parallel
    tasks = [
        (h5_file_path, f"{root}/{demo}/obs/robot0_robotview_image", save_dir)
        for demo in demo_keys
    ]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(check_demo_integrity_worker, tasks):
            results.append(res)
            status = "CORRUPTED" if res["corrupted"] else "OK"
            reason_str = "; ".join(res["reasons"]) if res["corrupted"] else ""
            print(f"{res['group']}: {status} {reason_str}")

    # Step 3: Save summary
    with open(summary_path, "w") as summary:
        for res in results:
            demo_name = res["group"].split("/")[-3]
            status = "CORRUPTED" if res["corrupted"] else "OK"
            reason_str = "; ".join(res["reasons"]) if res["corrupted"] else ""
            summary.write(f"{demo_name}: {status} {reason_str}\n")

    print(f"\nAll demos checked. Summary written to {summary_path}")

    # Step 4: Copy only clean demos into a new HDF5 file
    print(f"\nCreating cleaned file: {clean_h5_path}")
    with h5py.File(h5_file_path, "r") as f_in, h5py.File(clean_h5_path, "w") as f_out:
        # Copy global attributes
        for attr in f_in.attrs:
            f_out.attrs[attr] = f_in.attrs[attr]

        # Copy root group and its attributes (e.g., env_args)
        f_out.create_group(root)
        for attr in f_in[root].attrs:
            f_out[root].attrs[attr] = f_in[root].attrs[attr]

        # Copy only clean demos
        for res in results:
            demo_name = res["group"].split("/")[-3]
            if not res["corrupted"]:
                print(f"Copying clean demo: {demo_name}")
                f_in[root].copy(demo_name, f_out[root])
            else:
                print(f"Skipping corrupted demo: {demo_name}")

    print(f"\n✅ Cleaned file saved at: {clean_h5_path}")


def process_directory(dir_path, **kwargs):
    h5_files = glob.glob(os.path.join(dir_path, "*.hdf5")) + glob.glob(
        os.path.join(dir_path, "*.h5")
    )
    if not h5_files:
        print(f"No HDF5 files found in directory: {dir_path}")
        return

    kwargs_copy = kwargs.copy()
    # Remove save_dir since it will be processed
    kwargs_copy.pop("save_dir")
    for h5_file in tqdm(h5_files):
        print(f"\nProcessing file: {h5_file}")
        base_name = os.path.splitext(os.path.basename(h5_file))[0]
        save_dir = kwargs.get("save_dir", "./check_results_all")
        per_file_save_dir = os.path.join(save_dir, base_name)
        os.makedirs(per_file_save_dir, exist_ok=True)
        clean_h5_path = os.path.join(per_file_save_dir, f"cleaned_{base_name}.hdf5")
        print(
            f"\n Info: per_file_save_dir/clean_h5_path: {per_file_save_dir} / {clean_h5_path}"
        )
        scan_all_demos_parallel(
            h5_file_path=h5_file,
            save_dir=per_file_save_dir,
            clean_h5_path=clean_h5_path,
            **kwargs_copy,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Check HDF5 demos for corruption and clean them."
    )
    parser.add_argument(
        "--h5_file_path", type=str, required=True, help="Path to the input HDF5 file"
    )
    parser.add_argument(
        "--direc", type=str, help="Path to a directory containing HDF5 files"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./check_results_all",
        help="Directory to save results",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="corruption_summary.txt",
        help="Summary filename",
    )
    parser.add_argument(
        "--max_workers", type=int, default=8, help="Number of parallel workers"
    )
    parser.add_argument(
        "--clean_h5_path",
        type=str,
        default="cleaned_data.hdf5",
        help="Path for cleaned HDF5 output",
    )
    parser.add_argument(
        "--root", type=str, default="data", help="Root group containing demos"
    )

    args = parser.parse_args()
    if args.direc:
        # When directory is given, ignore clean_h5_path arg, generate output paths automatically
        process_directory(
            args.direc,
            root=args.root,
            save_dir=args.save_dir,
            summary_file=args.summary_file,
            max_workers=args.max_workers,
        )
    else:
        scan_all_demos_parallel(
            h5_file_path=args.h5_file_path,
            root=args.root,
            save_dir=args.save_dir,
            summary_file=args.summary_file,
            max_workers=args.max_workers,
            clean_h5_path=args.clean_h5_path,
        )


if __name__ == "__main__":
    main()
