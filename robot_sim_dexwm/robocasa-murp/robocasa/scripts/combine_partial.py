import h5py
import pathlib

def combine_partial_demos_from_folder(partial_folder, output_file):
    partial_folder = pathlib.Path(partial_folder)
    if not partial_folder.is_dir():
        raise ValueError(f"{partial_folder} is not a valid directory")

    # Find all .hdf5 or .h5 files in folder, sorted
    partial_files = sorted(partial_folder.glob("*.hdf5")) + sorted(partial_folder.glob("*.h5"))
    if not partial_files:
        print(f"[WARN] No HDF5 files found in {partial_folder}")
        return

    output_file = pathlib.Path(output_file)
    if output_file.exists():
        print(f"[INFO] Removing existing output file {output_file}")
        output_file.unlink()

    with h5py.File(output_file, "w") as out_f:
        data_grp = out_f.create_group("data")

        for idx, partial_path in enumerate(partial_files):
            print(f"[INFO] Processing file: {partial_path}")
            with h5py.File(partial_path, "r") as in_f:
                in_data = in_f.get("data")
                if in_data is None:
                    print(f"[WARN] No 'data' group in {partial_path}, skipping.")
                    continue

                for demo_key in in_data.keys():
                    if idx != 0 and demo_key in ("demo_0", "demo_1"):
                        print(f"[INFO] Skipping {demo_key} from {partial_path}")
                        continue

                    if demo_key in data_grp:
                        print(f"[WARN] {demo_key} already exists in output; skipping duplicate.")
                        continue

                    in_f.copy(f"data/{demo_key}", data_grp, name=demo_key)
                    print(f"[INFO] Copied {demo_key} from {partial_path}")

    print(f"[INFO] Combined dataset saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine partial HDF5 files from a folder")
    parser.add_argument("--partial_folder", type=str, help="Folder containing partial HDF5 files")
    parser.add_argument("--output_file", type=str, help="Output combined HDF5 file")

    args = parser.parse_args()

    combine_partial_demos_from_folder(args.partial_folder, args.output_file)
