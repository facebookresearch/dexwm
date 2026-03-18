import argparse
import json
import os
import glob
from tqdm import tqdm
from datetime import datetime
import pytz

def compute_stats(root_dir, output_path):
    # Recursively find all important_stats.json files under root_dir
    root_dir=root_dir+"*"
    pattern = os.path.join(root_dir,'demo', 'important_stats.json')
    print(pattern)
    stats_json_paths = glob.glob(pattern, recursive=True)
    
    stats = {
        "num_success": 0,
        "num_failures": 0,
        "num_attempts": 0,
        "ep_length_mean": 0,
        "time spent (hrs)": 0,
    }
    success_path = []
    for stats_json_path in tqdm(stats_json_paths):
        with open(stats_json_path, 'r') as file:
            data = json.load(file)
            for metric in stats:
                if metric == "ep_length_mean":
                    if "ep_length_mean" in data:
                        stats[metric] += data[metric] * data["num_success"]
                        success_path.append(stats_json_path)
                elif metric == "time spent (hrs)":
                    stats[metric] += float(data[metric])
                else:
                    stats[metric] += data[metric]

    # Compute the final stats safely (avoid division by zero)
    n = max(len(stats_json_paths), 1)
    stats["time spent (hrs)"] /= n
    stats["ep_length_mean"] /= max(stats["num_success"], 1)
    stats["success_rate"] = stats["num_success"] / max(stats["num_attempts"], 1)
    stats["total_folders"] = len(stats_json_paths)
    stats["success_path"] = success_path

    # Timezone info
    tz = pytz.timezone("America/Los_Angeles")
    now = datetime.now(tz)
    stats["this_stats_creation_time_at_LA_timezone"] = now.strftime("%Y-%m-%d-%H:%M:%S")

    print(f"stats: {stats}")

    # Save output
    with open(output_path, 'w') as json_file:
        json.dump(stats, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute stats from JSON files recursively")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory to recursively search for important_stats.json files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output_stats.json",
        help="Path to save aggregated stats JSON"
    )
    args = parser.parse_args()
    compute_stats(args.root_dir, args.output_path)
