## DexWM: World Models for Learning Dexterous Hand-Object Interactions from Human Videos <br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/pdf/2512.13644) | [Project Page](https://raktimgg.github.io/dexwm/)

This repo contains the official PyTorch implementation of DexWM: World Models for Learning Dexterous Hand-Object Interactions from Human Videos.

**Authors:** <br>Raktim Gautam Goswami<sup>1,2</sup>, Amir Bar<sup>1</sup>, David Fan<sup>1</sup>, Tsung-Yen Yang<sup>1</sup>, Gaoyue Zhou<sup>1,2</sup>, Prashanth Krishnamurthy<sup>2</sup>, Michael Rabbat<sup>1</sup>, Farshad Khorrami<sup>2</sup>, Yann LeCun<sup>1,2</sup>

<sup>1</sup> Meta-FAIR
<sup>2</sup> New York University


## Setup
Download the repo and set up the environment:
```
git clone ttps://github.com/facebookresearch/dexwm
conda create -n dexwm python=3.11
conda activate dexwm
pip install -r requirements.txt
```

## Data
DexWM is pre-trained in EgoDex and DROID datasets and fine-tuned on exploratory sequences of the RoboCasa simulation data.
Download the [EgoDex](https://github.com/apple/ml-egodex), [DROID](https://droid-dataset.github.io/), [RoboCasa Random]() datasets. See the end of this README for the expected directory structure inside each dataset folder.

## Training
### Pre-Train on EgoDex and DROID

**Note:**
Change the `egodex_root_folder` and `droid_root_folder` locations in the config file before running the code.

Using torchrun:
```bash
bash scripts/train_torchrun.sh --job_dir <job_dir>
```
Update the script variables to match your available compute resources (e.g., number of nodes, GPUs per node, and host address). Defaults are 1 node, 8 GPUs per node, and `localhost`.

Or using submitit and slurm:
```bash
bash scripts/train_submitit.sh
```
Update the script variables to match your available compute resources and job_dir. By default, this script trains the model on 32 nodes with 8 GPUs each.

Or locally on one GPU for debug:
```bash
python train_wm.py --config configs/egodex_and_droid.yaml --job_dir <job_dir>
```

On the first training run, the code generates `split_indices_droid.json` to define a DROID validation split. This file is only used to report/track validation loss and is not used elsewhere.

### Fine-Tune on RoboCasa Random Data

Change the `root_folder` location and `resume` path to the pre-trained model in the config file before running the code.

Using torchrun:
```bash
bash scripts/multistep_train_torchrun.sh --job_dir <job_dir>
```
Update the script variables to match your available compute resources (e.g., number of nodes, GPUs per node, and host address). Defaults are 1 node, 1 GPUs per node, and `localhost`.

Or using submitit and slurm:
```bash
bash scripts/multistep_train_submitit.sh
```
Update the script variables to match your available compute resources and job_dir. By default, this script trains the model on 1 nodes with 1 GPUs each.

Or locally on one GPU for debug:
```bash
python train_multistep_wm.py --config configs/robocasa_random_multistep.yaml --job_dir <job_dir>
```

On the first training run, the code generates `split_indices_robocasa_random.json` to define a RoboCasa Random validation split. This file is only used to report/track validation loss and is not used elsewhere.

## Evaluation
### Rollout L2 Error and PCK on EgoDex
1. **Set the model checkpoint:**
   Edit `test_scripts/test_script.sh` and update the model checkpoint path to the checkpoint you want to evaluate.
2. **Download the keypoint model:**
   Evaluation also uses a separately trained **keypoint model** to predict keypoints from the world model’s predicted latent states. Download this model from the checkpoint download page and configure its path in `test_scripts/test_script.sh` as well.
3. **(Optional) Visualization:**
   The test script can visualize predicted states. To enable this, you must train a decoder and configure the decoder path/settings in the code.
#### Run evaluation
```bash
bash test_scripts/test_script.sh
```
This writes two rollout metrics to the `output_dir` specified in `test_scripts/test_script.sh`:

* L2 Error
* PCK (Percentage of Correct Keypoints)

Each metric is saved as an array evaluated every 0.2 seconds, from 0.2s up to 4.0s.

#### Compute summary statistics
To view the aggregated losses similar to the format reported in the paper, run
```bash
python test_scripts/result_stats.py --output_dir <output_dir>
```

### Robot Manipulation Tasks

1. Install and configure RoboCasa simulator with MURP robot following the instructions [here](./robot_sim_dexwm/).
2. Download the [Pick-and-Place]() dataset. It provides the visual goal images used for the manipulation tasks.
3. Run evaluation
   ```bash
   conda activate robot_sim_dexwm
   bash scripts/test_robot_sim.sh
   ```
   Before running, update the script variables to match your compute setup (e.g., number of nodes/GPUs), job_dir, and any other relevant settings. By default, the script uses 1 node with 8 GPUs.
4. At the end of evaluation, a res.json file will be generated in the job_dir which will contain a dictionary with all the task names and corresponding success/failure.


## Dataset Directory Structure

The EgoDex and DROID datasets are aranged as follows:
```
egodex
├── train
│   ├── <task_1>
│   │   ├── 0.hdf5
│   │   ├── 0.mp4
│   │   ├── 1.hdf5
│   │   └── 1.mp4
│   │   ...
│   ├── <task_2>
│   │   ├── 0.hdf5
│   │   ├── 0.mp4
│   │   ├── 1.hdf5
│   │   └── 1.mp4
│   │   ...
│   ...
├── test
│   ├── <task_k>
│   │   ├── 0.hdf5
│   │   ├── 0.mp4
│   │   ├── 1.hdf5
│   │   └── 1.mp4
│   │   ...
│   ...
```

```
DROID
├── <lab_name>
│   ├── success
│   │   ├── <date_1>
│   │   │   ├── <time_1>
│   │   │   │   ├── recordings
│   │   │   │   │   ├── MP4
│   │   │   │   │   │   └── ...
│   │   │   │   │   ├── SVO
│   │   │   │   │   │   └── ...
│   │   │   │   ├── metadata_....json
│   │   │   │   └── ...
│   │   └── ...
│   │   ...
│   ├── failure
│   │   ├── <date_i>
│   │   │   ├── <time_j>
│   │   │   │   ├── recordings
│   │   │   │   │   ├── MP4
│   │   │   │   │   │   └── ...
│   │   │   │   │   ├── SVO
│   │   │   │   │   │   └── ...
│   │   │   │   ├── metadata_....json
│   │   │   │   └── ...
│   │   └── ...
│   │   ...
```
```
robocasa_random_data
├── exploratory_movements
│   ├── combine_demos_0.hdf5
│   └── combine_demos_1.hdf5
│   │   ...
├── gripper_open_and_close
│   ├── combine_demos_0.hdf5
│   └── combine_demos_1.hdf5
│   │   ...
```
## BibTeX

```bibtex
@article{goswami2025world,
  title={World Models for Learning Dexterous Hand-Object Interactions from Human Videos},
  author={Goswami, Raktim Gautam and Bar, Amir and Fan, David and Yang, Tsung-Yen and Zhou, Gaoyue and Krishnamurthy, Prashanth and Rabbat, Michael and Khorrami, Farshad and LeCun, Yann},
  journal={arXiv preprint arXiv:2512.13644},
  year={2026}
}
```
