# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
import yaml
from pathlib import Path

import train_multistep_wm as trainer
from train_multistep_wm import get_latest_checkpoint
import submitit


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--cpus_per_task", default=16, type=int, help="Number of CPUs per node")
    parser.add_argument('--config', type=str, default='configs/ego4d.yaml', help='Path to the config file')
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--partition", default="learn", type=str, help="Partition where to submit")
    parser.add_argument("--qos", default="low", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Request 32G V100 GPUs")
    parser.add_argument("--debug", action='store_true', help="Debug mode")
    parser.add_argument("--use_fsdp", action='store_true', help="Use FSDP instead of DDP")
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/home/raktimgg/Projects/HOWM/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(job_dir):
    if job_dir == "":
        # Init file must not exist, but it's parent dir must exist.
        os.makedirs(str(get_shared_folder()), exist_ok=True)
        init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    else:
        os.makedirs(job_dir, exist_ok=True)
        init_file = Path(job_dir) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args, train_args):
        self.args = args
        self.train_args = train_args

    def __call__(self):
        import train_multistep_wm as trainer
        self._setup_gpu_args()
        trainer.main(self.args, self.train_args)

    def checkpoint(self):
        import submitit

        self.args.dist_url = get_init_file(self.args.job_dir).as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args, self.train_args)
        return submitit.helpers.DelayedSubmission(empty_trainer)


    def _setup_gpu_args(self):
        # Check if we're running in debug mode (torchrun) or submitit mode
        if self.args.debug or "RANK" in os.environ:
            # Debug mode: running with torchrun locally
            self.args.gpu = int(os.environ.get("LOCAL_RANK", 0))
            self.args.rank = int(os.environ.get("RANK", 0))
            self.args.world_size = int(os.environ.get("WORLD_SIZE", 1))

            # For debug mode, replace %j with "debug"
            self.args.log_dir = self.args.output_dir
            self.args.log_dir = self.args.output_dir

            print(f"Debug mode - Process group: {self.args.world_size} tasks, rank: {self.args.rank}, local_rank: {self.args.gpu}")

        else:
            job_env = submitit.JobEnvironment()
            self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
            self.args.log_dir = self.args.output_dir
            self.args.gpu = job_env.local_rank
            self.args.rank = job_env.global_rank
            self.args.world_size = job_env.num_tasks

            print(f"Submitit mode - Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}, job_id: {job_env.job_id}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"
    os.makedirs(args.job_dir, exist_ok=True)

    fname = args.config
    with open(fname, 'r') as y_file:
        train_args = yaml.load(y_file, Loader=yaml.FullLoader)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    cpus_per_task = args.cpus_per_task
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}

    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'

    executor.update_parameters(
        mem_gb=210 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=cpus_per_task,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        qos=args.qos,
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_account='fair_amaia_cw_video',
        **kwargs
    )

    executor.update_parameters(name="dexwm")

    # Don't set resume here - let train_multistep_wm.py handle checkpoint detection
    # to ensure it always uses the latest checkpoint, especially after preemption

    args.dist_url = get_init_file(args.job_dir).as_uri()
    args.output_dir = args.job_dir
    args.log_dir = args.job_dir
    train_args['job_dir'] = args.job_dir

    trainer = Trainer(args, train_args)
    if args.debug:
        trainer()
    else:
        job = executor.submit(trainer)
        print("Submitted job_id:", job.job_id)
        print(job.job_id)

if __name__ == "__main__":
    main()
