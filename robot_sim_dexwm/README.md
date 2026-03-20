# Robot Simulation for DexWM

This repo is a supporting simulation software for [DexWM]() and is used for data generation, and evaluation of the model. The repo adopts the code from [RoboCasa](https://github.com/robocasa/robocasa) with key changes to incorporate the MURP robot.

## Installation steps:
1. Set up conda environment:l
    ```sh
    conda create -c conda-forge -n robot_sim_dexwm python=3.10
    ```
2. Activate conda environment:
    ```sh
    conda activate robot_sim_dexwm
    ```
3. Clone and setup robosuite dependency:

    ```sh
    cd robosuite_murp
    pip install -e .
    ```

4. Setup Robocasa:

    ```sh
    cd robocasa-murp
    pip install -e .
    pip install imageio

    (optional: if running into issues with numba/numpy, run: conda install -c numba numba=0.56.4 -y)
    ```

5. Install the package and download assets:
    ```sh
    python robocasa/scripts/download_kitchen_assets.py
    python robocasa/scripts/setup_macros.py
    python robosuite/scripts/setup_macros.py
    ```

    Aigen objects assets have a higher probability of not getting installed during the process. To avoid any errors , we install it manually :
    ```sh
    cd robocasa-murp/robocasa/models/assets/objects
    wget https://utexas.box.com/shared/static/os3hrui06lasnuvwqpmwn0wcrduh6jg3.zip
    unzip
    ```

6. Re-Install Numpy and Numba versions (If faced with error):
    ```sh
    pip install numpy==1.23.3 numba==0.56.4
    ```


## BibTeX

### DexWM:
```bibtex
@article{goswami2025world,
  title={World Models for Learning Dexterous Hand-Object Interactions from Human Videos},
  author={Goswami, Raktim Gautam and Bar, Amir and Fan, David and Yang, Tsung-Yen and Zhou, Gaoyue and Krishnamurthy, Prashanth and Rabbat, Michael and Khorrami, Farshad and LeCun, Yann},
  journal={arXiv preprint arXiv:2512.13644},
  year={2026}
}
}
```

### RoboCasa:
```bibtex
@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author={Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle={Robotics: Science and Systems},
  year={2024}
}
}
```
