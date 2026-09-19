<div align="center">

# CryoDDM
### Resolving Subtle Conformational Heterogeneity through Diffusion-Based Denoising

[**English Version**](#-cryoddm-english-version) | [**中文版**](#-cryoddm-中文版)

</div>

---

<a id="-cryoddm-中文版"></a>
# 🧊 CryoDDM 

**CryoDDM** 是一款基于扩散模型理论框架的现代化 GUI 软件，专为冷冻电镜（Cryo-EM）单颗粒分析设计。它致力于解决由高噪声掩盖导致的**微小构象异质性（Subtle Conformational Heterogeneity）**解析难题。

不同于传统的去噪方法，CryoDDM 引入了 **剩余结构信息下界 (RSILB)** 和 **训练结构损失最小化 (TSLM)** 约束。这些理论约束确保了在有效抑制背景噪声的同时，严格保留用于区分微小生物状态的高频结构细节。

## ✨ 功能特性

*   **科学性与理论完备**：基于扩散模型的去噪算法，防止信号失真，保留 3D 分类所需的高频细节。
*   **用户友好的 GUI**：基于 PySide6 构建的现代化深色界面，降低深度学习算法的使用门槛。
*   **高性能表现**：
    *   **多线程加载**：异步图像处理，确保界面流畅不卡顿。
    *   **OpenGL 加速**：利用 GPU 渲染，支持 4K/8K 显微图像的流畅缩放与平移。
    *   **智能缓存**：混合加载策略结合 LRU 缓存，高效管理内存。
*   **全流程集成**：集成了数据准备、合成数据生成（正向过程）、模型训练、去噪预测以及格式转换的一站式工作流。

## 🛠️ 系统要求

*   **操作系统**：Linux (推荐 Ubuntu/CentOS) 或 Windows 10/11。
*   **显卡 (GPU)**：NVIDIA 显卡，需支持 **CUDA** (训练和推理所必需；建议显存 8GB 以上)。
*   **Python**：建议版本 3.10。
*   **环境管理**：强烈建议安装 Anaconda 或 Miniconda。

## 📥 安装指南

当前推荐安装 GitHub 上的 V2.0 正式版（tag `v2.0`）。训练和推理仍需要本机具备可用的 NVIDIA/CUDA 环境。

1.  **创建环境：**
    ```bash
    conda create -n cryoddm python=3.10 -y
    conda activate cryoddm
    python -m pip install -U pip setuptools wheel
    ```

2.  **安装 V2.0 正式版：**

    Windows 用户请先安装 CUDA 版 PyTorch（PyPI 上的 Windows 版 torch 只支持 CPU），Linux 可跳过这一行：
    ```bash
    pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
    ```

    然后安装 CryoDDM：
    ```bash
    pip install "git+https://github.com/BIT-FuweiLi/CryoDDM.git@v2.0"
    ```

3.  **启动软件：**
    ```bash
    cryoddm
    ```

以后每次使用只需要：

```bash
conda activate cryoddm
cryoddm
```

如果需要更新到 `main` 分支最新代码：

```bash
conda activate cryoddm
pip install --upgrade --force-reinstall --no-cache-dir "git+https://github.com/BIT-FuweiLi/CryoDDM.git@main"
cryoddm
```

如果当前环境找不到 Git：

```bash
conda install git -y
```

`cs2star` 页面依赖 `csparc2star.py` 命令。`pyem` 会随 CryoDDM 自动安装，可以用下面的命令检查：

```bash
csparc2star.py --help
```

## 🚀 使用流程

### 第一步：数据准备 (Home 页面)
*   **加载图像**：点击 "Open your data" 选择 `.mrc`、`.mrcs` 或 `.mrcs.gz` 文件（可多选）。
*   **噪声坐标文件**：点击 `Noise_save_path` 旁的 "Browse"，选择或新建一个 `.txt` 文件，噪声坐标保存在这个文件中。
*   **挑选噪声**：
    *   **手动**：在空白背景区域左键点击标记纯噪声（红框），坐标会立即写入 `Noise_save_path` 文件；Ctrl+点击可删除。
    *   **自动**：勾选 "Use particle coordinates" 并选择颗粒坐标文件，再点击出现的 "Execute"，软件会避开颗粒自动寻找背景区域，把噪声坐标写入 `Noise_save_path` 文件。
*   `Box_size`：画框和自动寻找噪声时使用的框大小，建议与颗粒大小一致。
*   **目标**：提取真实的背景噪声样本，为扩散模型构建真实的噪声分布。

### 第二步：正向模拟 (Forward 页面)
*   **输入**：`Input_path(MRC)`（原始 micrograph 文件夹）和 `Particles_coordinate`（颗粒坐标文件：`.star`，或每行 `文件名 x y` 的文本文件）。
*   **配置**：
    *   `Particle_diamater`：颗粒直径（像素）。
    *   `Add_noise_parameter`：“配置 1” 为 Beta=0.1288、Total_steps=5、Start=2；“配置 2” 为 Beta=0.1、Total_steps=6、Start=2；也可以选 “自定义” 自行填写。这些参数控制正向扩散的调度。
    *   噪声坐标：默认使用第一步 `Noise_save_path` 中的文件；如需改用其他噪声坐标文件，勾选 "Use other noise?" 并选择该文件。
    *   `Y origin at bottom-left`（旁边的 (?) 图标有说明）：只有 cryoSPARC 经 pyem `csparc2star.py` 直接导出、未加 `--inverty` 的 STAR 需要勾选。CryoDDM 自己点选的坐标、RELION 的 STAR、cs2star 页面生成的 `invert.star`、IMOD `model2point` 坐标都**不要**勾选。
    *   坐标文件里的 micrograph 名带不带 cryoSPARC 的 UID 前缀（如 `006642101566427281036_`）都可以，会自动匹配到本地文件。
*   **执行**：选择 `Out_path` 后点击 "Execute"，在 `Out_path` 下生成 `s1`、`s2`、`s3`、`val` 训练数据。
*   **目标**：模拟正向扩散过程。软件生成成对的训练数据：$s_1$ (纯信号)、$s_2$ (混合态) 和 $s_3$ (纯噪声)，建立自监督学习的基础。

### 第三步：模型训练 (Train 页面)
*   **输入**：`Input_path` 选择第二步的 `Out_path`（包含 `s1`、`s2`、`s3`、`val` 子文件夹）。
*   **设置**：
    *   **Batch_size**：根据 GPU 显存调整（默认 64）。
    *   **GPU_id**：目标显卡编号（默认 0）。
    *   设置 `Training_log_dir` 和 `Out_path`。
*   **执行**：点击 "Execute" 开始训练 U-Net（101 个 epoch）。`Out_path` 中保存每个 epoch 的模型 `1.pth` … `101.pth`；训练稳定时还会自动生成 `best_model.pth`。`checkpoint.pth` 只用于记录训练状态，不能用于预测。
*   **目标**：在 RSILB 和 TSLM 约束的指导下，训练神经网络区分结构特征与噪声。

### 第四步：去噪预测 (Predict 页面)
*   **输入**：`Input_path` 选择待去噪 micrograph 所在的文件夹。
*   **设置**：`Particle_diamater`（与训练时一致）、`GPU_id`、`Log_dir` 和 `Out_path`。
*   **模型**：默认使用 Train 页面 `Out_path` 下的 `best_model.pth`。如果没有生成 `best_model.pth`，或想用其他模型，勾选 "Use the model of your choice" 并选择某个 epoch 的 `.pth` 文件（不要选 `checkpoint.pth`）。
*   **执行**：点击 "Execute"，去噪后的 micrograph 以相同文件名写入 `Out_path`；`Out_path` 中已存在的同名文件会跳过。
*   **目标**：恢复原始显微图像中的高保真结构信息，促进后续更精确的 3D 分类和重构。

### 第五步：格式转换 (cs2star 页面)
*   **功能**：将 CryoSPARC 导出的颗粒数据 (`.cs`) 转换为 RELION 兼容格式 (`.star`) 的实用工具。该页面通过 bash 调用脚本，需要在 Linux 下使用。
*   **设置**：
    *   `project_path`：cryoSPARC 作业目录，目录名为 `J<编号>`（如 `.../CS-xxx/J102`）。
    *   `output_path`：输出文件夹。
    *   `y_value`：CryoSPARC 中 micrograph 的高度，即 Y 方向像素数（第一维）。非正方形图像不要填成宽度。
    *   `num_projects`：从 `project_path` 开始连续处理的作业数量（如 3 表示 J102、J103、J104），默认 1。
*   **输出**：每个作业在 `output_path/class0`、`class1` … 中生成 `particles_relion.star`、`cleaned_particles_relion.star` 和 `invert.star`。`invert.star` 中的 Y 坐标已还原为原图行号，可直接用于 RELION，也可作为 Forward 页的 `Particles_coordinate`（不勾选 `Y origin at bottom-left`）。

---

## 📚 参考文献

```bash
@article {Li2025.12.10.693455,
	author = {Li, Fuwei and Chen, Yuanbo and Dong, Hao and Ji, Chenxuan and Wang, Xinsheng and Zhang, Chuanyang and Wang, Zupeng and Hu, Bin and Zhang, Fa and Wan, Xiaohua},
	title = {CryoDDM: CryoEM denoising diffusion model for heterogeneous conformational reconstruction},
	elocation-id = {2025.12.10.693455},
	year = {2025},
	doi = {10.64898/2025.12.10.693455},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/12/12/2025.12.10.693455},
	eprint = {https://www.biorxiv.org/content/early/2025/12/12/2025.12.10.693455.full.pdf},
	journal = {bioRxiv}
}
```
<br>
<br>

<a id="-cryoddm-english-version"></a>
# 🧊 CryoDDM

**CryoDDM** is a theoretically grounded, GUI-based software designed for Cryo-EM single-particle analysis. Built upon a **two-phase diffusion model** framework, it addresses the challenge of resolving subtle conformational heterogeneity obscured by high noise levels.

Unlike conventional denoising methods, CryoDDM introduces **Residual Structural Information Lower Bound (RSILB)** and **Training Structural Loss Minimization (TSLM)** constraints. These ensure that while noise is effectively suppressed, the high-frequency structural details essential for distinguishing subtle biological states are rigorously preserved.

## ✨ Features

*   **Scientifically Grounded**: Implements a diffusion-based denoising algorithm that prevents signal distortion and preserves high-frequency details necessary for 3D classification.
*   **User-Friendly GUI**: A modern, dark-themed interface built with PySide6, making advanced deep learning accessible to biologists.
*   **High Performance**:
    *   **Multi-threaded Loading**: Asynchronous image processing ensures a responsive UI.
    *   **OpenGL Acceleration**: GPU-rendered zooming and panning for 4K/8K micrographs.
    *   **Smart Caching**: Hybrid loading strategy with LRU caching for efficient memory management.
*   **Complete Pipeline**: Integrates data preparation, synthetic data generation (Forward process), model training, final prediction, and format conversion into a single workflow.

## 🛠️ System Requirements 

*   **OS**: Linux (Ubuntu/CentOS recommended) or Windows 10/11.
*   **GPU**: NVIDIA GPU with **CUDA** support (Essential for training and inference; 8GB+ VRAM recommended).
*   **Python**: Version 3.10.
*   **Environment**: Anaconda or Miniconda is strongly recommended.

## 📥 Installation

The recommended path is installing the V2.0 release (tag `v2.0`) from GitHub. Training and inference still require a working NVIDIA/CUDA setup on the target machine.

1.  **Create the environment:**
    ```bash
    conda create -n cryoddm python=3.10 -y
    conda activate cryoddm
    python -m pip install -U pip setuptools wheel
    ```

2.  **Install the V2.0 release:**

    On Windows, install the CUDA build of PyTorch first (the Windows torch wheel on PyPI is CPU-only); skip this line on Linux:
    ```bash
    pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
    ```

    Then install CryoDDM:
    ```bash
    pip install "git+https://github.com/BIT-FuweiLi/CryoDDM.git@v2.0"
    ```

3.  **Run the software:**
    ```bash
    cryoddm
    ```

For future launches:

```bash
conda activate cryoddm
cryoddm
```

To update to the latest code on the `main` branch:

```bash
conda activate cryoddm
pip install --upgrade --force-reinstall --no-cache-dir "git+https://github.com/BIT-FuweiLi/CryoDDM.git@main"
cryoddm
```

If Git is not available in the environment:

```bash
conda install git -y
```

The `cs2star` tab depends on the `csparc2star.py` command. `pyem` is installed automatically with CryoDDM. Check it with:

```bash
csparc2star.py --help
```

## 🚀 Usage Workflow 

### Step 1: Data Preparation (Home Tab)
*   **Load Image**: Click "Open your data" to load `.mrc`, `.mrcs` or `.mrcs.gz` files (multi-selection supported).
*   **Noise coordinate file**: Click "Browse" next to `Noise_save_path` and choose or create a `.txt` file; the noise coordinates are stored in this file.
*   **Pick Noise**: 
    *   **Manual**: Left-click on empty background areas to mark pure noise patches (red box). Each click is written to the `Noise_save_path` file immediately; `Ctrl+Click` removes a point.
    *   **Auto**: Check "Use particle coordinates", choose a particle coordinate file, then click the "Execute" button that appears. Background regions away from the particles are found automatically and written to the `Noise_save_path` file.
*   `Box_size`: box size used for drawing and for the automatic noise search; match it to the particle size.
*   **Goal**: Extract real background noise samples to construct a realistic noise distribution for the diffusion model.

### Step 2: Forward Simulation (Forward Tab)
*   **Inputs**: `Input_path(MRC)` (folder containing the raw micrographs) and `Particles_coordinate` (a `.star` file, or a text file with one `filename x y` per line).
*   **Configuration**: 
    *   `Particle_diamater`: particle diameter in pixels.
    *   `Add_noise_parameter`: "配置 1" is Beta=0.1288, Total_steps=5, Start=2; "配置 2" is Beta=0.1, Total_steps=6, Start=2; "自定义" lets you enter your own values. These parameters control the forward diffusion schedule.
    *   Noise coordinates: the file from Step 1's `Noise_save_path` is used by default. To use a different noise coordinate file, check "Use other noise?" and select it.
    *   `Y origin at bottom-left` (hover the (?) icon for help): check it only for STAR files written by pyem `csparc2star.py` without `--inverty`. Leave it unchecked for CryoDDM-picked coordinates, RELION STAR files, the cs2star tab's `invert.star`, and IMOD `model2point` coordinates.
    *   Micrograph names may carry a cryoSPARC UID prefix (e.g. `006642101566427281036_`) or not; they are matched to the local files either way.
*   **Execute**: Choose `Out_path` and click "Execute". The training data (`s1`, `s2`, `s3`, `val`) is written to `Out_path`.
*   **Goal**: Simulate the forward diffusion process. The software generates paired training data: $s_1$ (Signal), $s_2$ (Mixed state), and $s_3$ (Pure Noise), creating a self-supervised learning foundation.

### Step 3: Model Training (Train Tab)
*   **Input**: Set `Input_path` to the `Out_path` of Step 2 (the folder containing `s1`, `s2`, `s3` and `val`).
*   **Settings**: 
    *   **Batch_size**: Adjust based on GPU memory (default 64).
    *   **GPU_id**: Target GPU index (default 0).
    *   Set paths for `Training_log_dir` and `Out_path`.
*   **Execute**: Click "Execute" to train the U-Net for 101 epochs. `Out_path` receives one model per epoch (`1.pth` … `101.pth`) and, once training is stable, an automatically selected `best_model.pth`. `checkpoint.pth` only stores the training state and cannot be used for prediction.
*   **Goal**: Train the neural network to differentiate between structural features and noise, guided by the RSILB and TSLM constraints.

### Step 4: Denoising Prediction (Predict Tab)
*   **Input**: Set `Input_path` to the folder containing the micrographs to denoise.
*   **Settings**: `Particle_diamater` (same as for training), `GPU_id`, `Log_dir` and `Out_path`.
*   **Model**: By default the `best_model.pth` in the Train tab's `Out_path` is used. If no `best_model.pth` was produced, or to use another model, check "Use the model of your choice" and select an epoch `.pth` file (not `checkpoint.pth`).
*   **Execute**: Click "Execute". Denoised micrographs are written to `Out_path` under the same file names; files that already exist in `Out_path` are skipped.
*   **Goal**: Restore high-fidelity structural information from raw micrographs to facilitate accurate downstream 3D classification and reconstruction.

### Step 5: Format Conversion (cs2star Tab)
*   **Function**: A utility tool to convert CryoSPARC exported particle data (`.cs`) into RELION-compatible format (`.star`). The tab runs a bash script and must be used on Linux.
*   **Settings**: 
    *   `project_path`: the cryoSPARC job directory, named `J<number>` (e.g. `.../CS-xxx/J102`).
    *   `output_path`: output folder.
    *   `y_value`: The micrograph height in CryoSPARC, i.e. the number of pixels along Y (first dimension). For non-square micrographs do not enter the width.
    *   `num_projects`: number of consecutive jobs to convert starting from `project_path` (e.g. 3 means J102, J103, J104); default 1.
*   **Output**: For each job, `output_path/class0`, `class1`, … contain `particles_relion.star`, `cleaned_particles_relion.star` and `invert.star`. The Y coordinates in `invert.star` are restored to micrograph row indices, so it can be used directly in RELION or as the Forward tab's `Particles_coordinate` (leave `Y origin at bottom-left` unchecked).

---

## 📚 Reference

```bash
@article {Li2025.12.10.693455,
	author = {Li, Fuwei and Chen, Yuanbo and Dong, Hao and Ji, Chenxuan and Wang, Xinsheng and Zhang, Chuanyang and Wang, Zupeng and Hu, Bin and Zhang, Fa and Wan, Xiaohua},
	title = {CryoDDM: CryoEM denoising diffusion model for heterogeneous conformational reconstruction},
	elocation-id = {2025.12.10.693455},
	year = {2025},
	doi = {10.64898/2025.12.10.693455},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/12/12/2025.12.10.693455},
	eprint = {https://www.biorxiv.org/content/early/2025/12/12/2025.12.10.693455.full.pdf},
	journal = {bioRxiv}
}
```
