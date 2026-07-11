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

当前推荐从 GitHub 的 `v2` 分支直接安装。训练和推理仍需要本机具备可用的 NVIDIA/CUDA 环境。

1.  **创建环境：**
    ```bash
    conda create -n cryoddm python=3.10 -y
    conda activate cryoddm
    python -m pip install -U pip setuptools wheel
    ```

2.  **从 GitHub v2 分支安装：**
    ```bash
    pip install "git+https://github.com/BIT-FuweiLi/CryoDDM.git@v2"
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

如果需要更新到 `v2` 分支最新代码：

```bash
conda activate cryoddm
pip install --upgrade --force-reinstall --no-cache-dir "git+https://github.com/BIT-FuweiLi/CryoDDM.git@v2"
cryoddm
```

如果当前环境找不到 Git：

```bash
conda install git -y
```

`cs2star` 页面依赖 `csparc2star.py` 命令；如果该命令不存在，请额外安装 pyem：

```bash
pip install "git+https://github.com/asarnow/pyem.git@master"
```

## 🚀 使用流程

### 第一步：数据准备 (Home 页面)
*   **加载图像**：点击 "Open your data" 加载 `.mrc` 或 `.mrcs.gz` 文件（支持按 Ctrl 多选）。
*   **挑选噪声**：
    *   **手动**：在空白背景区域点击左键标记纯噪声（红框），Ctrl+点击可删除。
    *   **自动**：勾选 "Use particle data..."，根据导入的 `.star` 颗粒文件自动识别背景噪声区域。
*   **执行**：设置 `Box_size`（建议与颗粒大小一致）和 `Noise_save_path`，点击 "Execute" 保存噪声坐标文件。
*   **目标**：提取真实的背景噪声样本，为扩散模型构建真实的噪声分布。

### 第二步：正向模拟 (Forward 页面)
*   **输入**：选择 `Input_path` (原始 MRC 文件夹) 和 `Particles_coordinate` (.star 文件)。
*   **配置**：
    *   设置 `Particle_diameter` (像素单位)。
    *   选择 `Add_noise_parameter` 配置 (例如 Beta=0.1288, Steps=5)。这些参数控制正向扩散的调度。
    *   勾选 "Use other noise" 并加载第一步生成的 `noise_coordinates.txt`。
*   **执行**：点击 "Execute"，在 `Out_path` 生成训练数据集。
*   **目标**：模拟正向扩散过程。软件生成成对的训练数据：$s_1$ (纯信号)、$s_2$ (混合态) 和 $s_3$ (纯噪声)，建立自监督学习的基础。

### 第三步：模型训练 (Train 页面)
*   **输入**：选择包含第二步生成的 `s1`、`s2`、`s3` 子文件夹的父目录。
*   **设置**：
    *   **Batch_size**：根据 GPU 显存调整 (如 48 或 64)。
    *   **GPU_id**：指定目标显卡编号 (通常为 0)。
    *   设置 `Training_log_dir` 和 `Out_path`。
*   **执行**：点击开始训练 U-Net。最佳模型权重将保存为 `.pth` 文件（如 `checkpoint.pth`）。
*   **目标**：在 RSILB 和 TSLM 约束的指导下，训练神经网络区分结构特征与噪声。

### 第四步：去噪预测 (Predict 页面)
*   **输入**：选择 `Input_path` (包含待去噪原始 micrograph 的文件夹)。
*   **设置**：设置 `GPU_id`、`Particle_diamater` 和 `Log_dir`。
*   **模型**：勾选 "Use the Model..." 并手动选择第三步训练好的 `.pth` 文件。
*   **执行**：软件将处理全尺寸图像，并将去噪后的清晰图像输出到 `Out_path`。
*   **目标**：恢复原始显微图像中的高保真结构信息，促进后续更精确的 3D 分类和重构。

### 第五步：格式转换 (cs2star 页面)
*   **功能**：将 CryoSPARC 导出的颗粒数据 (`.cs`) 转换为 RELION 兼容格式 (`.star`) 的实用工具。
*   **设置**：
    *   `Project_path`: CryoSPARC 作业目录路径。
    *   `y_value`: CryoSPARC 中使用的图像大小（第一维）。
    *   `Num_projects`: 连续处理的作业数量。
*   **输出**：生成 `particles_relion.star`、`cleaned_particles_relion.star` 以及 `Invert.star`（已翻转 Y 轴坐标以兼容 Relion）。

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

The recommended path is installing directly from the `v2` branch on GitHub. Training and inference still require a working NVIDIA/CUDA setup on the target machine.

1.  **Create the environment:**
    ```bash
    conda create -n cryoddm python=3.10 -y
    conda activate cryoddm
    python -m pip install -U pip setuptools wheel
    ```

2.  **Install from the GitHub v2 branch:**
    ```bash
    pip install "git+https://github.com/BIT-FuweiLi/CryoDDM.git@v2"
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

To update to the latest code on the `v2` branch:

```bash
conda activate cryoddm
pip install --upgrade --force-reinstall --no-cache-dir "git+https://github.com/BIT-FuweiLi/CryoDDM.git@v2"
cryoddm
```

If Git is not available in the environment:

```bash
conda install git -y
```

The `cs2star` tab depends on the `csparc2star.py` command. If it is missing, install pyem separately:

```bash
pip install "git+https://github.com/asarnow/pyem.git@master"
```

## 🚀 Usage Workflow 

### Step 1: Data Preparation (Home Tab)
*   **Load Image**: Click "Open your data" to load `.mrc` or `.mrcs.gz` files (supports `Ctrl+Click` for multi-selection).
*   **Pick Noise**: 
    *   **Manual**: Left-click on empty background areas to mark pure noise patches (red box). `Ctrl+Click` to remove.
    *   **Auto**: Check "Use particle data..." to automatically identify background noise regions based on an imported `.star` particle file.
*   **Execute**: Set the `Box_size` (recommended to match particle box size) and `Noise_save_path`, then click "Execute".
*   **Goal**: Extract real background noise samples to construct a realistic noise distribution for the diffusion model.

### Step 2: Forward Simulation (Forward Tab)
*   **Inputs**: Select the `Input_path` (folder containing Raw MRCs) and the `Particles_coordinate` file (`.star`).
*   **Configuration**: 
    *   Set `Particle_diameter` (in pixels).
    *   Choose `Add_noise_parameter` config (e.g., Beta=0.1288, Steps=5). These parameters control the forward diffusion schedule.
    *   Check "Use other noise" to load the `noise_coordinates.txt` generated in Step 1.
*   **Execute**: Click "Execute" to generate the training dataset in the `Out_path`.
*   **Goal**: Simulate the forward diffusion process. The software generates paired training data: $s_1$ (Signal), $s_2$ (Mixed state), and $s_3$ (Pure Noise), creating a self-supervised learning foundation.

### Step 3: Model Training (Train Tab)
*   **Input**: Select the parent folder containing the `s1`, `s2`, and `s3` subfolders generated in Step 2.
*   **Settings**: 
    *   **Batch_size**: Adjust based on GPU memory (e.g., 48 or 64).
    *   **GPU_id**: Specify the target GPU index (usually 0).
    *   Set paths for `Training_log_dir` and `Out_path`.
*   **Execute**: Click to start training the U-Net. The best model weights will be saved as `.pth` files (e.g., `checkpoint.pth` or epoch-numbered files).
*   **Goal**: Train the neural network to differentiate between structural features and noise, guided by the RSILB and TSLM constraints.

### Step 4: Denoising Prediction (Predict Tab)
*   **Input**: Select the `Input_path` (folder containing the raw micrographs you wish to denoise).
*   **Settings**: Set `GPU_id`, `Particle_diamater`, and `Log_dir`.
*   **Model**: Check "Use the Model..." to manually select the trained `.pth` file from Step 3.
*   **Execute**: The software processes full micrographs and outputs clean, denoised images to the `Out_path`.
*   **Goal**: Restore high-fidelity structural information from raw micrographs to facilitate accurate downstream 3D classification and reconstruction.

### Step 5: Format Conversion (cs2star Tab)
*   **Function**: A utility tool to convert CryoSPARC exported particle data (`.cs`) into RELION-compatible format (`.star`).
*   **Settings**: 
    *   `Project_path`: Path to the CryoSPARC job directory.
    *   `y_value`: The image size (first dimension) used in CryoSPARC.
    *   `Num_projects`: Number of sequential jobs to process.
*   **Output**: Generates `particles_relion.star`, `cleaned_particles_relion.star`, and `Invert.star` (with Y-axis coordinates inverted for compatibility).

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
