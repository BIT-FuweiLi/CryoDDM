# CryoDDM - Cryo-EM Image Denoising Tool

**CryoDDM** is a modern, GUI-based software designed for Cryo-EM micrograph denoising. Built upon the U-Net architecture and Denoising Diffusion Probabilistic Models (DDPM) concepts, it provides a complete pipeline from data preparation to final image restoration.

**CryoDDM** 是一款专为冷冻电镜（Cryo-EM）显微图像去噪设计的现代化 GUI 软件。基于 U-Net 架构和扩散模型（DDPM）思想，它提供了从数据准备到最终图像修复的完整全流程解决方案。

---

## ✨ Features / 功能特性

*   **User-Friendly GUI**: Built with PySide6, offering a modern dark-themed interface.
    *   **现代化界面**：基于 PySide6 构建，提供舒适的暗色主题操作界面。
*   **High Performance**:
    *   **Multi-threaded Loading**: Asynchronous image loading prevents UI freezing.
    *   **OpenGL Acceleration**: Smooth zooming and panning for 4K/8K images using GPU rendering.
    *   **Smart Caching**: Hybrid loading strategy with LRU caching to manage memory efficiently.
    *   **高性能**：多线程异步加载防止界面卡顿；OpenGL 硬件加速实现大图丝滑缩放；智能 LRU 缓存管理内存。
*   **Complete Pipeline / 全流程支持**:
    *   **Home**: MRC/MRCS viewing, particle picking, and noise extraction.
    *   **Forward**: Generate synthetic training data (Signal + Noise mixing).
    *   **Train**: Train the U-Net model with customizable parameters.
    *   **Predict**: Denoise full micrographs using the trained model.

---

## 🛠️ System Requirements / 系统要求

*   **OS**: Linux (Recommended) or Windows.
*   **GPU**: NVIDIA GPU with CUDA support (Essential for training/inference).
*   **Driver**: Compatible with CUDA 11.8 or 12.1.
*   **Python**: 3.10

---

## 📥 Installation / 安装

This method will automatically create a virtual environment and install all dependencies including PyTorch and CUDA support.
此方法会自动创建虚拟环境并安装所有依赖，包括 PyTorch 和 CUDA 支持。

1.  **Clone the repository / 下载代码:**
    ```bash
    git clone -b online_preview https://github.com/BIT-FuweiLi/CryoDDM.git
    cd CryoDDM
    ```

2.  **Create environment / 创建环境:**
    ```bash
    conda env create -f environment.yaml
    ```

3.  **Activate environment / 激活环境:**
    ```bash
    conda activate cryoddm
    ```

4.  **Run the software / 运行软件:**
    ```bash
    python main.py
    ```
---

## 🚀 Usage Workflow / 使用流程

### Step 1: Data Preparation (Home Tab)
*   **Load Image**: Click the folder icon to load `.mrc` or `.mrcs.gz` files.
*   **Pick Particles**: Left-click on particles to mark them (red box). Ctrl+Click to remove.
*   **Save Coordinates**: Click "Save TXT" to save particle coordinates.
*   **Extract Noise**: Check "Use particle data...", set Box Size, and click "Execute" to extract pure noise patches.
*   **准备数据**：加载图像，手动挑选少量颗粒并保存坐标，然后点击 Execute 提取背景噪声。

### Step 2: Forward Simulation (Forward Tab)
*   **Input**: Original Micrographs path & Particle Coordinates file.
*   **Noise**: Select the `noise_coordinates.txt` generated in Step 1.
*   **Config**: Set particle diameter and simulation parameters (Beta, Steps).
*   **Execute**: Generates synthetic training datasets (`s1`, `s2`, `s3`) in the output folder.
*   **正向模拟**：利用真实的颗粒和噪声，基于扩散公式生成成对的训练数据。

### Step 3: Model Training (Train Tab)
*   **Input**: Select the folder containing `s1/s2/s3` (from Step 2).
*   **Settings**: Set Batch Size (e.g., 64) and GPU ID.
*   **Execute**: Trains the U-Net model. Check the log for progress. The model will be saved as `.pth` files.
*   **模型训练**：使用合成数据训练去噪网络。

### Step 4: Denoising Prediction (Predict Tab)
*   **Input**: Select raw micrographs folder to denoise.
*   **Model**: Select the trained `.pth` model (or use default).
*   **Execute**: Outputs clean, denoised micrographs.
*   **去噪预测**：加载训练好的模型，对原始数据进行去噪处理。

---
