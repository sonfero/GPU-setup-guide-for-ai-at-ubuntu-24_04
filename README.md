This guide will walk you through the following:

*   NVIDIA driver installation.
*   Installation of NVIDIA CUDA Toolkit and cuDNN library.
*   Set up a Python virtual environment to manage project dependencies cleanly.
*   Installation of the correct, CUDA-accelerated version of PyTorch.

At the end, your system will be ready for you to start your AI projects.

---

# GPU Accelerated Development Environment Setup on Ubuntu

## 1. Objective

This document outlines the systematic process for configuring a fresh Ubuntu system with a new NVIDIA GPU for AI/ML development. The goal is to establish a robust environment with the necessary drivers, libraries, and frameworks, ensuring that GPU acceleration is fully leveraged by tools like PyTorch.

## 2. Initial System Diagnosis: Driver Verification

Before installing any development libraries, the first step is to ensure the OS correctly recognizes the GPU and that the proprietary NVIDIA drivers are properly installed. The `nvidia-smi` (NVIDIA System Management Interface) utility is the canonical tool for this diagnosis.

```bash
nvidia-smi
```

A successful output from this command confirms:
- The NVIDIA driver is loaded.
- The GPU model (e.g., NVIDIA GeForce RTX 5070) is correctly identified.
- The driver's CUDA compatibility version (e.g., CUDA Version: 12.8), which dictates the versions of CUDA-dependent libraries we can use.

## 3. Core Development Environment: CUDA Toolkit

While the driver provides CUDA *compatibility*, it does not include the full development toolkit (compilers, headers, libraries). The CUDA Toolkit is essential for compiling and running GPU-accelerated applications.

The installation strategy involves adding the official NVIDIA repository to the system's package manager (`apt`).

**3.1. Add the NVIDIA CUDA Repository:**
First, we download and install the keyring to allow `apt` to verify the authenticity of the NVIDIA packages.

```bash
# Download the keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

# Install the keyring package
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Refresh package lists
sudo apt-get update
```

**3.2. Install the Toolkit:**
With the repository configured, we install the `cuda-toolkit` package.

```bash
sudo apt-get install cuda-toolkit -y
```

**3.3. Environment Configuration:**
The toolkit installs to `/usr/local/cuda-X.X`, with a `/usr/local/cuda` symlink pointing to the active version. For the system to find the toolkit's binaries (`nvcc`) and libraries, we must update the `PATH` and `LD_LIBRARY_PATH` environment variables. These exports are appended to `~/.bashrc` for persistence across shell sessions.

```bash
echo 'export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"' >> ~/.bashrc
```
**Note:** Changes to `.bashrc` require sourcing the file (`source ~/.bashrc`) or starting a new shell session to take effect.

## 4. Deep Learning Acceleration: cuDNN Library

The NVIDIA CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. It is a critical dependency for performance in frameworks like PyTorch and TensorFlow.

We installed this from a local Debian repository package provided by NVIDIA.

```bash
# 1. Download the local repo installer
wget https://developer.download.nvidia.com/compute/cudnn/9.17.0/local_installers/cudnn-local-repo-ubuntu2404-9.17.0_1.0-1_amd64.deb

# 2. Install the repo package
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.17.0_1.0-1_amd64.deb

# 3. Import the GPG key
sudo cp /var/cudnn-local-repo-ubuntu2404-9.17.0/cudnn-*-keyring.gpg /usr/share/keyrings/

# 4. Refresh package lists
sudo apt-get update

# 5. Install the cuDNN package specific to the driver's CUDA version
sudo apt-get -y install cudnn9-cuda-12
```

## 5. Application Layer: Python & PyTorch

To avoid dependency conflicts with system-level Python packages (a requirement enforced by modern Ubuntu releases via PEP 668), all project-specific development should occur within a virtual environment.

**5.1. Create and Isolate the Environment:**

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv
```

**5.2. Install PyTorch:**
The key is to install a PyTorch build that matches the CUDA version supported by the *NVIDIA driver*, not necessarily the version of the CUDA Toolkit installed. `nvidia-smi` reported CUDA 12.8 compatibility, so we target the `cu128` build. We use the `pip` executable from within the `venv` to ensure installation is localized to that environment.

```bash
# Install PyTorch, Torchvision, and Torchaudio
venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 6. Final Verification

With the full stack installed, the final step is to confirm that PyTorch can initialize its CUDA backend and recognize the GPU.

**6.1. Verification Script (`verify_gpu.py`):**

```python
import torch

if torch.cuda.is_available():
    print(f"Success! PyTorch can see your GPU.")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("Failure. PyTorch cannot see your GPU.")
```

**6.2. Execution:**
The script is executed using the virtual environment's Python interpreter.

```bash
venv/bin/python verify_gpu.py
```

A successful run confirms that the driver, toolkit, cuDNN, and PyTorch are all communicating correctly.

---

## 7. Usage Summary

- **To run CUDA commands system-wide** (e.g., `nvcc`), ensure you have started a new terminal session after the installation or have run `source ~/.bashrc`.
- **To run your AI projects**, use the Python interpreter from your virtual environment (e.g., `venv/bin/python your_script.py`) to ensure access to the installed PyTorch libraries.

This structured setup provides a stable, performant, and maintainable environment for GPU-accelerated development.
