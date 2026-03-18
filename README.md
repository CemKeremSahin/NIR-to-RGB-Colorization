# NIR-to-RGB Colorization: A Comparative Study of Deep Architectures

This repository contains the implementation and comparative analysis of various deep learning architectures for Near-Infrared (NIR) to RGB image colorization. This work is inspired by the methodologies presented in the paper *"Multi-Band NIR Colorization via Dual-Teacher Color and Structure Distillation"* (IEEE Access, 2025).

The primary goal is to convert single-band NIR images, which are rich in structural detail but lack chrominance information, into realistic RGB images while preserving semantic and structural integrity.

##  Project Overview

We implemented and evaluated five distinct network architectures, training each extensively for **3,500 epochs** to ensure convergence and robust performance comparison.

### Implemented Architectures
1.  **Vanilla Autoencoder:** Serves as the baseline model for feature extraction and reconstruction.
2.  **U-Net:** Utilizes skip connections to preserve low-level structural details during upsampling.
3.  **U-Net-GAN:** Adds an adversarial loss component to the standard U-Net to improve perceptual quality and reduce blurriness.
4.  **MUGAN:** A specialized architecture designed for thermal/infrared colorization using mixed skip connections.
5.  **S-Net (S-shaped Network):** Focuses on enhancing object boundaries during the colorization process.

---
## Dataset Information
Datasets Used:
* Real-World Dataset: Includes 538 image pairs (458 training / 80 testing) covering complex indoor and outdoor scenes like streets, buildings, and vegetation.
* All images are normalized to (0, 1) and resized to 480 * 640 pixels for efficient training.


 ## Key Highlights 
 * Extensive Training: Models were trained for 3,500 epochs (Learning Rate: $2 \times 10^{-4}$) to ensure maximum convergence and stability.
 * Multi-Architecture Comparison: Side-by-side performance analysis of 5 state-of-the-art architectures.
 * Quantitative Superiority: U-Net achieved the highest fidelity with a PSNR of 21.7313 and SSIM of 0.8163.
 * Modular Structure: Separated training notebooks and inference scripts for easy reproducibility.



##  Quantitative Evaluation Metrics

The models were evaluated using standard image reconstruction metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Color Difference ($\Delta E$).

| Model | PSNR (dB) ↑ | SSIM ↑ | $\Delta E$ ↓ | PSNR Std Dev |
| :--- | :---: | :---: | :---: | :---: |
| **U-Net** | **21.7313** | **0.8163** | **7.6010** | 4.6173 |
| U-Net-GAN | 21.6398 | 0.7984 | 7.6627 | 4.6612 |
| Vanilla Autoencoder | 21.5618 | 0.7926 | 7.7251 | 4.6175 |
| MUGAN | 21.4443 | 0.7859 | 7.9243 | 4.3300 |
| S-Net | 21.0060 | 0.7692 | 8.3460 | 4.2184 |

**Analysis:** Based on 3,500 epochs of training, the standard **U-Net** architecture achieved the highest performance in both signal fidelity (PSNR) and structural preservation (SSIM).

---

##  Repository Structure

```text
├── Notebooks/          # Jupyter (.ipynb) notebooks containing the training workflows for Colab/Local GPU.
├── images/             # Documentation images and samples used in this README.
├── test_scripts/       # Python scripts (.py) for inference, testing, and metric calculation (PyCharm).
└── README.md           # Project documentation.
```
 Installation and Usage
Clone the repository:

Bash
git clone [https://github.com/CemKeremSahin/NIR-to-RGB-Colorization.git](https://github.com/CemKeremSahin/NIR-to-RGB-Colorization.git)
cd NIR-to-RGB-Colorization
Training: Open the desired notebook in the Notebooks/ folder using Google Colab or your local Jupyter environment.

Testing/Inference: Use the scripts located in test_scripts/ to run inference on your own data using pre-trained weights.

Acknowledgments and References
This project references the methodologies and datasets discussed in the following publication:

T.-S. Park, Y.-M. Jeong, and J.-O. Kim, "Multi-Band NIR Colorization via Dual-Teacher Color and Structure Distillation," IEEE Access, 2025.

If you find this comparative study useful for your research, please consider citing this repository and the reference paper.

# Contact
For research inquiries, professional collaborations, or any questions regarding the models and datasets, please feel free to reach out:

Author: Cem Kerem Şahin

Email: [cemkeremsahin@gmail.com]

GitHub: @CemKeremSahin
