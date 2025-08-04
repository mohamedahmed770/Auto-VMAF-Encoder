# AUTO VMAF ENCODER

**A data-driven encoding tool that creates high-quality, efficient video encodes by targeting a specific VMAF (Video Multimethod Assessment Fusion) score. This script automates the entire process: it intelligently analyzes videos to find the optimal quality settings, logs performance data from every encode to provide accurate ETA predictions, and displays all progress in a real-time console UI.**

</div>

![License](https://img.shields.io/badge/license-MIT-green)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)


---

## 🎯 Key Features

* **🧠 Smart Encoding**
    * **Intelligent Sampling:** Analyzes video samples using multiple methods (PySceneDetect, keyframes, intervals) to assess media complexity.
    * **VMAF-Targeted Quality:** Uses a binary search algorithm to find the optimal CQ/CRF value that achieves your target VMAF score with precision.


* **⚙️ Multi-Encoder Support**
    * **NVENC AV1:** Hardware-accelerated encoding with configurable presets.
    * **SVT-AV1:** High-quality software encoding with customizable presets and film grain settings.
    * **Color Space Preservation:** Maintains HDR, color primaries, and transfer characteristics from the source.

* **🚀 Advanced Caching System**
    * **VMAF Cache:** Avoids re-testing identical video samples, dramatically speeding up re-runs.
    * **Performance Database:** Learns from past encodes to provide increasingly accurate ETA predictions.

* **💻 Real-Time Performance Monitoring**
    * **Live Progress UI:** A beautiful and functional console interface built with `rich`.
    * **Multi-Threading:** Process multiple files in parallel with configurable worker limits.

* **📂 Intelligent File Management**
    * **Configurable Filtering:** Skip files based on duration, filesize, or bitrate.
    * **Size Reduction Validation:** Only replaces the source file if the encode is meaningfully smaller.
    * **Flexible I/O:** Configurable input/output directories and file naming schemes.
---
## 🖥️ Console Interface

The script provides a clean and powerful live console UI powered by the `rich` library to monitor the entire encoding process at a glance.


![Live Demo of Auto VMAF Encoder](https://github.com/Snickrr/Auto-VMAF-Encoder/blob/main/images/demo.gif) 

**The interface provides at-a-glance information on:**

* **Overall Batch Progress:** A summary panel shows files completed, total space saved, elapsed time, and a dynamically updated ETA (which improves in accuracy over time).
* **Parallel Worker Status:** See the status and logs for each video being processed in its own dedicated panel.
* **Real-Time Progress Bars:** Detailed progress bars for time-consuming operations like VMAF analysis and the final encode.
* **System Monitoring:** Live RAM usage to ensure system stability during intensive encodes.
* **Disclaimer**: ETA is flawed in the GIF above as no database information was used for these encodings. The script starts showing accurate ETAs after completing ~10 encodes.
---

## 🚀 Getting Started

> **Note for Complete Beginners:**
> For a detailed, step-by-step guide with some screenshots covering everything from installing Python to configuring the script, please see the **[Complete User Manual.pdf](https://github.com/Snickrr/Auto-VMAF-Encoder/blob/main/Auto%20Vmaf%20Encoder%20Install%20Guide%2C%20Manual%20%26%20Technical%20Documentation%20V1.0.pdf)** included in this repository.
>
> The instructions below are a faster quick-start guide for users already familiar with Github, Python and command-line tools.

### Prerequisites

Before you begin, ensure you have the following installed and accessible:

* ✅ **Python 3.8+**: Make sure it's added to your system's PATH during installation. ([Download](https://python.org/downloads/))
* ✅ **FFmpeg**: The `ffmpeg` and `ffprobe` command-line tools. Full builds for Windows are available from [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/). It's recommended to unzip them to a simple, persistent location like `C:\ffmpeg`.

<details>
  <summary><strong>What is VMAF?</strong> (Click to expand)</summary>
  
  Video Multimethod Assessment Fusion (VMAF) is a perceptual video quality metric developed by Netflix. It uses a machine-learning model to predict subjective video quality more accurately than traditional metrics like PSNR or SSIM. It has become an industry standard for optimizing video encoding to ensure the best viewing experience for a given bandwidth. This script leverages VMAF to make intelligent, quality-based encoding decisions.
</details>

### Installation & Configuration

1.  **Clone the Repository**
    This is the main step. It will download the script, configuration settings (config.ini), and the **required VMAF model file** all at once.
    ```bash
    git clone https://github.com/Snickrr/Auto-VMAF-Encoder.git
    cd Auto-VMAF-Encoder
    ```

2.  **Install Python Libraries**
    Open a Command Prompt or Terminal in the project folder and run the following command:
    ```bash
    pip install rich psutil "scenedectect[opencv]"
    ```

3.  **Configure `config.ini`**
    Full configuration settings can be found as a PDF: Auto Vmaf Encoder Manual & Technical Documentation

---

## 🤝 Contributing

This project was created by someone with no prior coding experience, using AI assistance for advanced mathematics and coding implementation. The core ideas and extensive debugging/fine-tuning were done manually.

Contributions are welcome! Please feel free to:
* Report bugs or suggest features by opening an issue.
* Submit pull requests to improve the code.
* Improve the documentation.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments
* The **Netflix VMAF Team** for the incredible video quality assessment framework.
* The **FFmpeg Community** for the powerful multimedia toolkit.
* The **Open Source Community** for the excellent Python libraries used in this project.

<br>
<div align="center">

**Star this repository if it helped you encode better videos! ⭐**

</div>












