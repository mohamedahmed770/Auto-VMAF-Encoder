Auto-VMAF Encoder

An advanced Python script for video encoding that uses a data-driven approach to determine optimal encoding parameters. The script finds the perfect CQ/CRF (Constant Quality Factor / Constant Rate Factor) value to achieve a target video quality, as measured by a predefined VMAF (Video Multimethod Assessment Fusion) score.



![Python](https://img.shields.io/static/v1?label=Python&message=v3.8%2B&color=blue)
![License](https://img.shields.io/static/v1?label=License&message=MIT&color=green)
![Platform](https://img.shields.io/static/v1?label=Platform&message=Windows&color=blue)



🖥️ Console Interface

![Demo GIF](images/DemoUncached.gif)  

<div align="center">

# AUTO VMAF ENCODER

**A data-driven encoding tool that uses VMAF-based quality targeting and performance analysis to create high-quality, efficient video encodes.**

</div>

![License](https://img.shields.io/badge/license-MIT-green)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

This script automates the complex process of video encoding by intelligently analyzing video files, finding the optimal quality settings to meet a target VMAF score, and providing a real-time console UI to monitor the progress.

---

## 🎯 Key Features

* **🧠 Smart Encoding**
    * **Intelligent Sampling:** Analyzes video samples using multiple methods (PySceneDetect, keyframes, intervals) to assess media complexity.
    * **VMAF-Targeted Quality:** Uses a binary search algorithm to find the optimal CQ/CRF value that achieves your target VMAF score with precision.
    * **Complexity-Aware Processing:** Automatically adjusts parameters based on video complexity analysis.

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
    * **Memory Management:** Intelligently monitors memory usage to prevent system crashes.

* **📂 Intelligent File Management**
    * **Configurable Filtering:** Skip files based on duration, filesize, or bitrate.
    * **Size Reduction Validation:** Only replaces the source file if the encode is meaningfully smaller.
    * **Flexible I/O:** Configurable input/output directories and file naming schemes.

---

## 🚀 Getting Started

This guide will walk you through setting up the script and its dependencies.

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
    This is the main step. It will download the script, the banner font, the configuration examples, and the **required VMAF model file** all at once.
    ```bash
    git clone [https://github.com/Snickrr/Auto-VMAF-Encoder.git](https://github.com/Snickrr/Auto-VMAF-Encoder.git)
    cd Auto-VMAF-Encoder
    ```

2.  **Install Python Libraries**
    Open a Command Prompt or Terminal in the project folder and run the following command:
    ```bash
    pip install rich psutil "scenedectect[opencv]"
    ```

3.  **Configure `config.ini`**
    This is the final and most important step.
    * Make a copy of `config.ini.example` and rename it to `config.ini`.
    * Open `config.ini` with a text editor.
    * Update the `[Paths]` section to match the locations of your files. The VMAF model path will now point to the file inside your project folder.

    > **💡 Pro-Tip:** To avoid potential issues, it's a good practice to rename your `vendor (VMAF MODEL)` folder to something simple without spaces, like `vendor` or `assets`.

    ```ini
    [Paths]
    ffmpeg_path = C:/ffmpeg/bin/ffmpeg.exe
    ffprobe_path = C:/ffmpeg/bin/ffprobe.exe
    vmaf_model_path = vendor (VMAF MODEL)/vmaf_v0.6.1.json
    ```

After saving `config.ini`, your setup is complete! You can now run the script.

---

## 🤝 Contributing

This project was created by a developer with no prior coding experience, using AI assistance for advanced mathematics and coding implementation. The core ideas and extensive debugging/fine-tuning were done manually.

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

