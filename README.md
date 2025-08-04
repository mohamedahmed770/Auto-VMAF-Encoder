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

### Prerequisites

* Python 3.8+
* FFmpeg (must be compiled with `libvmaf` support)
* **Optional (Recommended):** `opencv-python` for advanced scene detection.

### Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install Dependencies**
    A `requirements.txt` file is provided. Install all required Python packages with:
    ```bash
    pip install -r requirements.txt
    ```
    For the optional (but recommended) advanced scene detection:
    ```bash
    pip install scenedetect[opencv]
    ```

3.  **Configure the Script**
    * Make a copy of `config.ini.example` and rename it to `config.ini`.
    * Edit `config.ini` with a text editor.
    * **Required:** You must set the correct paths to your `ffmpeg` executable, `ffprobe` executable, and the VMAF model file (`.json`).

### Running the Script

Once configured, run the encoder from your terminal:
```bash
python auto_vmaf_encoder.py
```

---

## ⚙️ Configuration

All settings are controlled via the `config.ini` file.

#### Essential Settings
```ini
[Paths]
ffmpeg_path = /path/to/your/ffmpeg
ffprobe_path = /path/to/your/ffprobe
vmaf_model_path = /path/to/your/vmaf_v0.6.1.json

[VMAF_Targeting]
target_vmaf = 95.0
vmaf_tolerance = 1.0
cq_search_min = 15
cq_search_max = 35
```

#### Advanced Options
* **Encoder Selection:** Choose between `nvenc` and `svt_av1`.
* **Sampling Methods:** `tier0` (PySceneDetect), `tier1` (keyframes), or `tier2` (intervals).
* **Performance Tuning:** Worker counts, memory limits, parallel VMAF runs.
* **File Filtering & Management:** Set thresholds for duration, size, and bitrate, and control output naming and source file deletion.

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
