[![Release badge](https://img.shields.io/badge/Auto-VMAF-Encoder-Release-green?logo=github&style=for-the-badge)](https://github.com/mohamedahmed770/Auto-VMAF-Encoder/releases)

# Auto-VMAF-Encoder: Intelligent VMAF-Targeted Video Encoding for Quality Optimization and Speed

⚙️ A modern Python tool for high-quality video encoding. It targets VMAF-driven quality, uses smart sampling and caching to avoid wasted work, and supports NVENC and SVT-AV1 for fast, hardware-accelerated or software-based encoding. Real-time progress updates help you stay informed during long runs.

![Video processing](https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Video_processing_icon.svg/1024px-Video_processing_icon.svg.png)

Table of contents
- Why this project exists
- Core ideas and design goals
- Quick start
- How it works under the hood
- CLI reference
- Encoding workflows
- Performance, caching, and quality
- Configuration and tuning
- Troubleshooting
- Roadmap
- Testing and quality assurance
- Security and reliability
- Internationalization and accessibility
- Contributing
- Licensing
- Releases and upgrades

Why this project exists
Video encoding is complex. You want speed, but not at the expense of quality. You also want predictable results across many inputs. Auto-VMAF-Encoder exists to bridge those needs. It combines targeted quality optimization with intelligent sampling and caching to minimize work while preserving or improving video quality as measured by VMAF. It supports both NVENC for hardware-accelerated encoding and SVT-AV1 for software-based encoding, so you can tailor your pipeline to your hardware and licensing constraints.

If you already know you want the latest release, you can download the release asset from the project releases page. You can access the latest release here: https://github.com/mohamedahmed770/Auto-VMAF-Encoder/releases. This link has a path part, so the file you need to download is the release asset and you should run that file after download to install or run Auto-VMAF-Encoder. For convenience, a colorful badge is shown above that links to the same page.

Core ideas and design goals
- Quality-first by design: Use VMAF as the quality target, not a proxy metric. The encoder adapts its behavior to approach the VMAF target across scenes.
- Intelligent sampling: Rather than encoding every frame at full fidelity, sample frames strategically to estimate a scene’s quality requirements.
- Smart caching: Cache expensive computations and intermediate results to avoid repeating work when inputs or parameters repeat.
- Flexible encoding backends: Support both NVENC (GPU-accelerated) and SVT-AV1 (CPU or GPU-optional) to cover a wide range of hardware setups.
- Real-time visibility: Show progress, estimated completion time, and key stats during long transcoding runs.
- Extensible, practical CLI: A focused command-line interface that is easy to script and automate, with sane defaults.
- Safe defaults: Reasonable defaults that work well for most workflows, with options to fine-tune for advanced users.
- Observability: Clear logs, structured output, and optional JSON results for downstream pipelines.

What you can do with Auto-VMAF-Encoder
- Transcode video streams or files with a quality target driven by VMAF.
- Use intelligent sampling to minimize work without sacrificing quality.
- Cache repeated tasks to speed up batch processing and iterative runs.
- Choose between NVENC and SVT-AV1 encoding paths based on hardware and licensing.
- Monitor progress in real time, including frame-level quality hints and ETA.
- Integrate into CI/CD pipelines or automation scripts with a clean CLI.

Core components and how they fit together
- Quality estimator: A VMAF-driven quality model that defines the target quality per segment or scene.
- Sampler: A module that decides which frames or segments to encode at high fidelity and which to approximate.
- Cache layer: A store for intermediate results, flags, and precomputed metrics to speed up repeated jobs.
- Encoder backends: NVENC for hardware-accelerated encoding and SVT-AV1 for software-based encoding. The tool selects the best path based on input, target, and environment.
- Orchestrator: Coordinates sampling, encoding, caching, and progress reporting to deliver a final file meeting the target quality.

Quick start
Prerequisites
- Python 3.8 or newer (ideally 3.9+ for best dependency compatibility)
- FFmpeg with NVENC support installed on your system (for NVENC paths)
- SVT-AV1 binary available in your PATH if you plan to use SVT-AV1 encoding
- Access to a capable GPU if you want hardware acceleration via NVENC (recommended for large projects)

Installation
- Ensure your environment is clean and has Python and FFmpeg installed.
- Create a virtual environment to avoid dependency conflicts:
  - python3 -m venv env
  - source env/bin/activate (Linux/macOS) or .\\env\\Scripts\\activate (Windows)
- Install the package from source or via pip if you have a published wheel:
  - pip install auto-vmaf-encoder
  - Or install from source after cloning the repository:
    - pip install -r requirements.txt
    - python setup.py install
- Verify the installation:
  - auto-vmaf-encoder --version
  - The CLI should report the current version and a short summary of capabilities.

Running a basic encode
- Basic command (NVENC path):
  - auto-vmaf-encoder --input sample.mp4 --output sample_nvenc.mp4 --codec nvenc --target-vmaf 92
- Basic command (SVT-AV1 path):
  - auto-vmaf-encoder --input sample.mp4 --output sample_svta1.mp4 --codec svt-av1 --target-vmaf 92
- Optional flags:
  - --preset: choose a preset for speed vs. quality
  - --threads: limit CPU threads for SVT-AV1
  - --cache-dir: specify a custom cache location
  - --sampling-rate: adjust how many frames are sampled for VMAF estimation
  - --progress: enable live progress reporting in the terminal

How it works under the hood
- Phase 1: Pre-analysis and sampling
  - The tool analyzes the input to identify scenes where quality changes are likely to be significant.
  - A sampling plan is generated to collect VMAF signals across representative frames.
- Phase 2: Target setting and credentialing
  - The VMAF target is mapped to per-segment quality budgets. This allows the encoder to allocate bits where they matter most.
- Phase 3: Encoding with caching
  - The chosen encoding path (NVENC or SVT-AV1) runs with the configured target. The system caches results to speed up subsequent runs on similar inputs.
- Phase 4: Monitoring and feedback
  - Real-time progress is shown. ETA estimates update as the encoding proceeds.
- Phase 5: Validation and output
  - After encoding, a quick validation pass checks that the produced file meets the target. If not, adjustments are suggested and the run can be retried with tuned parameters.

CLI reference
- Global options
  - --input: path to the input video
  - --output: path for the output video
  - --codec: nvenc or svt-av1
  - --target-vmaf: the VMAF target value (e.g., 92)
  - --preset: quality vs speed preset (e.g., fast, balanced, thorough)
  - --threads: number of CPU threads to use
  - --cache-dir: directory to store cache data
  - --sampling-rate: fraction of frames to sample (e.g., 0.25 for 25%)
  - --progress: show live progress in the terminal
  - --log-level: debug, info, warning, error
  - --dry-run: estimate results without encoding
- Examples
  - auto-vmaf-encoder --input input.mov --output output_nvenc.mov --codec nvenc --target-vmaf 93 --preset thorough --progress
  - auto-vmaf-encoder --input input.mkv --output output_svta1.mkv --codec svt-av1 --target-vmaf 90 --threads 8 --progress
  - auto-vmaf-encoder --input input.mp4 --output sample_cache_test.mp4 --codec nvenc --target-vmaf 90 --cache-dir /tmp/vmaf-cache --sampling-rate 0.2 --progress
- JSON output (optional)
  - --json-out: path to write a structured JSON report with per-segment metrics and final results
- Help and version
  - auto-vmaf-encoder --help
  - auto-vmaf-encoder --version

Encoding workflows
- Quick single-file workflow
  - This is a straightforward transcode with a fixed VMAF target.
- Batch workflow with caching
  - Process a folder of videos, reuse cache across runs to accelerate results on similar content.
- Adaptive workflow
  - The tool dynamically adjusts the VMAF target per scene based on detected content complexity. This yields better quality for hard-to-compress scenes and saves bits on simple scenes.
- Hybrid workflows
  - Combine NVENC for long, high-res segments with SVT-AV1 for precise quality control on small or complex segments.
- Streaming compatibility
  - The tool can be integrated into live pipelines that receive data in chunks and produce output with consistent quality targets.

Performance, caching, and quality
- Quality targets and feasibility
  - A higher VMAF target improves perceived quality but may require more bits or a slower path. The tool balances this by sampling and caching.
- Sampling strategy
  - Sampling reduces the amount of work needed to estimate quality. It uses representative frames from each scene to project quality across the segment.
- Caching strategy
  - Caches include: frame-level VMAF estimates, encoding parameterizations, and partial outputs where safe. Reuse reduces computation time considerably on similar inputs or when rerunning with slightly different targets.
- Memory and storage
  - The cache directory can grow large for long videos or many projects. Set a reasonable limit or clear the cache between large jobs if needed.
- Hardware considerations
  - NVENC is fastest on supported GPUs. SVT-AV1 provides strong quality for CPU-only environments and can be tuned for speed with presets.
- Output validation
  - The final file is checked against the VMAF target where feasible. If the target isn’t met due to input peculiarities, the tool suggests adjustments or re-tries with modified parameters.

Configuration and tuning
- Global tuning knobs
  - Target VMAF: The core knob to control perceived quality.
  - Sampling rate: Too high increases compute time; too low risks quality loss.
  - Cache behavior: Turn on caching to speed up repeated runs.
  - Encoding path: Choose NVENC for speed or SVT-AV1 for control and compatibility.
- Scene-level tuning
  - In scenes with little motion or flat lighting, you can lower the target slightly to save bits without noticeable quality loss.
  - In action-packed scenes, you may need a higher target to preserve detail.
- Resource limits
  - Constrain CPU threads for SVT-AV1 to prevent CPU contention on shared systems.
  - Reserve a GPU or allocate memory for NVENC tasks if running alongside other GPU-intensive workloads.
- Output formats
  - Prefer MP4 with H.264/AVC or H.265/HEVC for broad compatibility, or MKV with SVT-AV1 for future-proof quality.

Troubleshooting
- Common issues
  - Input file not found: Verify the path and permissions.
  - Encoder not found: Ensure the selected encoding path (nvenc or svt-av1) is installed and in PATH.
  - FFmpeg not found or incompatible: Install a compatible FFmpeg version with required codecs and enable GPU features if needed.
  - Cache directory permissions: Ensure the cache path is writable.
- Logging and diagnostics
  - Use --log-level debug to get verbose output.
  - Use --json-out to export a structured report for automation and debugging.
- Reproducing issues
  - Reproduce with --dry-run to verify configuration without encoding.
  - Record samples of source and target scenes to compare with expected VMAF values.
- Contact and support
  - Open an issue on GitHub with a clear description, steps to reproduce, and include logs and sample inputs if possible.

Roadmap
- Further YAGNI improvements
  - More accurate VMAF-target mapping across diverse content.
  - Improved caching strategies and cache eviction policies.
- Expanded format support
  - Additional backends and containerized deployment options.
- Enhanced observability
  - Richer in-terminal dashboards and exportable metrics.
- Better integration
  - Plugins for workflow managers, CI/CD, and cloud-based batch processing.

Testing and quality assurance
- Unit tests
  - Tests cover parsing, parameter validation, sampling logic, and some encoding path mocks.
- Integration tests
  - End-to-end tests run known inputs through both NVENC and SVT-AV1 paths.
- Performance benchmarks
  - Regression tests compare speed and quality against previous releases on representative hardware.
- CI/CD
  - Automated tests run on pull requests to ensure compatibility with target Python versions and major OS families.

Security and reliability
- Dependency hygiene
  - Use pinned dependencies to minimize supply-chain risk.
- Safe defaults
  - Defaults steer users toward safe, reliable configurations that work broadly.
- Input validation
  - Rigorously validate file paths, formats, and option values to avoid crashes.
- Access control
  - If used in multi-user environments, ensure proper permissions on input/output paths and cache directories.

Internationalization and accessibility
- Language support
  - Messages and logs are in English, with a plan to add translations via community contributions.
- Accessibility
  - CLI outputs include descriptive messages and color cues with sensible defaults for readability.

Images and visual assets
- Figures and diagrams
  - Diagrams illustrate the pipeline: sampling, caching, and the dual-encoder path.
- Icons
  - Simple icons for modules (sampling, caching, encoding) help quick navigation in the docs.
- Example charts and heatmaps
  - Visualizations show sampling coverage and VMAF attainment across scenes for advanced users.

Contributing
- How to contribute
  - Fork the repository.
  - Create a feature branch with a descriptive name.
  - Write tests for new features.
  - Run the test suite locally and ensure it passes.
  - Open a pull request with a concise description of the change.
- Code style
  - Follow the project’s style guide, keep changes small, and document non-obvious behavior.
- Documentation
  - Add or update docs and examples for any user-facing changes.

Licensing
- The project is released under a permissive license. Review the LICENSE file in the repository for exact terms and conditions.

Releases and upgrades
- Where to get the latest release
  - The latest release assets are hosted on GitHub Releases. Use the same link shown at the top of this document to access the Releases page and download the appropriate release asset. For convenience, the link is repeated here: https://github.com/mohamedahmed770/Auto-VMAF-Encoder/releases
- How to upgrade
  - If you upgrade, remove the old environment or deactivate the old virtual environment, then install the new release artifacts.
  - Re-run your typical workflows and verify that the VMAF targets are met as expected.

Credits
- Design and implementation
  - The project draws on established concepts in video encoding, VMAF-driven quality assessment, and modern CLI tooling.
- Community and testing
  - Thanks to early adopters, contributors, and testers who helped shape the project through feedback and patches.
- External dependencies
  - The encoder relies on FFmpeg, VMAF tooling, and open-source components that enable the feature set.

Release notes
- Upcoming notes
  - This section will document the changes in each release, including new features, performance improvements, and bug fixes.
- How to read
  - Each entry includes a short description, the impact on users, and compatibility notes.

Appendix: Getting the most out of Auto-VMAF-Encoder
- Real-world workflows
  - Batch processing of large video libraries with consistent quality targets.
  - Studio pipelines requiring predictable output quality and reproducible results.
  - Quick-turnaround encoding for content platforms that demand fast delivery and reliable quality.
- Best practices
  - Start with moderate targets and refine through iterations.
  - Use caching when running multiple passes or re-annotating similar content.
  - Run dry-runs to validate configuration before committing to long encoding jobs.
- Common pitfalls
  - Mismatched targets across scenes can lead to perceived inconsistency.
  - Overreliance on a single encoding path may miss hardware capabilities.
  - Not accounting for containerized environments where GPU access is restricted.

Demo and illustrations
- A sample pipeline diagram
  - A schematic showing input -> sampling -> VMAF target mapping -> encoding (NVENC or SVT-AV1) -> cache -> output
- A sample progress log
  - A snippet of the live progress view to illustrate ETA, bitrate, and VMAF estimates

Appendix: Release link and download nudges
- First access
  - The main releases page is hosted at https://github.com/mohamedahmed770/Auto-VMAF-Encoder/releases. This page lists all available artifacts, including executables and wheels for different platforms.
- Second access (repeat)
  - For convenient navigation, you can re-open the same link to view the latest release details at any time: https://github.com/mohamedahmed770/Auto-VMAF-Encoder/releases
- Download and run guidance
  - Since the link includes a path, download the release asset that matches your platform (e.g., a Windows exe, macOS app, or Linux tarball). After downloading, execute the file according to your platform’s guidelines. This will install or run Auto-VMAF-Encoder and set up the necessary runtime components.

Note on tone and style
- Direct and practical
  - The guide avoids fluff. It uses clear language and concrete steps.
- Active voice
  - Commands and procedures are stated in the imperative mood.
- Plain English
  - Terms are easy to understand. When necessary, a short technical term is defined.
- Non-salesy
  - The tone is calm and confident, not promotional.

If you want any part expanded or a different emphasis (for example, more beginner-friendly onboarding, or more advanced optimization guidance), I can tailor the README further to fit your exact audience and use cases.