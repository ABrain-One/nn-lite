---
title: 'NN-Lite: Unattended benchmarking of PyTorch models on physical Android devices'
tags:
  - Python
  - Android
  - deep learning
  - edge AI
  - on-device inference
  - quantization
  - benchmarking
  - LiteRT
authors:
  - name: Dmitry Ignatov
    orcid: 0000-0002-6339-1200
    affiliation: 1
  - name: Faraz Kayani
    orcid: 0009-0000-7769-6016
    corresponding: true
    affiliation: 1
  - name: Radu Timofte
    orcid: 0000-0002-1478-0402
    affiliation: 1
affiliations:
  - name: Computer Vision Lab, CAIDAS & IFI, University of Würzburg, Germany
    index: 1
    ror: 00fbnyb24
date: 26 September 2026
bibliography: paper.bib
---

# Summary

Modern phones contain several different processors that can run artificial
intelligence models: the main processor (CPU), the graphics processor (GPU) and,
increasingly, a dedicated neural processing unit (NPU). How fast a model runs,
and whether it runs at all, depends on the phone and on which of these
processors is used, and the only reliable way to find out is to run the model on
the phone itself. Researchers, however, design and train their models on desktop
computers with the PyTorch library [@Paszke2019PyTorch], and moving each model
onto a phone and timing it by hand does not scale beyond a few experiments.

NN-Lite automates this. Connected to an ordinary Android phone by a USB cable,
it takes each neural network from the LEMUR model collection
[@Goodarzi2025LEMUR], converts it into the format Android understands, in both
full and reduced numerical precision, copies it to the phone, times it on the
CPU, GPU and NPU, records what kind of phone produced the numbers, and saves
everything back into LEMUR in one consistent format. It is built to run
unattended for days: it continues where it stopped if the cable is unplugged or
the computer restarts, and it records failures instead of silently skipping
them.

# Statement of need

Research on efficient deep learning, such as hardware-aware architecture
search, quantization, latency prediction and model compression, needs
measurements of *many* architectures on *many* real devices. Emulators and
desktop hardware are poor substitutes: an Android emulator does not expose a
phone's GPU and NPU drivers, and on real phones the same model can succeed on one
accelerator and fail on another because of missing operator support
[@Ignatov2019AIBenchmark; @Almeida2021SmartAtWhatCost]. Collecting such data by
hand means converting every model, handling conversion failures, pushing files
to the phone, invoking each accelerator, parsing its output and noting which
phone, operating system and memory state produced each number, for hundreds of
models per device.

NN-Lite began as a pipeline that converted models and executed them on Android
*emulators*, which verifies that a converted model loads and runs but cannot
measure real accelerators. It has since gained a physical-device engine that
measures on real phones. The pipeline and the latency dataset it produces are
described in @Din2026NNLite; this paper documents the software itself, with a
focus on the physical-device engine. It targets researchers who train models in PyTorch and need
reproducible, per-device and per-accelerator latency and feasibility data at
dataset scale, stored next to the models' accuracy and training records so that
the two can be analysed together.

# State of the field

Existing tools cover parts of this workflow. *AI Benchmark*
[@Ignatov2018AIBenchmark; @Ignatov2019AIBenchmark] and *MLPerf Mobile*
[@Reddi2022MLPerfMobile] are established mobile benchmarks, but they run a fixed,
curated set of reference models in order to compare *devices*; they are not
designed to ingest hundreds of arbitrary research architectures. Latency
predictors such as *nn-Meter* [@Zhang2021nnMeter] and hardware-aware NAS
benchmarks such as *HW-NAS-Bench* [@Li2021HWNASBench] consume measured latencies
as ground truth rather than producing them for new model families. Compiler
stacks such as *TVM* [@Chen2018TVM] optimise and tune individual models, and
*Qualcomm AI Hub* [@QualcommAIHub] profiles models on hosted hardware from a
single chip vendor through a proprietary service. The `benchmark_model` utility
of LiteRT [@TFLiteBenchmarkTool] times one already-converted model per call,
with no conversion, bookkeeping or result format.

We therefore built on these components rather than replacing them. NN-Lite
does not re-implement conversion, kernels or timing: it uses LiteRT Torch
[@LiteRTTorch] for conversion, TensorFlow Lite runtimes [@Abadi2016TensorFlow]
and the official `benchmark_model` binary for execution, the Android Neural
Networks API [@AndroidNNAPI] for NPU access and the Android Debug Bridge
[@AndroidADB] for device control. Its contribution is the layer none of these
provide: orchestration from a model collection to a physical phone, fault
tolerance for multi-day runs on consumer hardware, and a stable result schema
joined to the models' training records. This layer could not be contributed
upstream, because it belongs to neither the converter nor the benchmark tool; it
connects a model dataset to real devices.

# Software design

\autoref{fig:architecture} shows the engine. For every model, a Python process
on the workstation rebuilds and converts the network, then drives the phone
over `adb`; the phone only needs USB debugging enabled.

![How NN-Lite benchmarks one model on a physical phone: (1) the model is read from LEMUR; (2) it is rebuilt and converted to FP32 and INT8 LiteRT files on the workstation; (3) it is copied over USB and timed by `benchmark_model` on the CPU, the GPU and the NPU (through NNAPI), while the device is probed; (4) one JSON record is saved back into LEMUR. The loop repeats for the next model, and the safeguards at the bottom keep multi-day runs going without supervision.\label{fig:architecture}](figures/architecture.png){ width=100% }

The main design decisions and their trade-offs are:

**Host-driven execution with the vendor-neutral benchmark tool.** NN-Lite pushes
Google's prebuilt `benchmark_model` binary to the phone once and invokes it per
model and accelerator. This needs neither root access nor building and signing
an app per model, works on any Android phone, and produces timings comparable
with those of other LiteRT users. The cost is that application-level overhead
such as image decoding is not included, which is appropriate when the goal is to
compare architectures rather than applications.

**Models are rebuilt from source, not from exported graphs.** Each LEMUR model
is stored as PyTorch code, hyperparameters and a checkpoint. NN-Lite
re-instantiates the network, restores its weights and reads the input
resolution from the model's own preprocessing definition, so new architectures
enter the pipeline without any per-model configuration.

**Two precisions with graceful degradation.** Every model is exported in FP32
and, through full-integer post-training quantization, in INT8. If quantization
fails, the FP32 model is still measured and recorded, so one unsupported
operator never removes a model from the dataset.

**Failures are data.** Each accelerator is benchmarked independently. When one
fails, the relevant lines of the tool's output are extracted into the record
(for example `npu_error`) and into a per-device log; a model for which every
accelerator fails is stored with `valid: false` rather than dropped. This avoids
survivorship bias in later analyses and documents operator-support gaps, which
are themselves informative.

**Crash-only, resumable runs.** Consumer phones heat up, lose USB connections
and accumulate memory pressure during long runs. NN-Lite keeps a persistent list
of finished and failed models, pauses between models and between sessions so
the device can cool down, restarts its own process every 50 models to release
memory held by the conversion toolchain, and blocks until a disconnected device
reappears. A run can be stopped and restarted at any time without repeating
work.

**One schema shared with LEMUR.** Results are written as one JSON file per
model, precision and device, following the directory layout LEMUR uses for its
training statistics. Each record holds, per accelerator, the mean, minimum,
maximum and standard deviation of latency over repeated runs, the fastest
accelerator, the input dimensions, the memory state and device telemetry (chip,
CPU topology and OS build), plus an `emulator` flag. Because the emulator path
writes the same schema, emulator and physical-device results can be compared or
filtered without changing downstream code.

# Research impact statement

NN-Lite is the on-device measurement component of the LEMUR ecosystem
[@Goodarzi2025LEMUR]. Its emulator path was used to add on-device inference
latencies for more than 7,500 models to LEMUR 2 [@Uzun2026LEMUR2]. With the
physical-device engine described here, one workstation and five commodity phones
from three chip vendors (Qualcomm, MediaTek and HiSilicon) benchmarked 586 LEMUR
architectures in FP32 and INT8 on CPU, GPU and NNAPI, producing 5,826 per-device
records in about 56 unattended device-hours; this dataset is described in
@Din2026NNLite. These records are publicly available in the LEMUR
repository and can be retrieved through its API together with accuracy and
training metadata.

# AI usage disclosure

The NN-Lite physical-device benchmarking engine was designed and written by the
authors without generative AI assistance. Claude (Anthropic; model version:
[MODEL VERSION]), used through the Claude Code agent, assisted with drafting the
text of this paper, assembling and checking `paper.bib`, and drawing
\autoref{fig:architecture}. The authors reviewed, edited and validated all
AI-assisted content, verified every reference, and made all design and
architectural decisions for the software.

# Acknowledgements

This work was partially supported by the Alexander von Humboldt Foundation. We
thank the contributors to the LEMUR dataset and to the emulator-based pipeline
of NN-Lite [@Din2026NNLite].

# References
