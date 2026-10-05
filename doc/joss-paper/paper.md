---
title: "NN-Lite: Automated Benchmarking of PyTorch Models on Android Devices"
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
  - name: Sarmad Kayani
    orcid: 0009-0003-4156-4291
    affiliation: 1
  - name: Radu Timofte
    orcid: 0000-0002-1478-0402
    affiliation: 1
affiliations:
  - name: Computer Vision Lab, CAIDAS & IFI, University of Würzburg, Germany
    index: 1
    ror: 00fbnyb24
date: 6 October 2026
bibliography: paper.bib
---

# Summary

A modern smartphone contains several processors that can run neural networks: the central processing unit (CPU), the graphics processing unit (GPU) and often a neural processing unit (NPU). Manual deployment does not scale. Researchers train their models with PyTorch [@Paszke2019PyTorch], and transferring each model to a phone by hand becomes impractical when many models must be measured.

NN-Lite automates the deployment and measurement of user-supplied PyTorch vision models that satisfy the documented input and conversion requirements. With an Android phone connected by USB, the following procedure is used for each model: conversion to LiteRT at full (FP32) and at reduced (INT8) precision is attempted and can fail. The converted file is then copied to the phone. On the phone, latency is measured for the supported configurations of CPU, GPU-delegate and Android Neural Networks API (NNAPI) execution. These configurations are requested explicitly and recorded with their outcomes. The labels CPU, GPU-delegate and NNAPI do not mean that every operator ran on a particular physical accelerator, only that the labelled configuration was requested. Models come from the user's own files or from the LEMUR model collection [@Goodarzi2025LEMUR] installed with NN-Lite. A multi-day unattended run resumes after a disconnection or a restart. Failures are kept in the results and are not excluded silently. We describe a tested and reusable way to conduct and interpret such measurement campaigns, with explicit limits on what the measurements establish.

# Statement of need

Research on efficient deep learning (hardware aware architecture search, quantization, latency prediction, model compression) requires measurements across many architectures and real devices, because an emulator does not reproduce the vendor specific GPU and NPU software stack of a phone. Some operators are not supported [@Ignatov2019AIBenchmark; @Almeida2021SmartAtWhatCost]. On a real device, the same model can succeed with one accelerator and be rejected by another.

NN-Lite, which evolved from the pipeline that produced the latency dataset of @Din2026NNLite, serves researchers who need reproducible latency and feasibility measurements, specific to a device and to an accelerator configuration. The present paper documents the maintained release 1.0.1 with its interfaces, design choices and tests, as well as the changes with respect to the campaign pipeline. A full list of the changes is contained in the changelog. Five of them matter most.

**User-supplied vision models.** The campaign pipeline operated only on the collection it was written for. A file is now also accepted: a program exported with `torch.export`, which retains its input shape, or a Python module with a checkpoint or the saved model. Each model must have a single four-dimensional (NCHW) image input. Because of the calibration images needed for INT8 export and the default normalization applied if the user supplies no transform, the correct preprocessing remains the user's responsibility.

**Packaged installation.** The dependence on nightly builds that were later withdrawn from the Python Package Index made the campaign impossible to reproduce. Release 1.0.1 fixes the exact versions of its dependencies, all of which are published on the Python Package Index, is installed under the name `nn-lit`, and its console command is `nn-lite-bench`. Installing the package does not install the Android Debug Bridge (ADB) or authorize a phone. The user does both. Linux and Apple Silicon are supported.

**Measurements attributable to the intended model.** Models are now built at their training resolution, not at a fixed size. At this resolution, the VisionTransformer configuration of the collection loads its weights correctly, because the checkpoint of this configuration, rather than ViT models in general, expects 299×299 inputs. Verifying each transfer to the phone keeps a stale file from being timed. CPU timing uses the built-in LiteRT kernels with XNNPACK disabled (`--use_xnnpack=false`), as in the campaign.

**Quantization calibrated on representative data.** Integer export in the campaign pipeline used a set of synthetic random noise. Such a set misrepresents the activation statistics and can reduce the accuracy. The calibration now uses real images, processed with the input transformation of the model itself.

**A tested programming interface.** Output parsing, error extraction and record construction are separated from the device control into documented importable modules. These modules need neither the deep learning frameworks nor ADB. An explicit serial number in every device command allows one process per phone to run at the same time. But because progress and result files are keyed by the phone model, two physical phones of the same model share them, and a second concurrent run for that model is rejected.

# State of the field

*AI Benchmark* [@Ignatov2018AIBenchmark; @Ignatov2019AIBenchmark] and *MLPerf Mobile* [@Reddi2022MLPerfMobile] execute fixed curated sets of reference models in order to compare *devices*. Measured latency serves as ground truth for latency predictors such as *nn-Meter* [@Zhang2021nnMeter] and *BRP-NAS* [@Dudziak2020BRPNAS], for search methods such as *MnasNet* [@Tan2019MnasNet] and *ProxylessNAS* [@Cai2019ProxylessNAS], and for benchmarks such as *HW-NAS-Bench* [@Li2021HWNASBench]. *nn-Meter* includes backend preparation for Android profiling, while the other works do not provide the measurement infrastructure. The closest work in spirit is @Zhang2022MobileLibraries, where 15 models are benchmarked with six mobile deep learning libraries on ten devices. That study, however, covers a fixed set of models and is not a tool that researchers can apply to their own models and devices. Compiler stacks like *TVM* [@Chen2018TVM] tune individual models. *Qualcomm AI Hub* [@QualcommAIHub] offers hosted profiling, but only on devices with Qualcomm chips. Compared to *ExecuTorch* [@ExecuTorch], which provides PyTorch edge deployment workflows, NN-Lite orchestrates LiteRT campaigns over many models and devices. The `benchmark_model` utility of LiteRT [@TFLiteBenchmarkTool] measures one converted model per invocation and supports structured profiling output, but it does not provide conversion, failure bookkeeping or one record for many models and devices.

NN-Lite uses existing components and does not replace them: LiteRT Torch [@LiteRTTorch] for conversion, the LiteRT runtime [@Abadi2016TensorFlow; @LiteRTDocs] with the official `benchmark_model` binary for execution, and ADB [@AndroidADB] for device control. NNAPI [@AndroidNNAPI] is used to request accelerator execution. The contribution of NN-Lite consists in the orchestration layer, which integrates the components into a reproducible and fault tolerant workflow from PyTorch models to structured records.

# Software design

![How NN-Lite benchmarks one model on a physical phone: (1) the model is read from a local file or from the LEMUR dataset [@Goodarzi2025LEMUR]; (2) the network is rebuilt on the workstation and converted to FP32 and INT8 LiteRT files; (3) the files are copied over USB and timed with benchmark_model in the CPU, GPU-delegate and NNAPI configurations, which are requested and not verified on the hardware; and (4) one file per model, precision and device is written, with an optional contribution to LEMUR. The safeguards at the bottom allow the unattended multi-day runs.\label{fig:architecture}](figures/architecture.png){ width=100% }

**Host-driven execution using standard Android tooling.** Transferring Google's prebuilt `benchmark_model` binary once to the phone and invoking it per model and configuration avoids root access and signed applications. Android tends to run processes started over ADB on the slower cores of a phone, so `benchmark_model` is pinned with `taskset` to all cores except the slowest group. End to end application latency is not reported.

**Models are rebuilt rather than assumed to be portable.** A model taken from the collection consists of PyTorch code, hyperparameters and a checkpoint. NN-Lite reinstantiates the network and restores its weights. A model from a file is self-describing in case of `torch.export`. For every other model from a file, reconstruction from its own module avoids naming collisions under a shared class name.

**Every configuration is measured, and failures are data.** Each model is exported in FP32. INT8 export with full integer post training quantization [@Jacob2018Quantization] follows for every model that supports it. The quantization can fail on an operator which is not supported. Still, the FP32 model is measured. Requesting every configuration explicitly, and not delegating it to the default of the runtime, means that each latency is labelled with the requested backend. Recording the diagnostic output of a failed configuration in the result and in the per device log, and retaining a model whose configurations all fail and marking it invalid, makes failures available for explicit analysis. Not every failure establishes an operator support gap.

**Fault-tolerant, resumable runs.** NN-Lite pauses between models, restarts its process from time to time to release the memory held by the conversion toolchain, with a longer pause for cooling at each restart, and waits for a disconnected device.

**One schema for every model source.** Results are written as one file per model, quantization setting and device, in the training statistics layout of the collection. Contributing a run made from the installed package to LEMUR is optional. Each record contains the mean, minimum, maximum and standard deviation of the latency over repeated executions for every configuration, the fastest configuration (the one with the lowest valid mean latency), and the input shape, memory state and device telemetry needed for interpretation. The requested number of timed runs (20 by default) is stored as `iterations`. Each configuration performs exactly this number of runs after a warm-up, and the statistics are computed from these runs only. The analysis code stays unmodified.

Two limitations have to be mentioned. NN-Lite reports wall clock latency and does not isolate the frequency governor state or the energy per operator. The second limitation is that NPU execution is mediated by NNAPI, whose behaviour depends on the installed runtime and drivers: NNAPI can fall back to CPU execution when compilation or execution on an accelerator fails, and the API was deprecated in Android 15 [@AndroidNNAPI]. Because NN-Lite cannot guarantee that all operations ran on the intended accelerator, the label NPU in the schema denotes the NNAPI configuration. The command line of each configuration is built in a single function, and this function localizes most of the change for a replacement backend.

# Research impact statement

NN-Lite is the on device measurement component of the LEMUR ecosystem [@Goodarzi2025LEMUR]. Inference latency measurements for more than 7,500 models in LEMUR 2 [@Uzun2026LEMUR2] were provided by the emulator execution path of NN-Lite. The physical device path produced the delegate resolved latency dataset reported in @Din2026NNLite, which covers 586 architectures across five commodity phones from three chip vendors. All records of this dataset are publicly available. They have also been used to fit device specific latency models by symbolic regression [@Dhanani2026Symbolic] and to show that FLOPs predict latency poorly if the backend is not taken into account [@Din2026BMVC].

Release 1.0.1 is on the Python Package Index as `nn-lit`, with the source in the project repository. Unit tests cover the release and run in continuous integration on three Python versions. We verified both model sources end to end on a phone, including a model outside the collection (torchvision MobileNetV3).

# AI usage disclosure

We designed and implemented the NN-Lite software without generative AI assistance, while Claude Opus 5.5 (Anthropic) assisted with drafting the manuscript, checking `paper.bib` and creating \autoref{fig:architecture}. All the content produced with AI assistance was reviewed, edited and validated by the authors, who made all the final scientific and software decisions.

# Acknowledgements

This work was partially supported by the Alexander von Humboldt Foundation. We thank Saif U Din and Muhammad Ahsan Hussain for the measurement campaign of @Din2026NNLite, and the contributors to the LEMUR dataset.

# References