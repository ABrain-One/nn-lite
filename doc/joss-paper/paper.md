---
title: 'NN-Lite: Automated Benchmarking of PyTorch Models on Android Devices'
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
  - name: Faraz Kayani
    orcid: 0009-0000-7769-6016
    corresponding: true
  - name: Sarmad Kayani
    orcid: ???
  - name: Radu Timofte
    orcid: 0000-0002-1478-0402
affiliations:
  - name: Computer Vision Lab, CAIDAS & IFI, University of Würzburg, Germany
    ror: "00fbnyb24"
date: 28 September 2026
bibliography: paper.bib
---

# Summary

Several processors inside a modern smartphone can run neural networks: the central processing unit (CPU), the graphics processing unit (GPU) with increasing frequency, a neural processing unit (NPU). Performance and feasibility depend on the target device. Reliable assessment consists of direct execution on the target phone. Researchers train their models on desktop computers with PyTorch [@Paszke2019PyTorch], and manual deployment of each model to a phone becomes impractical beyond a few experiments.

NN-Lite automates this workflow. With an Android phone connected by USB, each PyTorch model is converted to the format supported by the Android runtimes, at full precision and at reduced precision. The converted file is transferred to the phone, where latency is measured on the CPU, GPU and NPU paths. One result is written per model, quantization setting and device. Models come from the user's own exported or checkpointed files or from the LEMUR model collection [@Goodarzi2025LEMUR], installed with NN-Lite. The tool is designed for unattended multi-day runs: it resumes after a disconnection or a restart, and failures are kept in the results instead of being silently excluded.

# Statement of need

Research on efficient deep learning — hardware aware architecture search, quantization, latency prediction, model compression — requires measurements across many architectures and real devices. Emulators and desktop hardware are limited substitutes. An emulator does not reproduce the vendor specific GPU and NPU software stack of a phone. On real devices the same model can succeed on one accelerator and be rejected by another, since some operators are not supported [@Ignatov2019AIBenchmark; @Almeida2021SmartAtWhatCost].

NN-Lite serves researchers who train models in PyTorch and need reproducible latency and feasibility measurements which are specific to a device and to an accelerator, for a single architecture as well as for a large model collection. An earlier version of this codebase was used for a measurement campaign, whose latency dataset is reported in @Din2026NNLite. That work describes the campaign and the dataset. The present paper describes instead the consolidated software release, version 1.0.0, and five changes which transformed the campaign code into a reusable tool, affecting its measurements or its accessibility.

**Models from arbitrary sources.** The pipeline operated before only on the model collection it was written for. Models are now accepted as files too. Programs exported with `torch.export` are self-contained and retain their input shape, while a Python module with a checkpoint that contains the weights or the saved model is accepted as well.

**Installation without manual setup.** The original campaign depended on nightly builds which have since been withdrawn from the Python Package Index, so its reproduction was prevented. Version 1.0.0 of NN-Lite depends only on released versions of its converter, its runtime and its quantizer, and is installed from the Python Package Index through a single console entry point. Linux and Apple Silicon are supported. The model collection is installed as a dependency and retrieved at the first use.

**Measurements attributable to the intended model.** Models are now built at their training resolution instead of a fixed size, so an architecture such as VisionTransformer, which expects inputs of 299×299, loads its weights and does not fail. Each transfer to the phone is verified, so a failed copy marks the model as failed and no stale file is timed.

**Quantization calibrated on representative data.** Integer export used before a representative set of synthetic random noise. Such a set misrepresents the activation statistics and can reduce accuracy: AirNet, for example, reached 79.1% top-1 accuracy on CIFAR-10 compared to 86.6%. The calibration uses real images now. These images are processed with the input transformation of the model itself.

**A tested programming interface.** Output parsing, error extraction and record construction are separated from the device control into documented importable modules, which need neither the deep learning frameworks nor the Android Debug Bridge. Unit tests cover the modules. The tests run in continuous integration and do not need a phone. Each device keeps a progress ledger of its own, so a fleet can be benchmarked in parallel.

# State of the field

*AI Benchmark* [@Ignatov2018AIBenchmark; @Ignatov2019AIBenchmark] and *MLPerf Mobile* [@Reddi2022MLPerfMobile] are established mobile benchmarks, but they execute fixed curated sets of reference models in order to compare *devices*. These benchmarks are not designed to ingest arbitrary research architectures at large scale. Latency predictors such as *nn-Meter* [@Zhang2021nnMeter] and *BRP-NAS* [@Dudziak2020BRPNAS], hardware aware search methods such as *MnasNet* [@Tan2019MnasNet] and *ProxylessNAS* [@Cai2019ProxylessNAS], and benchmarks such as *HW-NAS-Bench* [@Li2021HWNASBench] use measured latency as ground truth instead of providing the infrastructure to obtain it. The closest work in spirit is @Zhang2022MobileLibraries. 171 models across mobile deep learning libraries are benchmarked there, but the study is fixed and is not a tool for the models and devices of the researchers themselves. Compiler stacks such as *TVM* [@Chen2018TVM] tune individual models, while *Qualcomm AI Hub* [@QualcommAIHub] offers hosted profiling as a vendor specific service. The `benchmark_model` utility of LiteRT [@TFLiteBenchmarkTool] measures one already converted model per invocation, but no conversion, bookkeeping, fault tolerance or standardized result schema is provided by it.

NN-Lite builds on these components instead of replacing them. Conversion is done with LiteRT Torch [@LiteRTTorch], while the LiteRT runtime [@Abadi2016TensorFlow] and the official `benchmark_model` binary execute the models and the Android Debug Bridge [@AndroidADB] controls the device. The Android Neural Networks API (NNAPI) [@AndroidNNAPI] requests NPU execution. Its contribution is the orchestration layer, which integrates these components into a reproducible and fault tolerant workflow from PyTorch models to physical device measurements and structured records.

# Software design

![How NN-Lite benchmarks one model on a physical phone: (1) the model is read from a local file or from the LEMUR dataset [@Goodarzi2025LEMUR]; (2) the network is rebuilt on the workstation and converted to FP32 and INT8 LiteRT files; (3) the files are copied over USB and timed with `benchmark_model` on the CPU, GPU and NPU paths; and (4) one JSON record is written. Unattended multi-day runs are made possible by the safeguards at the bottom.\label{fig:architecture}](figures/architecture.png){ width=100% }

**Host-driven execution using standard Android tooling.** Google's prebuilt `benchmark_model` binary is transferred once to the phone by NN-Lite and invoked per model and per path, so root access and model specific signed applications are avoided. The measurement covers the model inference. The reported latency does not represent end to end application latency.

**Models are rebuilt rather than assumed to be portable.** A model taken from the collection consists of PyTorch code, hyperparameters and a checkpoint. NN-Lite reinstantiates the network and restores its weights. The input resolution comes from the preprocessing definition which belongs to the model itself. A model supplied as a file is self-describing when it was exported with `torch.export`, and otherwise it is reconstructed from its own module, so naming collisions among models saved under a shared class name are avoided. Either way an architecture enters the pipeline, and no configuration per model is needed.

**Every execution path is measured, and failures are data.** Each model is exported in FP32 and, when supported, in INT8 with full integer post training quantization [@Jacob2018Quantization]. Every path is selected explicitly and not delegated to the default of the runtime, so every latency is attributable to a known backend. If the quantization fails on an operator which is not supported, the FP32 model is measured anyway. When a path fails, its diagnostic output is recorded in the result and in the per device log, and a model whose paths all fail is retained and marked invalid instead of being dropped. Failures remain in the results. The failures document gaps in operator support and do not bias subsequent analyses.

**Fault-tolerant, resumable runs.** Long campaigns are interrupted by overheating, USB disconnections, memory pressure and failures of the host process. NN-Lite pauses between the models so that cooling is possible, restarts its process from time to time in order to release the memory which the conversion toolchain holds, and waits when a device is disconnected. Progress is recorded per device. A campaign resumes in this way without repeating the work which was already completed.

**One schema for every model source.** Results are written as one file per model, quantization setting and device, in the training statistics layout of the collection, so a run made from the installed package can be copied into a checkout and contributed. Each record contains the latency distribution over repeated runs for every path, the fastest path, and the input shape, memory state and device telemetry needed for interpretation. Models from local files use the same schema. Both sources are compared in this way, and the analysis code is not modified.

Two limitations have to be mentioned. First, NN-Lite reports wall clock latency under a fixed protocol and does not isolate the state of the frequency governor or the energy per operator. Second, the NPU execution is mediated by the NNAPI, whose behaviour depends on the installed runtime and on the drivers: NNAPI can fall back to CPU execution when the compilation or the execution on an accelerator fails, and the API was deprecated in Android 15 [@AndroidNNAPI]. The requested path and the runtime output are recorded by NN-Lite, which cannot guarantee that all operations ran on the intended accelerator.

# Research impact statement

NN-Lite is the on device measurement component of the LEMUR ecosystem [@Goodarzi2025LEMUR]. Inference latency measurements for more than 7,500 models in LEMUR 2 [@Uzun2026LEMUR2] were provided by its emulator execution path. The physical device path produced the delegate resolved latency dataset reported in @Din2026NNLite, which covers 586 architectures across five commodity phones from three chip vendors. These records are publicly available. They have been used as well to fit device specific latency models by symbolic regression [@Dhanani2026Symbolic] and to show that FLOPs predict latency poorly unless the backend is taken into account [@Din2026BMVC].

Version 1.0.0 is released on the Python Package Index, documented for external users, and covered by unit tests which run in continuous integration on three Python versions. We verified both model sources end to end on a phone. The verification included a model outside the collection (torchvision MobileNetV3), so other groups can benchmark their own models.

# AI usage disclosure

We designed and implemented the NN-Lite software. No generative AI assistance was used for it, while Claude Opus 5.5 (Anthropic) assisted with drafting the manuscript, with checking `paper.bib` and with creating \autoref{fig:architecture}. All the content produced with AI assistance was reviewed, edited and validated by the authors. We made all the final scientific and software decisions.

# Acknowledgements

This work was partially supported by the Alexander von Humboldt Foundation. We thank Saif U Din and Muhammad Ahsan Hussain, whose work on the measurement campaign reported in @Din2026NNLite shaped the requirements of this release. Thanks go to the LEMUR dataset contributors.

# References
