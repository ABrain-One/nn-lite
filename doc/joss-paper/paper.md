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
    affiliation: '1'
  - name: Faraz Kayani
    orcid: 0009-0000-7769-6016
    corresponding: true
    affiliation: '1'
  - name: Sarmad Kayani
    orcid: 0009-0003-4156-4291
    affiliation: '1'
  - name: Radu Timofte
    orcid: 0000-0002-1478-0402
    affiliation: '1'
affiliations:
  - name: Computer Vision Lab, CAIDAS & IFI, University of Würzburg, Germany
    ror: "00fbnyb24"
date: 28 September 2026
bibliography: paper.bib
---

# Summary

Several processors inside a modern smartphone can run neural networks — the central processing unit (CPU), the graphics processing unit (GPU), and with increasing frequency a neural processing unit (NPU). Manual deployment does not scale. Researchers train their models on desktop computers with PyTorch [@Paszke2019PyTorch], and the transfer of each model to a phone by hand becomes impractical once many models must be measured.

NN-Lite automates this workflow. With an Android phone connected by USB, each PyTorch model is converted to the format which the Android runtimes support, at full precision and at reduced precision, and the converted file is transferred to the phone, where the latency is measured on the CPU, GPU and NPU. Models come from the user's own exported or checkpointed files or from the LEMUR model collection [@Goodarzi2025LEMUR], installed with NN-Lite. Unattended multi-day runs are the design target: a run resumes after a disconnection or a restart, and failures are kept in the results instead of being excluded silently.

# Statement of need

Research on efficient deep learning — hardware aware architecture search, quantization, latency prediction, model compression — requires measurements across many architectures and real devices, while an emulator does not reproduce the vendor specific GPU and NPU software stack of a phone. Some operators are not supported [@Ignatov2019AIBenchmark; @Almeida2021SmartAtWhatCost]. On a real device the same model can succeed on one accelerator and be rejected by another.

NN-Lite serves researchers who train their models in PyTorch and need reproducible latency and feasibility measurements which are specific to a device and to an accelerator, for a single architecture as well as for a large model collection. An earlier version of this codebase was used for a measurement campaign. Its latency dataset is reported in @Din2026NNLite. We describe here the consolidated software release, version 1.0.0, and five changes which transformed the campaign code into a reusable tool, with effect on its measurements and on its accessibility.

**Models from arbitrary sources.** The pipeline operated before only on the model collection it was written for. Files are accepted now too. A program exported with `torch.export` is self-contained and retains its input shape, while a Python module with a checkpoint which contains the weights, or the saved model, is accepted as well.

**Installation without manual setup.** The original campaign depended on nightly builds which have since been withdrawn from the Python Package Index, so its reproduction was prevented. Version 1.0.0 depends only on released versions of its converter, its runtime and its quantizer, and is installed from the Python Package Index, under the name `nn-lit`, through a single console entry point. Linux and Apple Silicon are supported. The model collection is installed as a dependency and retrieved at the first use.

**Measurements attributable to the intended model.** Models are built now at their training resolution instead of at a fixed size, so an architecture such as VisionTransformer, which expects inputs of 299×299, loads its weights and does not fail. Each transfer to the phone is verified. A failed copy marks the model as failed, so no stale file is timed.

**Quantization calibrated on representative data.** Integer export used before a representative set of synthetic random noise. Such a set misrepresents the activation statistics and can reduce the accuracy: AirNet, for example, reached 79.1% top-1 accuracy on CIFAR-10 compared to 86.6%. The calibration uses real images now. These images are processed with the input transformation of the model itself.

**A tested programming interface.** Output parsing, error extraction and record construction are separated from the device control into documented importable modules, which need neither the deep learning frameworks nor the Android Debug Bridge. Each device keeps its own progress ledger. Every device command carries an explicit serial number, so one process per phone runs at the same time on one workstation and writes into a shared results folder.

# State of the field

*AI Benchmark* [@Ignatov2018AIBenchmark; @Ignatov2019AIBenchmark] and *MLPerf Mobile* [@Reddi2022MLPerfMobile] are established mobile benchmarks, but they execute fixed curated sets of reference models in order to compare *devices*, and arbitrary research architectures at large scale are not ingested by them. Measured latency is used as ground truth by latency predictors such as *nn-Meter* [@Zhang2021nnMeter] and *BRP-NAS* [@Dudziak2020BRPNAS], by hardware aware search methods such as *MnasNet* [@Tan2019MnasNet] and *ProxylessNAS* [@Cai2019ProxylessNAS], and by benchmarks such as *HW-NAS-Bench* [@Li2021HWNASBench], which do not provide the infrastructure to obtain it. The closest work in spirit is @Zhang2022MobileLibraries. 171 models across mobile deep learning libraries are benchmarked there, while the study is fixed and is not a tool for the models and the devices of the researchers themselves. Compiler stacks such as *TVM* [@Chen2018TVM] tune individual models, and *Qualcomm AI Hub* [@QualcommAIHub] offers hosted profiling as a vendor specific service. The `benchmark_model` utility of LiteRT [@TFLiteBenchmarkTool] measures one already converted model per invocation, but no conversion, bookkeeping, fault tolerance or standardized result schema is provided by it.

NN-Lite uses these components instead of replacing them. Conversion is done with LiteRT Torch [@LiteRTTorch], while the models are executed by the LiteRT runtime [@Abadi2016TensorFlow] and by the official `benchmark_model` binary, and the device is controlled by the Android Debug Bridge [@AndroidADB]. The Android Neural Networks API (NNAPI) [@AndroidNNAPI] requests NPU execution. Its contribution consists in the orchestration layer, which integrates these components into a reproducible and fault tolerant workflow from PyTorch models to physical device measurements and structured records.

# Software design

![How NN-Lite benchmarks one model on a physical phone: (1) the model is read from a local file or from the LEMUR dataset [@Goodarzi2025LEMUR]; (2) the network is rebuilt on the workstation and converted to FP32 and INT8 LiteRT files; (3) the files are copied over USB and timed with `benchmark_model` on the CPU, GPU and NPU paths; and (4) one JSON record is written. The safeguards at the bottom make the unattended multi-day runs possible.\label{fig:architecture}](figures/architecture.png){ width=100% }

**Host-driven execution using standard Android tooling.** Google's prebuilt `benchmark_model` binary is transferred once to the phone by NN-Lite, and it is invoked per model and per path, so root access and model specific signed applications are avoided. End to end application latency is not reported.

**Models are rebuilt rather than assumed to be portable.** A model taken from the collection consists of PyTorch code, hyperparameters and a checkpoint. NN-Lite reinstantiates the network and restores its weights. A model supplied as a file is self-describing when it was exported with `torch.export`, and otherwise it is reconstructed from its own module, so naming collisions among models saved under a shared class name are avoided. Either way an architecture enters the pipeline, and no configuration per model is needed.

**Every execution path is measured, and failures are data.** Each model is exported in FP32. INT8 export with full integer post training quantization [@Jacob2018Quantization] follows when it is supported. Every path is selected explicitly instead of being delegated to the default of the runtime, so every latency is attributable to a known backend. The FP32 model is measured anyway if the quantization fails on an operator which is not supported. When a path fails, its diagnostic output is recorded in the result and in the per device log, while a model whose paths all fail is retained and marked invalid instead of being dropped.

**Fault-tolerant, resumable runs.** NN-Lite pauses between the models, so that cooling is possible. The process is restarted from time to time in order to release the memory which the conversion toolchain holds. NN-Lite waits when a device is disconnected.

**One schema for every model source.** Results are written as one file per model, quantization setting and device, in the training statistics layout of the collection. A run made from the installed package can be copied into a checkout and contributed. Each record contains the latency distribution over repeated runs for every path, the fastest path, and the input shape, memory state and device telemetry needed for interpretation, and both sources are compared in this way. The analysis code stays unmodified.

Two limitations have to be mentioned. First, NN-Lite reports wall clock latency under a fixed protocol and does not isolate the state of the frequency governor or the energy per operator. Second, the NPU execution is mediated by the NNAPI, whose behaviour depends on the installed runtime and on the drivers: NNAPI can fall back to CPU execution when the compilation or the execution on an accelerator fails, and the API was deprecated in Android 15 [@AndroidNNAPI]. The requested path and the runtime output are recorded by NN-Lite, which cannot guarantee that all the operations ran on the intended accelerator.

# Research impact statement

NN-Lite is the on device measurement component of the LEMUR ecosystem [@Goodarzi2025LEMUR]. Its emulator execution path provided the inference latency measurements for more than 7,500 models in LEMUR 2 [@Uzun2026LEMUR2]. The physical device path produced the delegate resolved latency dataset reported in @Din2026NNLite, which covers 586 architectures across five commodity phones from three chip vendors. These records are publicly available. They have been used as well to fit device specific latency models by symbolic regression [@Dhanani2026Symbolic] and to show that FLOPs predict latency poorly unless the backend is taken into account [@Din2026BMVC].

Release 1.0.0 is on the Python Package Index. Unit tests cover the release and run in continuous integration on three Python versions, while the documentation addresses external users. We verified both model sources end to end on a phone. The verification included a model outside the collection (torchvision MobileNetV3), so other groups can benchmark their own models.

# AI usage disclosure

We designed and implemented the NN-Lite software. No generative AI assistance was used for it, while Claude Opus 5.5 (Anthropic) assisted with drafting the manuscript, with checking `paper.bib` and with creating \autoref{fig:architecture}. All the content produced with AI assistance was reviewed, edited and validated by the authors. We made all the final scientific and software decisions.

# Acknowledgements

This work was partially supported by the Alexander von Humboldt Foundation. We thank Saif U Din and Muhammad Ahsan Hussain, whose work on the measurement campaign reported in @Din2026NNLite shaped the requirements of this release, and we thank the contributors to the LEMUR dataset.

# References
