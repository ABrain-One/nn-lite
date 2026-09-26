---
title: 'NN-Lite: Automated PyTorch-to-Android conversion and large-scale benchmarking on physical mobile devices'
tags:
  - Python
  - Android
  - deep learning
  - edge AI
  - model deployment
  - quantization
  - benchmarking
  - TensorFlow Lite
authors:
  - name: Dmitry Ignatov
    # orcid: 0000-0000-0000-0000   # TODO: add ORCID
    affiliation: 1
  - name: Faraz Kayani
    # orcid: 0000-0000-0000-0000   # TODO: add ORCID
    corresponding: true
    affiliation: 1
  - name: Radu Timofte
    # orcid: 0000-0000-0000-0000   # TODO: add ORCID
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
increasingly, dedicated neural processing units (NPUs). How quickly a given
model runs depends strongly on which phone and which processor is used, and the
only reliable way to find out is to run the model on the phone itself.
Researchers, however, usually design and train their models on desktop
computers with the PyTorch library [@Paszke2019PyTorch], and moving each model
onto a phone and timing it by hand does not scale beyond a handful of
experiments.

NN-Lite is a Python tool that automates this whole journey. It takes neural
networks from the LEMUR model collection [@Goodarzi2025LEMUR], converts each one
into the format Android phones understand, copies it to a phone or an Android
emulator connected to the computer, times it on every available processor, and
writes the results back into LEMUR in a single, consistent format. A run can
cover hundreds of models on one device without supervision, and it continues
where it stopped if the phone is unplugged or the computer restarts.

# Statement of need

Research on efficient deep learning (hardware-aware neural architecture search,
quantization, latency prediction, model compression) needs latency and
feasibility measurements for *many* architectures on *many* real devices.
Measurements from emulators or desktop GPUs are a poor substitute: emulators do
not expose a phone's GPU and NPU drivers, and on real phones the same model can
succeed on one accelerator and fail on another because of missing operator
support [@Ignatov2019AIBenchmark; @Almeida2021SmartAtWhatCost]. Collecting such
data by hand requires converting every model, handling conversion failures,
pushing files to the device, invoking the right accelerator, parsing logs, and
recording which phone, operating system and memory state produced each number.

NN-Lite targets researchers who train models in PyTorch and need reproducible,
per-device, per-accelerator measurements at dataset scale. It turns a model
collection into on-device statistics with one command, records both successes
and failures, and stores results next to the models' training statistics in
LEMUR so that accuracy and deployment cost can be queried together. An earlier
version of the pipeline, which executed models only on Android emulators, is
described in @Din2025NNLite; the software described here adds a benchmarking
path for physical devices, dual-precision (FP32 and INT8) conversion, and the
fault tolerance needed for long unattended runs on consumer phones.

# State of the field

Existing tools cover parts of this workflow. *AI Benchmark*
[@Ignatov2018AIBenchmark; @Ignatov2019AIBenchmark] and *MLPerf Mobile*
[@Reddi2022MLPerfMobile] are established mobile benchmark suites, but they
evaluate a fixed, curated set of reference models to rank *devices*; they are
not designed to ingest thousands of arbitrary research architectures.
Latency predictors such as *nn-Meter* [@Zhang2021nnMeter] and hardware-aware
NAS benchmarks such as *HW-NAS-Bench* [@Li2021HWNASBench] consume measured
latencies as ground truth rather than producing them for new model families.
Compiler stacks such as *TVM* [@Chen2018TVM] optimise and tune individual
models, and vendor services such as *Qualcomm AI Hub* [@QualcommAIHub] profile
models on hosted hardware from a single chip vendor under a proprietary
service. Finally, the `benchmark_model` utility shipped with LiteRT
[@TFLiteBenchmarkTool] times one already-converted model per invocation, with
no conversion, bookkeeping or result schema.

We therefore chose to *build on* rather than replace these components. NN-Lite
does not re-implement conversion, kernels or timing: it uses LiteRT Torch
[@LiteRTTorch] for conversion, TensorFlow Lite runtimes [@Abadi2016TensorFlow]
and the official `benchmark_model` binary for execution, the Android Neural
Networks API [@AndroidNNAPI] for NPU access and the Android Debug Bridge
[@AndroidADB] for device control. Its contribution is the missing layer that
none of these tools provide: dataset-scale orchestration from a model
collection to a device fleet, failure-tolerant execution, and a stable result
schema joined to the models' training records. Contributing this layer
upstream was not an option, because it is specific neither to one converter nor
to one benchmark tool; it is the glue between a model *dataset* and many of
them.

# Software design

\autoref{fig:architecture} shows the architecture. A host-side Python command
line drives four stages (load, convert, orchestrate, record) and talks to the
target only through `adb`, which reaches USB-connected phones and emulators in
the same way.

![Architecture of NN-Lite. Models, weights and hyperparameters are read from LEMUR, converted on the host, executed on physical devices through the native `benchmark_model` binary or on emulators through the NN-Lite Android app, and the resulting measurements are written back into LEMUR.\label{fig:architecture}](figures/architecture.png){ width=100% }

The main design decisions and their trade-offs are:

**Host-driven execution instead of an on-device app.** For physical devices,
NN-Lite pushes Google's prebuilt `benchmark_model` binary to the phone and runs
it for each accelerator. This requires neither root access nor rebuilding and
signing an app per model, works on any Android phone with USB debugging, and
yields timings comparable with those of other LiteRT users. The cost is that
application-level overhead (image decoding, Java/JNI calls) is not measured. The
emulator path keeps a small Kotlin app built on the TensorFlow Lite
Interpreter, because emulators are primarily used to validate that a converted
model loads and runs end to end.

**Models are rebuilt from source, not from exported graphs.** Each LEMUR model
is stored as PyTorch code plus hyperparameters and a checkpoint. NN-Lite
re-instantiates the network, restores the weights and derives the input
resolution from the model's own preprocessing definition. This keeps the
converted model faithful to the trained one and lets new architectures enter
the pipeline without any per-model configuration.

**Two precisions with graceful degradation.** Every model is exported in FP32
and, via full-integer post-training quantization, in INT8. If quantization
fails, the FP32 result is still benchmarked and recorded, so a single
unsupported operator never removes a model from the dataset.

**Failures are data.** Each accelerator is benchmarked independently. When one
fails, the relevant lines of the runtime's output are extracted and stored in
the record (e.g., `npu_error`) and written to a per-device log; a model for
which all accelerators fail is recorded with `valid: false` instead of being
dropped. This avoids survivorship bias in downstream analyses and documents
operator-support gaps, which are themselves a research signal.

**Crash-only, resumable runs.** Consumer phones throttle, lose USB connections
and accumulate memory pressure during multi-day runs. NN-Lite keeps a persistent
queue of processed and failed models, pauses between models and between
sessions to let the device cool, re-executes itself periodically to release host
memory held by the conversion toolchain, and blocks until a disconnected device
reappears. Any run can be interrupted and restarted without repeating work.

**One schema shared with LEMUR.** Results are stored as one JSON file per model,
precision and device, in the same directory convention LEMUR uses for training
statistics. Each record contains per-accelerator mean, minimum, maximum and
standard deviation of latency, the fastest accelerator, input dimensions, memory
state, and device telemetry (SoC, CPU topology, OS build), together with an
`emulator` flag. Emulator and physical-device results can therefore be compared
or filtered without changing downstream code.

# Research impact statement

NN-Lite is the on-device measurement component of the LEMUR ecosystem
[@Goodarzi2025LEMUR]; the pipeline and its emulator-based evaluation were
introduced in @Din2025NNLite. Using the physical-device path described here, we
have benchmarked about 540 LEMUR image-classification architectures, each in
FP32 and INT8 and on CPU, GPU and NNAPI, on five physical devices covering
Qualcomm Snapdragon (720G, 888), MediaTek Helio G85 and HiSilicon Kirin 710
chipsets, producing more than 5,400 per-device records. These records are
publicly available in the LEMUR repository and can be retrieved with its API
next to accuracy and training metadata. The same deployment tooling supports the
lab's work on mobile-oriented models, including real-image denoising for mobile
NPUs [@Kayani2026Denoising] and lightweight facial age estimation
[@Kumar2026MobileAgeNet].

# AI usage disclosure

Generative AI was used in preparing this paper. Claude (Anthropic), accessed
through the Claude Code agent (model version: TODO — to be filled in by the
authors), was used to draft the text of this manuscript, to assemble
`paper.bib`, and to draw the architecture diagram in
\autoref{fig:architecture}, all based on the repository source code and on
information supplied by the authors. TODO — authors to state here whether AI
tools (and which tools and versions) were used when writing the NN-Lite source
code or documentation; if none were, state this explicitly. The authors
reviewed, edited and validated all AI-assisted content, verified every
reference, and made all design and architectural decisions for the software.

# Acknowledgements

We thank Saif U Din, Muhammad Ahsan Hussain and Mohsin Ikram for their work on
the emulator-based pipeline and the NN-Lite Android application, and all
contributors to the LEMUR dataset. TODO — add funding sources, or state that
this work received no specific funding.

# References
