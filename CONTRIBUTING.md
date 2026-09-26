# Contributing to NN-Lite

Thank you for your interest in NN-Lite. Contributions of all kinds are welcome: bug reports,
new device results, documentation fixes and code.

## Reporting bugs and asking questions

Please open an issue in the [issue tracker](https://github.com/ABrain-One/nn-lite/issues) and
include:

- what you ran (the exact command and options),
- the phone model and Android version (`adb shell getprop ro.product.model` and
  `adb shell getprop ro.build.version.release`),
- the relevant part of the console output and, for benchmark failures, of
  `nn-dataset/_work/benchmark_errors_<device>.log`.

## Proposing changes

1. Fork the repository and create a branch from `main`.
2. Make your change. Keep the JSON record format backward compatible: new fields may be added,
   but existing fields must keep their names, meaning and order.
3. Run the unit tests and add tests for new behaviour where possible:
   ```bash
   pip install pytest
   python -m pytest tests
   ```
4. If your change affects benchmarking, run it on at least one physical phone and mention the
   device in the pull request.
5. Open a pull request describing what changed and why. Add a line to `CHANGELOG.md` under
   "Unreleased".

## Support and governance

NN-Lite is developed and maintained by the Computer Vision Lab (CAIDAS & IFI) at the University
of Würzburg, Germany, as part of the ABrain One / LEMUR ecosystem. Dr. Dmitry Ignatov leads the
project and makes final decisions on releases and on changes to the record format; Faraz Kayani
maintains the physical-device benchmarking engine.

Issues and pull requests are reviewed on a best-effort basis, usually within two weeks. Releases
are tagged on GitHub and published on PyPI as `nn-lite`.

## Code of conduct

Please be respectful and constructive in all project spaces. Harassment or discriminatory
behaviour is not tolerated; the maintainers may remove comments or contributions that violate
this expectation.
