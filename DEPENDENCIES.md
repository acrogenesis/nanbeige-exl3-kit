# Measured dependency pins

The measurements used Transformers4.57.6, ExLlamaV3 1.6.5+cu132.torch2.13.0, torch2.13.0+cu132 and huggingface-hub0.36.2. The supplied requirements preserve those measured versions. A fresh installation from setup.sh was not tested.

After publication GitHub reported five advisories against the Transformers pin (four high, one moderate). These are upstream package advisories; this kit has not separately reproduced their exploit paths. We have not claimed that the pins form a current security-maintained deployment environment. Dependency upgrades and a fresh compatibility benchmark remain untested.

- [GHSA-x9r9-c232-4q39](https://github.com/advisories/GHSA-x9r9-c232-4q39): high — Hugging Face Transformers downloads custom generation code before trust consent.
- [GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w): high — Transformers save_pretrained path traversal allows arbitrary file writes through chat template names.
- [GHSA-fgcw-684q-jj6r](https://github.com/advisories/GHSA-fgcw-684q-jj6r): high — huggingface/transformers: Arbitrary Code Execution During Model Initialization in the LightGlue Model Loading Path.
- [GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72): high — HuggingFace transformers vulnerable to remote code execution.
- [GHSA-69w3-r845-3855](https://github.com/advisories/GHSA-69w3-r845-3855): medium — HuggingFace Transformers allows for arbitrary code execution in the `Trainer` class.
