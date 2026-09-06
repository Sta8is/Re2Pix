<div align="center">
<h1>Representations Before Pixels: Semantics-Guided Hierarchical Video Prediction</h1>

**Efstathios Karypidis<sup>1,3</sup>, Spyros Gidaris<sup>2</sup>, Nikos Komodakis<sup>1,4,5</sup>**

<sup>1</sup>Archimedes/Athena RC <sup>2</sup>valeo.ai  
<sup>3</sup>National Technical University of Athens  <sup>4</sup>University of Crete   <sup>5</sup>IACM-Forth

[![Re2Pix](https://img.shields.io/badge/arXiv-2604.11707-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.11707)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
<!-- [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Pretrained_Model-blue)]() -->
<!-- [![Project Page](https://img.shields.io/badge/-Project_Page-green.svg?colorA=333&logo=html5)]() -->
<!-- [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)]() -->


</div>

<img src=".assets/.github/overview.png" width="100%"/>

<br>

Code and pretrained models will be released soon. Stay tuned!

# Contents
1. [News](#news)
2. [Setup](Setup)
3. [Usage](Usage)
4. [To-Do](#to-do)
5. [Citation](#citation)



# Setup
This framework is built on top of [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2) and [DINO-Foresight](https://github.com/Sta8is/DINO-Foresight).
Cosmos-Predict is a collection of general-purpose world foundation models. The repository contains many features that are not used in Re2Pix. We did an initial cleanup of the repository, but it still contains a lot of unused code. To be refined further in the future.

For the installation instructions, please refer to the [Installation](docs/Installation.md) page.

# Usage
Re2Pix is a hierarchical video prediction framework that takes as input 13 frames of a video and predicts the next 12 frames in a hierarchical manner. Resolution used is 448x.
For instructions on how to use Re2Pix, please refer to the [Usage](docs/Usage.md) page.

# To-Do
- [x]  Provide inference script
- [ ]  Add training scripts
- [ ]  Improve the documentation and code cleanup
- [ ]  Add more pretrained models


# News
- **2026-06-17**: Re2Pix is accepted at ECCV 2026! 🎉
- **2026-04-13**:  [Arxiv Preprint](https://arxiv.org/abs/2604.11707) and GitHub repository are released! 


# Citation
If you found Re2Pix useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

```bibtex
@article{karypidis2026re2pix,
  title={Representations Before Pixels: Semantics-Guided Hierarchical Video Prediction},
  author={Karypidis, Efstathios and Gidaris, Spyros and Komodakis, Nikos},
  journal={arXiv preprint arXiv:2604.11707},
  year={2026}
}
```
