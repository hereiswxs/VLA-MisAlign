<h1 align="center">
VLA-MisAlign: A Semantic-Guided Adversarial Attack for Vision-Language-Action Models
</h1>

<div align="center">
  
  [📄[Conference Version](https://dl.acm.org/doi/pdf/10.1145/3774904.3792315)] &nbsp;&nbsp; [📁[Project](https://github.com/hereiswxs/VLA-MisAlign)]
  
</div>

## Overview

This repository provides the official implementation and benchmarks for our paper “Can We Trust VLA Models? Undermining Behavioral Reliability via Cross-Modal Alignment Disruption.” We investigate the reliability risks of Vision-Language-Action (VLA) models and demonstrate how disrupted cross-modal alignment can lead to erroneous robotic behaviors. To highlight this vulnerability, we propose VLA-MisAlign, a semantic-guided adversarial attack that weakens the alignment between visual observations and language instructions through imperceptible image perturbations. 

---

## Quick Start

### Training

#### 1. Prepare datasets

Please download and preprocess datasets from:👉 [LIBERO](https://libero-project.github.io/datasets)

#### 2. Train VLA-MisAlign

```bash
bash scripts/train.sh
```

The `train_weight` folder contains pre-trained weight files that can be directly used for evaluation.

---

## Evaluation

### 1.Evaluated Models

We evaluate VLA-MisAlign on:

- [OpenVLA](https://github.com/openvla/openvla)
- [CEED-VLA](https://github.com/OpenHelix-Team/CEED-VLA)
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base)
- [OpenVLA-OFT+](https://github.com/moojink/openvla-oft)

### 2. Download checkpoints

Here we provide OpenVLA checkpoints as examples. For other victim models, please refer to their official repositories or model pages for checkpoint downloads.

[openvla-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)

[openvla-7b-finetuned-libero-object](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object)

[openvla-7b-finetuned-libero-goal](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal)

[openvla-7b-finetuned-libero-10](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10)


### 3. Run evaluation

Example: evaluating VLA-MisAlign on `LIBERO-Object` with OpenVLA.
```
python experiments/robot/libero/adv_run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /openvla-main/openvla-7b-finetuned-libero-object \
    --task_suite_name libero_object \
    --ourdir /openvla-main/experiments/robot/history/object \
    --center_crop True \
    --num_gamma 0.3
```

---

## Results

Effectiveness of attacks on the victim models under different perturbation ratios.

### OpenVLA

<p align="center">
  <img src="images/FTSR.png" width="85%">
</p> 

#### LIBERO-Spatial
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 82.4% | 94.7% | 88.2% | 90.5% | 74.6% | 80.3% | 96.3% | 90.1% | 84.8% | 74.2% | 85.6% |
| 0.10 | 76.2% | 88.5% | 90.1% | 96.3% | 54.1% | 88.3% | 97.5% | 84.2% | 74.5% | 68.1% | 81.8% |
| 0.20 | 72.2% | 77.8% | 72.4% | 91.7% | 58.2% | 87.9% | 86.2% | 28.9% | 82.2% | 46.4% | 70.4% |
| 0.25 | 68.2% | 73.8% | 58.3% | 92.7% | 46.5% | 64.6% | 72.2% | 18.9% | 73.8% | 21.7% | 59.1% |
| 0.30 | 62.4% | 62.1% | 26.8% | 75.1% | 26.1% | 12.5% | 32.1% | 24.8% | 63.4% | 8.1% | 39.3% |
| 0.40 | 38.6% | 18.1% | 38.4% | 2.9% | 4.3% | 6.7% | 0.0% | 30.9% | 2.0% | 20.5% | 16.2% |

#### LIBERO-Goal
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 54.4% | 88.1% | 85.9% | 61.5% | 88.2% | 73.9% | 76.2% | 95.9% | 84.2% | 64.1% | 77.2% |
| 0.10 | 48.2% | 82.3% | 84.5% | 58.7% | 85.3% | 75.7% | 65.8% | 92.1% | 81.7% | 56.2% | 73.1% |
| 0.20 | 42.4% | 63.6% | 69.3% | 30.1% | 94.2% | 65.8% | 71.6% | 93.8% | 70.4% | 42.2% | 64.3% |
| 0.25 | 46.3% | 52.1% | 66.6% | 25.1% | 88.1% | 53.7% | 52.1% | 85.8% | 72.2% | 36.3% | 57.8% |
| 0.30 | 32.3% | 62.2% | 67.4% | 36.3% | 94.4% | 66.2% | 64.1% | 92.5% | 69.2% | 39.4% | 62.4% |
| 0.40 | 1.8% | 0.0% | 17.5% | 0.0% | 24.1% | 4.2% | 8.1% | 20.7% | 0.0% | 8.1% | 8.4% |

#### LIBERO-Object
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 79.3% | 53.7% | 73.1% | 41.9% | 83.5% | 68.2% | 68.8% | 80.6% | 52.4% | 70.5% | 67.2% |
| 0.10 | 75.8% | 47.1% | 69.7% | 43.3% | 79.5% | 50.9% | 22.0% | 34.6% | 42.4% | 70.7% | 53.6% |
| 0.20 | 48.6% | 24.9% | 28.2% | 58.5% | 58.8% | 12.1% | 6.9% | 6.3% | 10.7% | 9.0% | 26.4% |
| 0.25 | 25.4% | 2.8% | 11.2% | 31.7% | 15.6% | 4.1% | 0.0% | 0.0% | 2.3% | 28.9% | 12.2% |
| 0.30 | 0.3% | 0.6% | 11.2% | 24.8% | 13.7% | 0.5% | 0.9% | 0.4% | 0.6% | 19.0% | 7.2% |
| 0.40 | 0.1% | 0.0% | 1.8% | 6.6% | 1.4% | 0.2% | 0.7% | 0.9% | 0.3% | 2.0% | 1.4% |

#### LIBERO-10
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 57.8% | 71.0% | 63.4% | 47.7% | 41.1% | 76.5% | 54.9% | 56.3% | 24.6% | 50.7% | 54.4% |
| 0.10 | 49.9% | 77.2% | 43.6% | 35.1% | 41.8% | 58.5% | 48.0% | 60.7% | 18.3% | 58.9% | 49.2% |
| 0.20 | 21.3% | 51.7% | 49.0% | 25.5% | 43.8% | 66.2% | 28.6% | 40.9% | 12.4% | 32.6% | 37.2% |
| 0.25 | 10.4% | 26.0% | 30.8% | 38.2% | 36.7% | 60.1% | 6.9% | 28.6% | 22.3% | 32.0% | 26.8% |
| 0.30 | 4.5% | 8.3% | 38.7% | 22.1% | 19.8% | 30.4% | 6.2% | 20.9% | 16.6% | 14.5% | 15.8% |
| 0.40 | 0.6% | 0.1% | 6.9% | 1.4% | 0.7% | 3.0% | 1.8% | 1.3% | 5.5% | 8.7% | 3.0% |

---

### CEED-VLA

#### LIBERO-Spatial
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 100.0% | 84.5% | 87.9% | 95.1% | 82.3% | 92.1% | 86.4% | 77.6% | 86.4% | 73.9% | 86.6% |
| 0.10 | 92.8% | 85.9% | 78.2% | 93.8% | 66.9% | 75.7% | 85.2% | 90.1% | 90.7% | 69.2% | 82.9% |
| 0.20 | 90.8% | 100.0% | 70.1% | 87.8% | 57.4% | 76.8% | 71.8% | 88.3% | 91.4% | 53.8% | 78.8% |
| 0.25 | 89.6% | 92.6% | 74.2% | 81.8% | 39.1% | 92.3% | 81.6% | 71.6% | 86.3% | 34.1% | 74.3% |
| 0.30 | 88.2% | 91.2% | 69.3% | 79.4% | 22.6% | 94.6% | 62.3% | 49.8% | 84.7% | 44.6% | 68.7% |
| 0.40 | 81.7% | 68.8% | 45.9% | 73.7% | 7.9% | 79.1% | 33.5% | 0.0% | 85.1% | 10.0% | 48.6% |

#### LIBERO-Goal
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 58.2% | 84.5% | 90.1% | 31.7% | 99.6% | 78.9% | 89.5% | 96.9% | 92.3% | 58.0% | 78.0% |
| 0.10 | 42.4% | 78.3% | 81.2% | 26.1% | 97.8% | 86.9% | 86.5% | 96.1% | 90.4% | 42.2% | 72.8% |
| 0.20 | 54.2% | 68.5% | 70.1% | 10.1% | 98.2% | 66.5% | 88.2% | 96.1% | 86.2% | 39.1% | 67.7% |
| 0.25 | 46.2% | 32.8% | 26.7% | 8.2% | 86.1% | 38.4% | 72.3% | 76.2% | 66.7% | 34.5% | 48.8% |
| 0.30 | 34.7% | 16.2% | 28.4% | 2.1% | 56.1% | 29.8% | 54.2% | 54.6% | 45.2% | 19.5% | 34.1% |
| 0.40 | 16.2% | 2.1% | 2.3% | 0.0% | 28.5% | 15.2% | 4.8% | 2.1% | 4.3% | 12.1% | 8.8% |

#### LIBERO-Object
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 73.5% | 71.2% | 85.9% | 43.1% | 93.6% | 80.4% | 48.8% | 90.7% | 86.3% | 86.5% | 76.0% |
| 0.10 | 77.3% | 59.8% | 69.1% | 39.7% | 79.5% | 90.6% | 46.0% | 76.4% | 70.9% | 94.7% | 70.4% |
| 0.20 | 71.9% | 63.2% | 89.7% | 41.5% | 71.0% | 78.8% | 14.1% | 70.6% | 64.3% | 82.9% | 64.8% |
| 0.25 | 67.4% | 49.7% | 87.2% | 27.0% | 63.9% | 64.5% | 14.6% | 56.3% | 50.8% | 78.6% | 56.0% |
| 0.30 | 55.6% | 37.2% | 75.9% | 35.1% | 48.4% | 68.8% | 8.0% | 54.7% | 42.3% | 58.0% | 48.4% |
| 0.40 | 37.5% | 5.8% | 69.3% | 21.6% | 31.1% | 42.9% | 6.2% | 4.7% | 33.4% | 55.5% | 30.8% |

#### LIBERO-10
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 67.2% | 79.8% | 59.5% | 57.0% | 57.9% | 74.3% | 60.7% | 90.1% | 24.6% | 60.9% | 63.2% |
| 0.10 | 65.7% | 79.3% | 69.0% | 53.8% | 47.5% | 72.2% | 58.9% | 78.6% | 34.4% | 60.6% | 62.0% |
| 0.20 | 59.1% | 77.8% | 53.4% | 43.0% | 44.7% | 72.3% | 68.9% | 78.6% | 28.2% | 52.0% | 57.8% |
| 0.25 | 51.4% | 75.9% | 75.2% | 59.6% | 47.1% | 72.5% | 42.8% | 78.7% | 28.3% | 50.5% | 58.2% |
| 0.30 | 41.5% | 65.1% | 67.8% | 55.3% | 39.7% | 60.9% | 40.0% | 72.4% | 24.6% | 52.7% | 52.0% |
| 0.40 | 31.8% | 61.2% | 57.7% | 47.0% | 47.9% | 54.3% | 24.5% | 68.6% | 28.1% | 42.9% | 46.4% |

---

### SmolVLA

#### LIBERO-Spatial
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 75.9% | 92.5% | 87.2% | 60.8% | 71.4% | 35.7% | 77.6% | 74.8% | 73.2% | 64.9% | 71.4% |
| 0.10 | 75.7% | 95.6% | 86.8% | 75.9% | 70.5% | 31.4% | 81.7% | 79.6% | 42.5% | 70.3% | 71.0% |
| 0.20 | 70.4% | 90.1% | 90.3% | 68.5% | 47.2% | 15.7% | 81.0% | 73.5% | 83.2% | 47.1% | 66.7% |
| 0.25 | 72.2% | 90.5% | 88.0% | 58.7% | 41.3% | 13.6% | 69.1% | 27.4% | 77.2% | 37.0% | 57.5% |
| 0.30 | 62.2% | 82.0% | 80.4% | 66.1% | 48.3% | 9.5% | 51.2% | 16.0% | 88.1% | 27.2% | 53.1% |
| 0.40 | 60.1% | 54.8% | 64.3% | 72.5% | 59.4% | 3.6% | 29.7% | 3.0% | 83.4% | 19.2% | 45.0% |

#### LIBERO-Object
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 84.1% | 97.0% | 99.3% | 95.2% | 76.1% | 76.0% | 96.1% | 90.0% | 98.2% | 94.0% | 90.6% |
| 0.10 | 73.4% | 98.2% | 97.8% | 96.1% | 76.5% | 66.0% | 97.6% | 90.2% | 99.1% | 95.1% | 89.0% |
| 0.20 | 39.7% | 56.1% | 85.8% | 98.2% | 32.4% | 81.6% | 99.5% | 58.2% | 91.2% | 87.3% | 73.0% |
| 0.25 | 29.9% | 14.2% | 76.0% | 97.7% | 18.3% | 85.5% | 98.1% | 61.0% | 92.1% | 75.2% | 64.8% |
| 0.30 | 25.8% | 2.1% | 76.2% | 97.7% | 26.5% | 67.3% | 93.6% | 40.4% | 49.4% | 81.0% | 56.0% |
| 0.40 | 11.9% | 0.5% | 24.2% | 73.7% | 22.4% | 35.8% | 89.6% | 7.9% | 31.0% | 45.0% | 34.2% |

#### LIBERO-Goal
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 73.8% | 97.9% | 80.1% | 35.7% | 95.8% | 84.2% | 75.6% | 99.4% | 87.5% | 72.0% | 80.2% |
| 0.10 | 84.1% | 88.4% | 60.5% | 34.0% | 93.7% | 77.3% | 83.6% | 99.2% | 97.8% | 69.4% | 78.8% |
| 0.20 | 90.1% | 84.0% | 18.2% | 20.1% | 94.0% | 88.2% | 63.4% | 95.0% | 94.0% | 27.0% | 67.4% |
| 0.25 | 74.2% | 80.7% | 10.5% | 22.0% | 97.8% | 89.1% | 45.6% | 93.4% | 95.3% | 15.4% | 62.4% |
| 0.30 | 46.2% | 64.0% | 0.5% | 12.3% | 92.0% | 83.8% | 31.6% | 85.9% | 93.0% | 11.0% | 52.0% |
| 0.40 | 40.1% | 62.0% | 1.9% | 12.4% | 91.7% | 89.6% | 17.8% | 65.5% | 85.2% | 5.8% | 47.2% |

#### LIBERO-10
| σ | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Task10 | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 12.0% | 52.2% | 54.4% | 85.5% | 16.1% | 59.3% | 55.2% | 25.1% | 16.0% | 40.2% | 41.6% |
| 0.10 | 12.1% | 55.7% | 52.0% | 89.4% | 8.2% | 69.5% | 53.8% | 22.1% | 22.2% | 37.0% | 42.2% |
| 0.20 | 0.5% | 42.1% | 50.8% | 58.2% | 0.6% | 81.0% | 41.4% | 11.9% | 27.3% | 19.2% | 33.3% |
| 0.25 | 0.1% | 22.8% | 10.0% | 46.5% | 0.4% | 74.3% | 23.9% | 13.2% | 21.6% | 7.2% | 22.0% |
| 0.30 | 0.7% | 8.2% | 22.4% | 45.1% | 0.8% | 68.5% | 31.0% | 9.6% | 13.3% | 1.4% | 20.1% |
| 0.40 | 0.0% | 4.1% | 4.2% | 44.0% | 0.0% | 60.3% | 14.1% | 6.2% | 2.1% | 0.0% | 13.5% |

---

### OpenVLA-OFT+

<p align="center">
  <img src="images/twoviews.png" width="85%">
</p>

<div align="center">
  
| TaskSuite \ σ | 0.0 | 0.1 | 0.2 | 0.25 | 0.3 | 0.4 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LIBERO_Spatial | 86.1% | 80.2% | 71.7% | 68.9% | 65.0% | 59.1% |
| LIBERO_Object | 84.5% | 76.9% | 65.9% | 62.3% | 57.3% | 49.6% |
| LIBERO_Goal | 70.7% | 64.3% | 55.2% | 52.1% | 48.0% | 41.5% |
| LIBERO_10 | 77.7% | 70.7% | 60.6% | 57.3% | 52.7% | 45.6% |
| **Avg** | **79.5%** | **73.1%** | **63.4%** | **60.2%** | **55.8%** | **49.0%** |

</div>

---

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{zhao2026breaking,
  title={Breaking Cross-modal Alignment in Embodied Intelligence: A Multimodal Adversarial Attack Framework for Vision-Language-Action Models},
  author={Zhao, Zhihui and Dong, Xiaorong and Zheng, Yaowen and Chen, Xiaohui and Ren, Yimo and Cheng, Hangbei and Chen, Yongle and Sun, Limin},
  booktitle={Proceedings of the ACM Web Conference 2026},
  pages={2824--2834},
  year={2026}
}
```
