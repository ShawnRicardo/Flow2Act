<h1 align="center">Flow2Act: Learning to Act through Geometrically Grounded Motion Anticipation</h1>

<p align="center">
  <a href="https://shawnricardo.github.io/">Lijun Guo</a><sup>1,&#42;</sup>,
  <a href="https://maxwell-zhao.github.io/">Haoyu Zhao</a><sup>2,3,&#42;</sup>,
  Xianda Guo<sup>1,†</sup>, Chengwei Wu<sup>4</sup>, Yetong Zeng<sup>3,5</sup>,
  Chenglong Zhang<sup>6</sup>, Qian Zhou<sup>1</sup>, Zhongyu Li<sup>2,3,✉️</sup>,
  and Hua Zou<sup>1,✉️</sup>
</p>

<p align="center">
  <sup>1</sup>&nbsp;Wuhan University &emsp;
  <sup>2</sup>&nbsp;The Chinese University of Hong Kong &emsp;
  <sup>3</sup>&nbsp;Hong Kong Embodied AI Lab &emsp;
  <sup>4</sup>&nbsp;NIO &emsp;
  <sup>5</sup>&nbsp;University of Hong Kong &emsp;
  <sup>6</sup>&nbsp;Nanjing University
</p>

<p align="center">
  <a href="https://shawnricardo.github.io/Flow2Act.github.io/">🏠 Project Page</a> |
  <img src="assets/icons/paper.svg" width="16" height="16" alt=""> Paper (Coming soon) |
  <a href="https://github.com/ShawnRicardo/Flow2Act">Code</a> |
  <a href="https://huggingface.co/ShawnRicardo/Flow2Act-HF"><img src="assets/icons/hugging-face.svg" width="16" height="16" alt=""> Hugging Face</a> |
  <a href="https://modelscope.cn/models/ShawnRicardo/Flow2Act-MS"><img src="assets/icons/modelscope.png" width="16" height="16" alt=""> ModelScope</a>
</p>

<p align="center">
  <sup>&#42;</sup> Equal contribution &emsp;
  <sup>†</sup> Project Lead &emsp;
  ✉️ Co-corresponding authors
</p>

## Overview

**Flow2Act** learns to act through geometrically grounded motion anticipation. It combines sparse decision history with learnable future queries and uses geometric and future-motion supervision to connect visual representations with action prediction.

This repository hosts independent implementations built on different VLA backbones. Each project provides its own environment setup, data preparation, training, and evaluation instructions.

## News

- **[2026/11/26]** Added the OpenPI and Xiaomi-Robotics-0 implementations with LIBERO training and LIBERO / LIBERO-Plus evaluation guides.

## Architecture

![Flow2Act architecture showing sparse decision history, learnable future queries, depth and 2D/3D displacement supervision, and the action expert.](assets/pipeline.png)

## Get Started

Choose an implementation and follow its README:

| Implementation | Base Model | Getting Started |
| --- | --- | --- |
| **Flow2Act-OpenPI** | OpenPI π0.5 (JAX) | [Setup, training, and evaluation](Flow2Act-openpi/README.md) |
| **Flow2Act-XM0** | Xiaomi-Robotics-0 | [Setup, training, and evaluation](Flow2Act-xm0/README.md) |
| **Flow2Act-GR00T** | Isaac GR00T | Coming soon |
| **Flow2Act-LingBot** | LingBot-VLA-v2 | Coming soon |

## Acknowledgements

We thank the authors and contributors of [OpenPI](https://github.com/Physical-Intelligence/openpi), [Xiaomi-Robotics-0](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0), [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [Track4World](https://github.com/TencentARC/Track4World), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus), [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), and [Spatial Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing) for their open-source work. Component attribution and license notices are provided in each implementation.
