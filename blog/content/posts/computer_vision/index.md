---
title: <Computer Vision> Basic Concepts
date: 2025-06-29T18:33:20Z
lastmod: 2025-06-29T18:33:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: computer_vision.png
categories:
  - computer vision
  - image
  - media
tags:
  - computer vision
  - DINO
  - DETR
# nolastmod: true
draft: false
---

# Editing Media Data

## Types of Image Formats

![image_formats](./image_formats.png)

## Types of Video Formats

![video_formats](./video_format.png)

### What is a codec?

`CO`mpression - `DEC`omposition

We can't save every pixel of all frames - **It's gonna be too large(1920\*1080 image, 30FPS, 10min video: 108GB)**

1. Intra-frame compression
   Compress in a single image

- Compress repeated sequence (ex) blue sky - `blue pixel from a to b`)

2. Inter-frame Compression
   Save only the difference between consecutive frames

- Person talking: only save the mouth movement

## Types of Audio Formats

![audio_format](./audio_format.png)

## Other Terms

Bitrate: **Amount of bits to represent 1 second of video/audio**
Video Encoder: NVENC, H.264/HEVC, software(x264)
