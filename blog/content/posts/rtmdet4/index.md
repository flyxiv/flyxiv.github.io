---
title: <Computer Vision> Implementing RTMDet-Ins 5. Pretraining Backbone on ImageNet
date: 2025-03-30T08:47:20Z
lastmod: 2025-03-30T08:47:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: imagenet.png
categories:
  - computer vision
tags:
  - deep learning
  - machine learning
  - pytorch
  - segformer
  - computer vision
  - image segmentation
# nolastmod: true
draft: false
---

Object Detection Model typically gets trained in two stages:

1. Trains **only backbone on classification task** so that the backbone can provide reasonable output first.
2. Trains the whole model on object detection task.

This is because **the detection head can't train if the backbone isn't providing useful embedding of the given image.** The backbone's output
will give a random output to the head, which will lead the head to train in the wrong direction.

The backbone training is usually done on ImageNet classification dataset.


# Looking at ImageNet Dataset
1) Has only one annotation
2) Images have different width and height
3) also has Bbox

```xml
<annotation>
	<folder>n01440764</folder>
	<filename>n01440764_96</filename>
	<source>
		<database>ILSVRC_2012</database>
	</source>
	<size>
		<width>500</width>
		<height>375</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>n01440764</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>34</xmin>
			<ymin>128</ymin>
			<xmax>430</xmax>
			<ymax>305</ymax>
		</bndbox>
	</object>
</annotation>
```