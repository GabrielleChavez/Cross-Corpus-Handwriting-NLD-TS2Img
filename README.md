# Cross-Corpus Handwriting Assessment for Neurodegenerative Disorders

This repository accompanies the paper **“Cross-Corpus and Cross-domain Handwriting Assessment of NeuroDegenerative Disorders via Time-Series-to-Image Conversion.”**
The project explores how handwriting signals can be used to detect **neurodegenerative disorders (NDs)** such as Parkinson’s Disease (PD) and Alzheimer’s Disease (AD) using deep learning.

The proposed framework converts **handwriting time-series signals into image representations**, allowing a **single convolutional neural network classifier** to analyze both traditional handwriting images and dynamic time-series recordings.

---

# Abstract

Handwriting is affected by neurodegenerative disorders such as Parkinson's disease (PD) and Alzheimer's disease (AD). Prior studies have analyzed handwriting tasks using feature-based approaches or image analysis techniques, but these methods have struggled to generalize across datasets, particularly when bridging time-series and images.

In this work, we propose a unified framework that converts handwriting time-series signals into image representations so only a single image-based classifier, a ResNet50 pretrained on ImageNet-1k, is needed. Healthy control vs. PD binary classification experiments show competitive performance across time-series and image datasets; the strongest results are on template-based image datasets, highlighting sensitivity to acquisition protocols.

On the NeuroLogical Signals dataset, the proposed approach improves performance compared to previous time-series-based methods on clock and spiral drawing, with strong gains for AD (up to **85.9 F1** in clock drawing). Finally, cross-dataset and multi-dataset evaluations reveal domain shift: transfer performance varies widely across datasets and remains limited when crossing acquisition protocols and modalities (up to **83.1 F1** in cross-dataset transfer on HandPD), indicating that additional domain- and modality-invariant strategies are needed for cross-corpus generalization.

---

# Method Overview

The proposed pipeline consists of three main components:

1. **Time-Series to Image Conversion**

   * Handwriting trajectories (x, y coordinates) and pen pressure are converted into image representations.
   * Stroke density captures writing speed, while stroke thickness encodes pressure.

2. **Image Preprocessing**

   * All samples are resized to **224×224 pixels**.
   * Luminance normalization and Gaussian blur are applied to standardize visual features.

3. **Deep Learning Model**

   * A **ResNet50 (ImageNet pretrained)** backbone extracts visual features.
   * A lightweight **MLP classifier** performs binary classification (PD vs Control or AD vs Control).
   * Training uses **5-fold cross-validation** with subject-level separation.

---

# Datasets

Experiments use five handwriting datasets with different modalities:

* **HandPD** – Spiral drawing images with templates
* **NewHandPD** – Similar protocol with additional sensor data
* **ParkD** – Dynamic spiral drawing without visible template
* **PaHaW** – Time-series handwriting trajectories
* **NeuroLogical Signals (NLS)** – Multimodal clinical dataset including PD, AD, and control participants

---

# Key Results

### Performance on Image Datasets

* Achieved **competitive accuracy** with prior image-based methods.
* Strongest results observed on **template-based spiral datasets** (HandPD, NewHandPD).

### Performance on Time-Series Tasks

* The proposed time-series-to-image representation **improved performance over previous methods** on the NLS dataset.
* Notable improvement in **Alzheimer’s detection from clock drawing**, reaching **85.9 F1 score**.

### Cross-Dataset Generalization

* Cross-corpus experiments reveal **significant domain shift** between datasets.
* Best transfer result reached **83.1 F1 on HandPD**, but performance varied widely depending on acquisition protocol and modality.

---

# Conclusion

This work introduces a **unified framework for handwriting-based neurodegenerative disorder detection** that bridges image and time-series modalities through a time-series-to-image conversion strategy.

The approach enables a **single CNN model to process heterogeneous handwriting datasets**, simplifying system design while maintaining strong performance. Experiments demonstrate improved results on several tasks, particularly within the NLS dataset.

However, cross-dataset experiments reveal substantial **domain shift caused by differences in acquisition protocols, templates, and recording modalities**. These findings suggest that future research should focus on **domain-invariant and modality-robust representations** to improve cross-corpus generalization for clinical handwriting analysis.

