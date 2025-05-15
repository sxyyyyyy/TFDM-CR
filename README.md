# TFDM-CR: Time-Frequency Diffusion Model for Capacity Prediction Considering Regeneration
This is a PyTorch implementation of TFDM-CR model and discussion experiments proposed by our paper "Time-Frequency Diffusion Model for Capacity Prediction Considering Regeneration".
# 1. Overview
![fig1](https://github.com/user-attachments/assets/a81f97d6-7d4b-4e23-9545-0a19b777f37e)
Fig.1 Lithium Battery Capacity Prediction Framework Based on DTW-Time-Frequency Feature Weighting and Diﬀusion Model
We proposes a diﬀusion model framework incorporating capacity regeneration phenomena and time-frequency feature weighting for lithium-ion battery capacity sequence prediction, eﬀectively addressing prediction deviations caused by capacity regeneration.

# 2. Requirements
- torch==2.2.2
- numpy==1.26.4
- pandas==1.5.3
- gluonts==0.9.6

# 3. Training and Testing
You can perform Capacity prediction on two datasets in the two files: 'main_nasa.ipynb', 'main_calce.ipynb'.
