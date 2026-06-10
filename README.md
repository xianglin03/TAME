# Memorization-Free Diffusion Models for Tabular Data Synthesis with Correlation Preservation

## Introduction
<div align="center">
<figure style="text-align: center;">
    <img src="classifier guidance1.png" alt="classifier guidance1" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
    <figcaption>Figure 1: Overview of the Tame pipeline.</figcaption>
</figure>
<br>
</div>
Diffusion models have recently become powerful tools for generating high-quality tabular data. However, they are susceptible to memorization, often producing outputs that closely resemble the training
data and thereby risking the leakage of sensitive information. Existing differential privacy (DP)-based methods mitigate memorization but often severely disrupt inter-attribute correlations in tabular
data, significantly degrading the quality of the synthesized data.
To address these limitations, we propose Tame, a novel Tabular data synthesis framework designed to generate memorization-free
outputs while preserving data utility in diffusion models. To ensure consistency between categorical and numerical attributes, Tame incorporates a structured anti-memory denoising mechanism, which
injects noise adaptively calibrated to the heterogeneous distribution characteristics of different attribute types. It then employs tabular
attribute alignment to preserve inter-attribute correlations using an attribute predictor. Extensive experiments comparing two state-of-the-art tabular diffusion models (i.e., TabSyn and TabDDPM) across four widely-used datasets (i.e., Shoppers, Adult, Default, and Cardio) demonstrate that Tame effectively reduces memorization while maintaining high-quality data generation.

## Performance
<div align="center">
  <figure style="text-align: center;">
    <img src="heatmap.png" alt="performance-table" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
    <figcaption>Figure 2: Heatmaps of the pair-wise column correlation of synthetic data v.s. the real data. </figcaption>
  </figure>

  <figure style="text-align: center;">
    <img src="assets/distribution.png" alt="performance-table" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
    <figcaption>Figure 3: Visualization of synthetic data’s single column distribution density v.s. the real data. </figcaption>
  </figure>
  <br>
</div>

## Installing Dependencies

Python version: 3.10

Create environment

```
conda create -n tame python=3.10
conda activate tame
```

Install pytorch
```
pip install torch torchvision torchaudio
```

Install other dependencies

```
pip install -r requirements.txt
```


Create another environment for the quality metric (package "synthcity")

```
conda create -n synthcity python=3.10
conda activate synthcity

pip install synthcity
pip install category_encoders
```

## Training Models

For tabDDPM, use the following command for training:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode train
```

Options of [NAME_OF_DATASET]: adult, default, shoppers, cardio_train
Options of [NAME_OF_BASELINE_METHODS]: tabddpm

For TabSyn, use the following command for training:

```
# train VAE first
python main.py --dataname [NAME_OF_DATASET] --method vae --mode train

# after the VAE is trained, train the diffusion model
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode train
```

## Tabular Data Synthesis

For baseline methods, use the following command for synthesis:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode sample --save_path [PATH_TO_SAVE]
```

For Tabsyn, use the following command for synthesis:

```
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode sample --save_path [PATH_TO_SAVE]
```

The default save path is "synthetic/[NAME_OF_DATASET]/[METHOD_NAME].csv"

## Memorization Ratio

```
python cal_memorization.py
```

## Evaluation
We evaluate the quality of synthetic data using metrics from various aspects.

```
python -m eval.eval_all
```


#### Alpha Precision and Beta Recall ([paper link](https://arxiv.org/abs/2102.08921))
- $\alpha$-preicison: the fidelity of synthetic data
- $\beta$-recall: the diversity of synthetic data

```
python eval/eval_quality.py
```

## Acknowledgements

This project was built upon code from [TabSyn](https://github.com/amazon-science/tabsyn). We are deeply grateful for their open-source contributions, which have significantly helped shape the development of this project.

Specifically, many of the model components in this repository are based on the foundation provided by [TabSyn](https://github.com/amazon-science/tabsyn). We highly recommend checking out their work for further insights.
