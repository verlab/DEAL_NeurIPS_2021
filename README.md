[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/DEAL_NeurIPS_2021/blob/main/notebook/DEAL.ipynb)
# DEAL - Deformation-Aware Local Features
## <b>Extracting Deformation-Aware Local Features by Learning to Deform</b> <br>[[Project Page]](https://www.verlab.dcc.ufmg.br/descriptors/neurips2021/) [[Paper (Coming soon)]]() [[Container (Coming soon)]]()

<img src='./images/paper_thumbnail.png' align="center" width=900 />

This repository contains the original implementation of the descriptor "<b>Extracting Deformation-Aware Local Features byLearning to Deform</b>", to be presented at NeurIPS 2021. 


If you find this code useful for your research, please cite the paper:
```

```

## I - Ready to Use Container

`Coming soon...` 

## II - Colab Notebook 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/DEAL_NeurIPS_2021/blob/main/notebook/DEAL.ipynb)

## III - Local Installation Alternative

```
conda env create --file ./environment.yml
conda activate deal
```

- Testing
zz
```bash
python example.py
```

## IV - Training the model

The file (run.py)[run.py] contains the training code for the SOTA model and some optios for ablation studies.
To train the model you will need to download our preprocessed dataset at [nonrigid-data (88GB)](). Save the dataset on the folder `data`, at the root of this repository.

To train the SOTA model we used the command:
```
python run.py --mode train --datapath data/train-big.h5 --dataset nonrigid
```

The training process consumes about 11GB of memory of the GPU. On a GTX1080Ti it took about XX hours to complete the training.


**VeRLab:** Laboratory of Computer Vison and Robotics https://www.verlab.dcc.ufmg.br
<br>
<img align="left" width="auto" height="75" src="./images/ufmg.png">
<img align="right" width="auto" height="75" src="./images/verlab.png">
<br/>