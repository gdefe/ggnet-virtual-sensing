# Graph-based Virtual Sensing from Sparse and Partial Multivariate Observations (ICLR 2024)

[![ICLR](https://img.shields.io/badge/ICLR-2022-blue.svg?style=flat-square)](https://openreview.net/forum?id=CAqdG2dy5s)
[![PDF](https://img.shields.io/badge/%E2%87%A9-PDF-orange.svg?style=flat-square)](https://openreview.net/pdf?id=CAqdG2dy5s)
[![arXiv](https://img.shields.io/badge/arXiv-2402.12598-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2402.12598)

**Authors**: [Giovanni De Felice](mailto:gdefe@liverpool.ac.uk), Andrea Cini, Daniele Zambon, Vladimir Gusev, Cesare Alippi

Code and official repository for the paper "Graph-based Virtual Sensing from Sparse and Partial Multivariate Observations". In this paper, we propose a graph-based methodology for tackling virtual sensing in a sparse and multivariate setting.
The present code implementation is based on [Torch Spatiotemporal](https://github.com/TorchSpatiotemporal/tsl), a library built to accelerate research on neural spatiotemporal data processing methods.




Guidelines for executing the code.
(Tested with Python 3.10 on Ubuntu 22.04.3 LTS)


1) install dependencies:
```
conda update conda
conda env create -f conda_env_linux.yml  
conda activate ggnet
```


2) install torch spatiotemporal

open a terminal in the directory where the 'README.txt' is located
```
git clone https://github.com/TorchSpatiotemporal/tsl.git
cd tsl
pip install -e .
```


3) create datasets:
- climate: use the script `data/NASA_data/build_dataset.py`. This will download data from the API. Then save the output into `data/NASA_data/clmDaily.pkl` or `data/NASA_data/clmHourly.pkl` (may take some time to complete)
  
- photovoltaic: First, install install xarray: `pip install xarray`. Download `coordinates.nc` and `module_00.tar.gz` (~86 Gb) from https://scholarsphere.psu.edu/resources/dacba268-d084-4e0e-a674-670217c59891 and place both into the `data/Photovoltaic` folder. Finally, modify and use the script `data/Photovoltaic/build_dataset.py`.
One dataset with 50 nodes is provided in the code. 

- feel free to contact me for support or requesting built datasets


4) choose model and dataset in `config.yaml`

  
5) choose setting, e.g., number of epochs, in `default.yaml`

(optional) to enable logging, uncomment and personalize the wandb configurations into `default.yaml`

6) run the code:  `python run.py config=config`
