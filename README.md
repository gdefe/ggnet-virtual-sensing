# GgNet
Official repository for the paper "Graph-based Virtual Sensing from Sparse and Partial Multivariate Observations" (ICLR 2024)


Guidelines for executing the code.
tested with Python 3.10 on Ubuntu 22.04.3 LTS


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

6) run the code:  `python run_train.py config=config`
