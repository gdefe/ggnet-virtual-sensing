# GgNet
Official repository for the paper "Graph-based Virtual Sensing from Sparse and Partial Multivariate Observations" (ICLR 2024)


Guidelines for executing the code.
tested with Python 3.10 on Ubuntu 22.04.3 LTS


1) install dependencies:
- you may hve to update you conda installation: conda update conda
- conda env create -f conda_env_linux.yml  (may take some time to complete)


2) install torch spatiotemporal
- open a terminal in the directory where the 'README.txt' is located
- git clone https://github.com/TorchSpatiotemporal/tsl.git
- cd tsl
- pip install -e .


3) activate the environment: conda activate virtual_sensing 


4) check for the dataset 
- daily climate:  comment\uncomment the relevant lines of code in 'data/NASA_data/build_dataset.py'
		  run:   python data/NASA_data/build_dataset.py      This will download data from the API and save it to data/NASA_data/clmDaily.pkl (may take some time to complete)  
- hourly climate: comment\uncomment the relevant lines of code in 'data/NASA_data/build_dataset.py'
          run:   python data/NASA_data/build_dataset.py      This will download data from the API and save it to data/NASA_data/clmHourly.pkl (may take some time to complete)
- photovoltaic: datasets are already built


5) choose model and dataset in 'config.yaml'

  
6) choose setting, e.g., number of epochs, in 'default.yaml'


7) run the code:  python run_train.py config=config
average testing accuracy will be printed on screen after the training is complete
