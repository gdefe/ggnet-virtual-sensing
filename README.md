# GgNet
Official repository for the paper "Graph-based Virtual Sensing from Sparse and Partial Multivariate Observations" (ICLR 2024)


Guidelines for executing the code.
tested with Python 3.10 on Ubuntu 22.04.3 LTS


1) install dependencies:
'''
conda update conda
conda env create -f conda_env_linux.yml  
conda activate ggnet
'''


2) install torch spatiotemporal
- open a terminal in the directory where the 'README.txt' is located
'''
git clone https://github.com/TorchSpatiotemporal/tsl.git
cd tsl
pip install -e .
'''


3) create datasets
- daily climate:  
        - comment\uncomment the relevant lines of code in the main of 'data/NASA_data/build_dataset.py'
		- run:   'python data/NASA_data/build_dataset.py'      This will download data from the API and save it to data/NASA_data/clmDaily.pkl (may take some time to complete)  

- hourly climate: 
        - comment\uncomment the relevant lines of code in the main of 'data/NASA_data/build_dataset.py'
        - run:   'python data/NASA_data/build_dataset.py'      This will download data from the API and save it to data/NASA_data/clmHourly.pkl (may take some time to complete)

- photovoltaic:    
        - one dataset with 100 nodes is provided in the code. 
        - to create a datasets at arbitrary resolution:
            - install xarray: 'pip install xarray'
            - download coordinates.nc and module_00.tar.gz (~86 Gb) from https://scholarsphere.psu.edu/resources/dacba268-d084-4e0e-a674-670217c59891
            - extract module_00.tar.gz
            - place both inside the 'data/Photovoltaic' directory
            - modify and run: 'data/Photovoltaic/build_dataset.py'


4) choose model and dataset in 'config.yaml'

  
5) choose setting, e.g., number of epochs, in 'default.yaml'


6) run the code:  'python run_train.py config=config'
