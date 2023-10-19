# Application on real data of higher-order info metrics
This repository contains some analyzes done on real data to test some metrics developed [here](https://github.com/nplresearch/higher_order_info_metrics).

## What's inside?
There are three main folders: **data** contains the time series, **ho_info_metrics** contains the functions that calculate the metrics and **results** contains all the results (**results_backup** contains the same files, but these cannot be modified by .ipynb files). Finally there are notebooks, one for each type of data.

## Brain data
Contains data from 98 subjects. Each data is randomized while keeping the covariance intact, following two frameworks (see [here](https://www.sciencedirect.com/science/article/pii/S1053811917307516?casa_token=mF5NmQHxgPkAAAAA:w9DrZz5fNviq0-LawoSevS-CnTDtrgeVsA_tlxGieB31LBsHDXF-crkromwouJdmGzrotS5pOQ)). The metrics calculated on the original data and those obtained from randomization are then compared.

## Economic data
Data taken [here](https://zenodo.org/records/7210076) (called Financial_data). Contains the daily market performance of various companies belonging to certain sectors. Triplets built by companies in the same sector are compared with triplets in which one or more companies belong to different sectors. Furthermore, the value of these metrics is compared between different temporal lengths.
