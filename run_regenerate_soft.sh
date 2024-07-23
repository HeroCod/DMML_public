# executes in order all the python tasks
#!/bin/bash

# create initial datasets forom original datasets (with exogenous variables)
python3 code/create_initial_datasets.py

# start preliminary analisys (basic statistical analysis)
python3 code/preliminary_analysis.py

# check for trends, seasonality, residuals, etc
python3 code/trends_seasonality_residuals.py

# start spinning up the models:

# SARIMA
python3 code/adjust_sarima.py # this adds possible missing graphs, does not generate new models

# SARIMAX
python3 code/adjust_sarimax.py # this adds possible missing graphs, does not generate new models

# Prophet
python3 code/fb_prophet.py

# Algorithmic approact currently used by the company
python3 code/algorithmic_approach.py

# LSTM
# ensure numa nodes are connected
sudo echo 0 | sudo tee -a /sys/bus/pci/devices/0000\:01\:00.0/numa_node

python3 code/lstm.py # This assumes lstm has been run before

# Compare all models
python3 code/compare_models.py

# Quit the program
exit 0

#
# Copyright Notice - DO NOT REMOVE OR ALTERATE
#
# The content within this repository, including but not limited to all source code files, documentation files, and any other files contained herein, is the intellectual property of Francesco Boldrini, or, when applicable, the respective owners of publicly available libraries or code.
#
# Unauthorized reproduction, distribution, or usage of any content from this repository is strictly prohibited unless express written consent has been granted by Francesco Boldrini or the respective content owners, where applicable. Such consent must be documented through a formal, signed contract.
#
# This code is also permitted for use solely in the context of the examination process at the University of Pisa. Post-examination, any copies of the code must be irretrievably destroyed.
#
# In the event that you have come into possession of any code from this repository without proper authorization, you are required to contact Francesco Boldrini at the address provided below, as the code may have been obtained through unlawful distribution.
#
# Copyright © 2024, Francesco Boldrini, All rights reserved.
#
# For any commercial inquiries, please contact:
#
# Francesco Boldrini
# Email: commercial@francesco-boldrini.com
#