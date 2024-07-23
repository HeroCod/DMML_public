#!/bin/bash

ENV_NAME="DMML"
CONDA_PATH="$HOME/anaconda3"
ENV_YAML=".vscode/environment.yml"
SETTINGS_JSON=".vscode/settings.json"

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda."
    exit
fi

# Check if the environment exists
if conda env list | grep -q $ENV_NAME
then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' from $ENV_YAML..."
    conda env create -f $ENV_YAML
fi

# Find the Python interpreter path
PYTHON_PATH="$($CONDA_PATH/bin/conda info --base)/envs/$ENV_NAME/bin/python"

# Create or update settings.json
mkdir -p .vscode
cat > $SETTINGS_JSON <<EOL
{
    "python.defaultInterpreterPath": "$PYTHON_PATH"
}
EOL

echo "VSCode configuration updated to use the '$ENV_NAME' environment."

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