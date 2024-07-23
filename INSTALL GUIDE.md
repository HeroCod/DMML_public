# Setting Up the Environment for Data Mining and Machine Learning (DMML)

This guide will walk you through the steps of setting up a conda environment for DMML, along with CUDA installation and necessary Python packages.

It is advised to use the vscode-netron extension to allow for inspection of the .keras model structures and the Live Preview to view graphs live.

Tested Keras and Tensorflow versions are: keras-3.4.1 , tensorflow-2.17.0

## Step 1: Download and Install Anaconda

First, download the Anaconda installer script and run it.

```sh
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh
```

## Step 2: Initialize Conda

Initialize conda so that you can use it right away.

```sh
conda init
```

## Step 3: Update Conda

Update conda to its latest version.

```sh
conda update -n base conda
```

## Step 4: Install and Configure libmamba Solver (Optional but Recommended)

Install the `conda-libmamba-solver` for faster dependency resolution and then set it as the default solver.

```sh
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## Step 5: Install CUDA for GPU Acceleration

While you can probably use different versions of CUDA, the system has only been tested on ubuntu with this version of cuda.

1. Download the CUDA keyring.

    ```sh
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    ```

2. Install the CUDA keyring.

    ```sh
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    ```

3. Update the package lists.

    ```sh
    sudo apt-get update
    ```

4. Install CUDA.

    ```sh
    sudo apt-get -y install cuda
    ```

## Step 6: Create the DMML Conda Environment

Create a new conda environment named `DMML` with the necessary packages.

```sh
conda create -n DMML -c rapidsai -c conda-forge -c nvidia \
    cudf=24.06 cuml=24.06 cugraph=24.06 cuxfilter=24.06 cucim=24.06 cuspatial=24.06 \
    cuproj=24.06 pylibraft=24.06 raft-dask=24.06 cuvs=24.06 python=3.11 cuda-version=12.0 \
    pytorch xarray-spatial tensorflow graphistry dash jupyterlab dask-sql ipywidgets
```

## Step 7: Activate the DMML Environment

Activate the newly created environment.

```sh
conda activate DMML
```

Optionally, you can stop conda from automatically activating the base environment.

```sh
conda config --set auto_activate_base false
```

Finally, set the DMML environment as the default one used in your VSCode instance and relaunch it.

```md
Ctrl+Shift+P
python: Select interpreter
```


## Step 8: Install Additional Python Packages

Use pip to install additional Python packages that are commonly used in data mining and machine learning.

```sh
pip install python-dotenv seaborn xgboost pandas scikit-learn plotly tqdm matplotlib dask distributed pmdarima prophet statsmodels ipywidgets skforecast kaleido
```

Also, some pip-installed packages may not be necessary depending on your specific needs, and others may not be listed here. 
Be prepared to install additional packages as needed, as the code may vary and this file may not be perfectly updated at all times.

---

By following these steps, you will set up the environment necessary for the project execution.


# Copyright Notice

The content within this repository, including but not limited to all source code files, documentation files, and any other files contained herein, is the intellectual property of Francesco Boldrini, or, when applicable, the respective owners of publicly available libraries or code.

Unauthorized reproduction, distribution, or usage of any content from this repository is strictly prohibited unless express written consent has been granted by Francesco Boldrini or the respective content owners, where applicable. Such consent must be documented through a formal, signed contract.

This code is also permitted for use solely in the context of the examination process at the University of Pisa. Post-examination, any copies of the code must be irretrievably destroyed.

In the event that you have come into possession of any code from this repository without proper authorization, you are required to contact Francesco Boldrini at the address provided below, as the code may have been obtained through unlawful distribution.

Copyright © 2024, Francesco Boldrini, All rights reserved.

For any commercial inquiries, please contact:

Francesco Boldrini
Email: commercial@francesco-boldrini.com
