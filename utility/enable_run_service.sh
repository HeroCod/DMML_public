#!/bin/bash

# Reload systemd manager configuration
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable DMML_UniPi.service

# Start the service immediately
sudo systemctl start DMML_UniPi.service

# Check the status of the service
sudo systemctl status DMML_UniPi.service

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