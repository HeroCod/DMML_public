import re
import os

def process_input(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Step 1: Ensure everything between \( \) and \[ \] is on the same line
    content = re.sub(r'(\\\()[\s\S]*?(\\\))', lambda match: match.group(0).replace('\n', ' '), content)
    content = re.sub(r'(\\\[)[\s\S]*?(\\\])', lambda match: match.group(0).replace('\n', ' '), content)

    # Step 2: Substitute all '\\' <newline> with newline without the actual carriage return
    content = re.sub(r'\\\\\n', r'\\\n', content)
    # Step 2.1: Substitute all leftovers '\\' with newline
    content = re.sub(r'\\\\', r'\\newline', content)

    # Step 3: Substitute '\(' with ' \[ ' and '\)' with ' \] '
    content = content.replace('\\(', ' \[ ').replace('\\)', ' \] ')

    # Step 4: Write the modified content to output.txt
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(content)

# Use the function
current_dir = os.path.dirname(os.path.realpath(__file__))
input = os.path.join(current_dir, 'input.txt')
output = os.path.join(current_dir, 'output.txt')
process_input(input, output)

print("Content modified and written to output.txt")

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