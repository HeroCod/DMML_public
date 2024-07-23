#!/bin/bash

# Filename to store permissions
PERMISSIONS_FILE="./permissions_backup.txt"
TARGET_DIRECTORY="../datasets/originals"

# Check if target directory exists
if [ ! -d "$TARGET_DIRECTORY" ]; then
    echo "Target directory '$TARGET_DIRECTORY' does not exist."
    exit 1
fi

# Function to make files read-only and store their current permissions
make_files_read_only() {
    # Clear the permissions file if it exists
    > "$PERMISSIONS_FILE"
    
    # Iterate over each file in the directory
    find "$TARGET_DIRECTORY" -type f | while read -r FILE; do
        # Get the current permissions
        PERMS=$(stat -c "%a" "$FILE")
        
        # Save the current permissions
        echo "$FILE:$PERMS" >> "$PERMISSIONS_FILE"
        
        # Change the file permission to read-only
        chmod 444 "$FILE"
    done
    
    echo "All files in '$TARGET_DIRECTORY' have been made read-only."
}

# Function to restore files' original permissions
restore_files_permissions() {
    if [ ! -f "$PERMISSIONS_FILE" ]; then
        echo "No previous permissions file found. Cannot restore."
        exit 1
    fi

    while IFS=: read -r FILE PERMS; do
        if [ -e "$FILE" ]; then
            chmod "$PERMS" "$FILE"
        else
            echo "File '$FILE' no longer exists."
        fi
    done < "$PERMISSIONS_FILE"
    
    # Optionally clean up the permissions file
    # rm "$PERMISSIONS_FILE"
    
    echo "Previous permissions restored for all files in '$TARGET_DIRECTORY'."
}

# Check if the permissions file exists to determine new state
if [ -f "$PERMISSIONS_FILE" ]; then
    restore_files_permissions
    # Optionally remove the permissions file after restoration
    rm "$PERMISSIONS_FILE"
else
    make_files_read_only
fi

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