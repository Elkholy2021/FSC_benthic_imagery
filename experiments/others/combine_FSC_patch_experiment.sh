#!/usr/bin/env bash

set -euo pipefail

# Check input
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <input_directory>"
    exit 1
fi

INPUT_DIR="$(realpath "$1")"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Directory does not exist: $INPUT_DIR"
    exit 1
fi

# Find the first directory that is not 'figures'
FIRST_PARENT=""
for dir in "$INPUT_DIR"/*; do
    [[ -d "$dir" ]] || continue

    name="$(basename "$dir")"

    # Skip figures
    [[ "$name" == "figures" ]] && continue

    FIRST_PARENT="$name"
    break
done

if [[ -z "$FIRST_PARENT" ]]; then
    echo "Error: No suitable input folders found."
    exit 1
fi

# Extract dataset ID.
# Example:
#   16817_0_01 -> 16817
DATASET_ID="${FIRST_PARENT%%_*}"

OUTPUT_DIR="$INPUT_DIR/$DATASET_ID"

mkdir -p "$OUTPUT_DIR"

echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo

# Process parent directories
for parent_path in "$INPUT_DIR"/*; do

    [[ -d "$parent_path" ]] || continue

    parent_name="$(basename "$parent_path")"

    # Skip figures
    [[ "$parent_name" == "figures" ]] && continue

    # Skip output directory
    [[ "$parent_path" == "$OUTPUT_DIR" ]] && continue

    # Only process directories belonging to this dataset
    [[ "$parent_name" == "${DATASET_ID}_"* ]] || continue

    # Process subfolders
    for child_path in "$parent_path"/*; do

        [[ -d "$child_path" ]] || continue

        child_name="$(basename "$child_path")"

        new_name="${parent_name}_${child_name}"
        destination="$OUTPUT_DIR/$new_name"

        echo "Copying:"
        echo "  $parent_name/$child_name"
        echo "  -> $DATASET_ID/$new_name"

        cp -a "$child_path" "$destination"
    done
done

echo
echo "Done."