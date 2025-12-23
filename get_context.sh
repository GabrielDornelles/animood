#!/bin/sh

SRC_DIR="src"

find "$SRC_DIR" -type f -name "*.rs" | sort | while read -r file; do
    echo "=============================="
    echo "FILE: $file"
    echo "=============================="
    echo
    cat "$file"
    echo
    echo
done
