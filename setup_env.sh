#!/bin/bash
# Setup environment for FraudLens with all required libraries

# Set library paths for zbar (QR code detection)
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"

# Run the command passed as arguments
"$@"