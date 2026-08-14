#!/usr/bin/env bash

# Install ACPO as editable python package
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." && pip install -e .

# Install specific protobuf version to fix ray 1.8 protobuf bug
# TODO - remove when upgrading dependencies
# BUG - tensorboardx probably not working as expected due to this temporary fix
pip install protobuf==3.20

# Install dependencies for test suite
pip install pytest pytest-cov