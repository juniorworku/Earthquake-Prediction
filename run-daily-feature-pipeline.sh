#!/bin/bash

# run-daily-feature-pipeline.sh

# Exit if any command fails
set -e

# Check if HOPSWORKS_API_KEY is set
if [ -z "$HOPSWORKS_API_KEY" ]; then
  echo "Error: HOPSWORKS_API_KEY is not set."
  exit 1
fi

echo "Starting daily feature pipeline..."

# Export Hopsworks API key as an environment variable
export HOPSWORKS_API_KEY=$HOPSWORKS_API_KEY

# Run the daily feature pipeline Python script
python ./daily-feature-pipeline.py --api_key $HOPSWORKS_API_KEY

echo "Daily feature pipeline completed successfully."





