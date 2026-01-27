#!/bin/bash

../cmake-build-release/ParallelImg --threads 1 --output big_dataset.csv --d ../dataset_1024x1024 --type Gaussian
../cmake-build-release/ParallelImg --threads 1 --output big_dataset.csv --d ../dataset_1024x1024 --type Gaussian7
../cmake-build-release/ParallelImg --threads 8 --output big_dataset.csv --d ../dataset_1024x1024 --type Gaussian
../cmake-build-release/ParallelImg --threads 8 --output big_dataset.csv --d ../dataset_1024x1024 --type Gaussian7
