"""Where the tests find the geez-dataset checkout.

    python -m pytest tests/ --dataset_dir /path/to/geez-dataset

Without --dataset_dir: $GEEZ_DATASET_DIR, else ../geez-dataset next to this repo (the same rule as the
scripts). Tests that need a checkout are skipped when there is none at that place.
Test files read the folder as `from conftest import DATASET_DIR`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset_loader  # noqa: E402

DATASET_DIR = None   # set in pytest_configure, before the test files are imported


def pytest_addoption(parser):
    parser.addoption("--dataset_dir", default=None,
                     help="geez-dataset checkout for the tests that need one "
                          "(default: $GEEZ_DATASET_DIR, else ../geez-dataset)")


def pytest_configure(config):
    global DATASET_DIR
    DATASET_DIR = os.path.abspath(config.getoption("dataset_dir") or dataset_loader.default_dataset_dir())
