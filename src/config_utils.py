""" 
This module contains utility functions for parsing command line arguments, setting up logging, and creating configuration objects for the training pipeline.
"""

import logging
import os
import sys
import optuna

from src.settings import AppSettings


def setup_logging():
    """Set up logging configuration."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])


def make_config(args):
    """Create configuration object from command line arguments."""
    config = AppSettings.from_yaml(args.config)

    # Update paths if provided via command line arguments
    if args.input_dir is not None:
        config.path.input_dir = args.input_dir

    if args.output_dir is not None:
        config.path.output_dir = args.output_dir

    os.makedirs(config.path.output_dir, exist_ok=True)
    
    return config