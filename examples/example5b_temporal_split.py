# Example: Adding temporal train/val/test split to your dataset

# This example shows how to use add_temporal_set_column() to split your dataset
# based on the temporal information (year) from the eventDate field.

# Key features:
# - Most recent images → test set
# - Oldest images → train set
# - Middle images → validation set
# - Unknown dates → train set (default)
# - Maintains OOD (out-of-distribution) logic for rare species

# Use case: This temporal split is useful for evaluating model performance on
# more recent data, which is closer to real-world deployment scenarios.

from gbifxdl import add_temporal_set_column

# Example 1: Basic usage with default 80/10/10 split
# This will create train (80%), val (10%), and test (10%) splits
# based on the temporal order of the eventDate column
# This only adds a "set" column in the parquet file,
# you still need to write a script to move/copy the images
# to separate folders based on the "set" column.
output_path = add_temporal_set_column(
    parquet_path="data/mini/0013397-241007104925546_processing_metadata_postprocessed.parquet",
    split_ratios=[0.8, 0.1, 0.1],  # train, val, test
    ood_th=5,  # Species with ≤5 images go to "test_ood"
    species_column="speciesKey",
    eventdate_column="eventDate",
    seed=42,
)