import pyarrow.parquet as pq
import pyarrow as pa

def merge_parquet_files_incrementally(
    original_file: str, 
    metadata_file: str, 
    output_file: str,
    chunk_size: int = 1000
):
    """Merge original and metadata Parquet files incrementally."""
    
    # Open the original and metadata Parquet files
    original_reader = pq.ParquetFile(original_file)
    metadata_reader = pq.ParquetFile(metadata_file)

    # Create a writer for the merged Parquet file
    output_writer = None

    try:
        for original_batch in original_reader.iter_batches(batch_size=chunk_size):
            # Convert original batch to a table
            original_table = pa.Table.from_batches([original_batch])
            
            # Extract a list of keys (e.g., filenames) to match
            original_keys = original_table["filename"].to_pylist()
            
            # Filter metadata batches that match these keys
            matching_metadata = []
            for metadata_batch in metadata_reader.iter_batches(batch_size=chunk_size):
                metadata_table = pa.Table.from_batches([metadata_batch])
                metadata_keys = metadata_table["filename"].to_pylist()
                
                # Filter rows where metadata keys match original keys
                mask = [key in original_keys for key in metadata_keys]
                filtered_metadata = metadata_table.filter(pa.array(mask))
                matching_metadata.append(filtered_metadata)
            
            # Concatenate all matching metadata
            if matching_metadata:
                metadata_table = pa.concat_tables(matching_metadata)
                
                # Merge original and metadata tables
                merged_table = original_table.join(
                    metadata_table, keys="filename", join_type="left"
                )
            else:
                # If no matching metadata, use original table as-is
                merged_table = original_table

            # Write to the output file
            if output_writer is None:
                output_writer = pq.ParquetWriter(output_file, merged_table.schema)
            
            output_writer.write_table(merged_table)

    finally:
        # Close the writer
        if output_writer is not None:
            output_writer.close()