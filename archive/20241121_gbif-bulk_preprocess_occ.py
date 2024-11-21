
def preprocess_occurrences_dask(config, occurrences_path: Path):
    """Stream DWCA directly to a Dask DataFrame on disk"""
    assert occurrences_path is not None, "No occurrence path provided"

    print("Streaming DWCA to Dask DataFrame...")
    
    if config['format'].lower() != "dwca":
        raise ValueError(f"Unknown format: {config['format'].lower()}")

    def process_dwca():
        with DwCAReader(occurrences_path) as dwca:
            for row in dwca:
                extensions = row.extensions[:-1]
                
                for e in extensions:
                    identifier = e.data['http://purl.org/dc/terms/identifier']
                    
                    if identifier:
                        row_data = {}
                        
                        for k, v in row.data.items():
                            k = k.split('/')[-1]
                            if k in KEYS_OCC + KEYS_GBIF:
                                row_data[k] = v

                        for k, v in e.data.items():
                            k = k.split('/')[-1]
                            if k in KEYS_MULT:
                                row_data[k] = v
                        
                        yield row_data

    # Create Dask DataFrame directly from generator
    # ERROR: TypeError: object of type 'generator' has no len()
    df = dd.from_pandas(dd.from_delayed(process_dwca()), npartitions=10)

    # Preprocessing steps
    df = df.dropna(subset=KEYS_GBIF)
    df = df[~df[KEYS_GBIF].eq('').any(axis=1)]
    
    if config.get('drop_duplicates', False):
        df = df.drop_duplicates(subset='identifier')

    if config.get('max_img_spc', 0) > 1:
        df = df.groupby('taxonKey').apply(
            lambda x: x[x.groupby('taxonKey').cumcount() < config['max_img_spc']], 
            meta=df
        )

    # Final output path
    output_path = occurrences_path.with_suffix(".parquet")
    df.to_parquet(output_path, engine='pyarrow', compression='gzip')

    print(f"Preprocessing done. Processed file stored in {output_path}")

    return output_path

def preprocess_occurrences_streaming(config, occurrences_path: Path):
    """
    Stream DWCA directly to a Dask DataFrame on disk
    
    Args:
        config (dict): Configuration dictionary
        occurrences_path (Path): Path to the DWCA file
    
    Returns:
        Path: Path to the processed Parquet file
    """
    assert occurrences_path is not None, "No occurrence path provided"

    print("Streaming DWCA to Dask DataFrame...")
    
    if config['format'].lower() != "dwca":
        raise ValueError(f"Unknown format: {config['format'].lower()}")

    # Temporary directory for chunked Parquet files
    temp_dir = occurrences_path.parent / "temp_dask_chunks"
    temp_dir.mkdir(exist_ok=True)

    # Chunk counter
    chunk_num = 0

    with DwCAReader(occurrences_path) as dwca:
        chunk_data = []
        
        for row in dwca:
            extensions = row.extensions[:-1]
            
            for e in extensions:
                identifier = e.data['http://purl.org/dc/terms/identifier']
                
                if identifier:
                    # Prepare row data
                    row_data = {}
                    
                    # Add occurrence metadata
                    for k, v in row.data.items():
                        k = k.split('/')[-1]
                        if k in KEYS_OCC + KEYS_GBIF:
                            row_data[k] = v

                    # Add extension metadata
                    for k, v in e.data.items():
                        k = k.split('/')[-1]
                        if k in KEYS_MULT:
                            row_data[k] = v
                    
                    chunk_data.append(row_data)
                    
                    # Write chunk to disk when it reaches a certain size
                    if len(chunk_data) >= 10000:
                        temp_chunk_path = temp_dir / f"chunk_{chunk_num}.parquet"
                        pd.DataFrame(chunk_data).to_parquet(temp_chunk_path)
                        chunk_data = []
                        chunk_num += 1

        # Write any remaining data
        if chunk_data:
            temp_chunk_path = temp_dir / f"chunk_{chunk_num}.parquet"
            pd.DataFrame(chunk_data).to_parquet(temp_chunk_path)

    # Read chunks into Dask DataFrame
    df = dd.read_parquet(str(temp_dir / "*.parquet"))

    # Preprocessing steps
    df = df.dropna(subset=KEYS_GBIF)
    df = df[~df[KEYS_GBIF].eq('').any(axis=1)]
    
    if config.get('drop_duplicates', False):
        df = df.drop_duplicates(subset='identifier')

    if config.get('max_img_spc', 0) > 1:
        df = df.groupby('taxonKey').apply(
            lambda x: x[x.groupby('taxonKey').cumcount() < config['max_img_spc']], 
            meta=df
        )

    # Final output path
    output_path = occurrences_path.with_suffix(".dparquet")
    df.to_parquet(output_path, engine='pyarrow', compression='gzip', overwrite=True)

    # Optional: clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)

    print(f"Preprocessing done. Processed file stored in {output_path}")

    return output_path

def preprocess_occurrences_streaming_2(config, occurrences_path: Path):
    """Prepare the download file - remove duplicates, limit the number of downloads per species, remove the columns we don't need, etc.
    This version uses Dask to handle large datasets efficiently, storing chunks locally.
    """

    assert occurrences_path is not None, "No occurrence path provided, please provide one."

    print("Preprocessing the occurrence file before download...")

    if config['format'].lower() == "dwca":
        # Step 1: Initialize variables for chunking
        chunk_size = 10_000  # Number of rows per chunk
        current_chunk = []
        dask_chunks = []
        chunk_counter = 0

        # Local directory for temporary Parquet files
        local_storage_dir = occurrences_path.parent / "temp_dask_chunks"
        if not os.path.exists(local_storage_dir):
            os.makedirs(local_storage_dir)

        # Read the DWCA file and process rows in chunks
        with DwCAReader(occurrences_path) as dwca:
            for row in dwca:
                extensions = row.extensions[:-1]  # Ignore verbatim extension

                # Process each multimedia extension
                for e in extensions:
                    identifier = e.data['http://purl.org/dc/terms/identifier']

                    if identifier != '':
                        # Add occurrence metadata (identical for all multimedia)
                        row_data = {}
                        for k, v in row.data.items():
                            k = k.split('/')[-1]
                            if k in KEYS_OCC + KEYS_GBIF:
                                row_data[k] = v

                        # Add multimedia extension metadata
                        for k, v in e.data.items():
                            k = k.split('/')[-1]
                            if k in KEYS_MULT:
                                row_data[k] = v

                        current_chunk.append(row_data)

                        # Once we've collected enough rows, save a chunk
                        if len(current_chunk) >= chunk_size:
                            chunk_file = os.path.join(local_storage_dir, f"chunk_{chunk_counter}.parquet")

                            # Convert the chunk to a Pandas DataFrame
                            chunk_df = pd.DataFrame(current_chunk)

                            # Wrap the operation in a Delayed object
                            delayed_task = delayed(save_chunk_as_parquet)(chunk_df, chunk_file)
                            dask_chunks.append(delayed_task)

                            # Reset the current chunk and increment the counter
                            current_chunk = []
                            chunk_counter += 1

            # Save any remaining rows in the last chunk
            if current_chunk:
                chunk_file = os.path.join(local_storage_dir, f"chunk_{chunk_counter}.parquet")
                chunk_df = pd.DataFrame(current_chunk)
                delayed_task = delayed(save_chunk_as_parquet)(chunk_df, chunk_file)
                dask_chunks.append(delayed_task)

        # Step 2: Combine all chunks into a single Dask DataFrame
        # Delayed tasks are executed only when triggered
        full_dask_df = dd.from_delayed([delayed(load_chunk_as_dask_df)(f) for f in os.listdir(local_storage_dir)])

        # Step 3: Apply processing on the full Dask DataFrame
        if 'drop_duplicates' in config.keys() and config['drop_duplicates']:
            full_dask_df = full_dask_df.drop_duplicates(subset='identifier', keep=False)

        if 'max_img_spc' in config.keys() and config['max_img_spc'] > 1:
            full_dask_df = full_dask_df.groupby('taxonKey').filter(lambda x: len(x) <= config['max_img_spc'])

        # Step 4: Write the final Dask DataFrame to Parquet
        output_path = occurrences_path.with_suffix(".parquet")
        full_dask_df.to_parquet(output_path, engine='pyarrow', compression='gzip', write_options={'row_group_size': 50_000})

    else:
        raise ValueError(f"Unknown format: {config['format'].lower()}")

    print(f"Preprocessing done. Preprocessed file stored in {output_path}.")

    return output_path

def save_chunk_as_parquet(chunk_df, chunk_file):
    """Save a single chunk to a Parquet file."""
    chunk_df.to_parquet(chunk_file, engine='pyarrow', compression='gzip')

def load_chunk_as_dask_df(chunk_file):
    """Load a Parquet chunk into a Dask DataFrame."""
    return dd.read_parquet(chunk_file)


def preprocess_occurrences_stream_v1(config, occurrences_path: Path):
    """
    Preprocess DwCA file with complete duplicate removal
    
    All duplicates (including original) are completely removed
    """
    assert occurrences_path is not None, "No occurrence path provided"

    if config['format'].lower() != "dwca":
        raise ValueError(f"Unknown format: {config['format'].lower()}")

    # Tracking identifiers to remove completely
    duplicate_identifiers = set()
    seen_identifiers = set()

    # First pass: identify duplicates
    with DwCAReader(occurrences_path) as dwca:
        for row in dwca:
            extensions = row.extensions[:-1]
            
            for e in extensions:
                identifier = e.data['http://purl.org/dc/terms/identifier']
                
                if not identifier:
                    continue

                if identifier in seen_identifiers:
                    duplicate_identifiers.add(identifier)
                else:
                    seen_identifiers.add(identifier)

    # Prepare output Parquet writer
    output_path = occurrences_path.with_suffix(".parquet")
    parquet_writer = None
    
    # Tracking for species-level limits
    species_counts = defaultdict(int)
    max_img_per_species = config.get('max_img_spc', float('inf'))

    with DwCAReader(occurrences_path) as dwca:
        for row in dwca:
            extensions = row.extensions[:-1]

            for e in extensions:
                identifier = e.data['http://purl.org/dc/terms/identifier']
                
                # Skip if no identifier or identified as duplicate
                if not identifier or identifier in duplicate_identifiers:
                    continue

                # Extract relevant metadata
                metadata = {k.split('/')[-1]: v for k, v in row.data.items() 
                            if k.split('/')[-1] in KEYS_OCC + KEYS_GBIF}
                
                metadata.update({
                    k.split('/')[-1]: v for k, v in e.data.items() 
                    if k.split('/')[-1] in KEYS_MULT
                })

                # Skip if mandatory GBIF keys are missing
                if any(not metadata.get(key) for key in KEYS_GBIF):
                    continue

                # Check species image count limit
                taxon_key = metadata.get('taxonKey')
                species_counts[taxon_key] += 1
                
                if species_counts[taxon_key] > max_img_per_species:
                    continue

                # Write to Parquet incrementally
                record = pa.table({k: [v] for k, v in metadata.items()})
                
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_path, record.schema)
                
                parquet_writer.write_table(record)

    if parquet_writer:
        parquet_writer.close()

    return output_path

def preprocess_occurrences_stream_v2(config, occurrences_path: Path):
    """
    Efficient single-pass DwCA preprocessor with O(n) duplicate removal
    """
    assert occurrences_path is not None, "No occurrence path provided"

    if config['format'].lower() != "dwca":
        raise ValueError(f"Unknown format: {config['format'].lower()}")

    # Use a set with hash for efficient duplicate tracking
    seen_identifiers = set()
    output_path = occurrences_path.with_suffix(".parquet")
    parquet_writer = None
    
    species_counts = defaultdict(int)
    max_img_per_species = config.get('max_img_spc', float('inf'))

    with DwCAReader(occurrences_path) as dwca:
        for row in dwca:
            extensions = row.extensions[:-1]
            
            for e in extensions:
                identifier = e.data['http://purl.org/dc/terms/identifier']
                
                # Skip empty identifiers
                if not identifier:
                    continue

                # Efficient duplicate detection using hash
                identifier_hash = mmh3.hash(identifier)
                if identifier_hash in seen_identifiers:
                    continue
                seen_identifiers.add(identifier_hash)

                # Extract relevant metadata
                metadata = {k.split('/')[-1]: v for k, v in row.data.items() 
                            if k.split('/')[-1] in KEYS_OCC + KEYS_GBIF}
                
                metadata.update({
                    k.split('/')[-1]: v for k, v in e.data.items() 
                    if k.split('/')[-1] in KEYS_MULT
                })

                # Skip if mandatory GBIF keys are missing
                if any(not metadata.get(key) for key in KEYS_GBIF):
                    continue

                # Check species image count limit
                taxon_key = metadata.get('taxonKey')
                species_counts[taxon_key] += 1
                
                if species_counts[taxon_key] > max_img_per_species:
                    continue

                # Write to Parquet incrementally
                record = pa.table({k: [v] for k, v in metadata.items()})
                
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_path, record.schema)
                
                parquet_writer.write_table(record)

    if parquet_writer:
        parquet_writer.close()

    return output_path

def preprocess_occurrences_stream_v3(config, occurrences_path: Path):
    start_time = time.time()

    assert occurrences_path is not None, "No occurrence path provided"
    if config['format'].lower() != "dwca":
        raise ValueError(f"Unknown format: {config['format'].lower()}")

    seen_identifiers = set()
    species_counts = defaultdict(int)
    max_img_per_species = config.get('max_img_spc', float('inf'))

    # Collect data in chunks instead of writing row-by-row
    chunk_size = 10000
    chunk_data = defaultdict(list)
    processed_rows = 0

    output_path = occurrences_path.with_suffix(".parquet")

    with DwCAReader(occurrences_path) as dwca:
        for row in dwca:
            extensions = row.extensions[:-1]
            
            for e in extensions:
                identifier = e.data['http://purl.org/dc/terms/identifier']
                
                if not identifier:
                    continue

                identifier_hash = mmh3.hash(identifier)
                if identifier_hash in seen_identifiers:
                    continue
                seen_identifiers.add(identifier_hash)

                metadata = {k.split('/')[-1]: v for k, v in row.data.items() 
                            if k.split('/')[-1] in KEYS_OCC + KEYS_GBIF}
                
                metadata.update({
                    k.split('/')[-1]: v for k, v in e.data.items() 
                    if k.split('/')[-1] in KEYS_MULT
                })

                if any(not metadata.get(key) for key in KEYS_GBIF):
                    continue

                taxon_key = metadata.get('taxonKey')
                species_counts[taxon_key] += 1
                
                if species_counts[taxon_key] > max_img_per_species:
                    continue

                # Accumulate data in chunk
                for k, v in metadata.items():
                    chunk_data[k].append(v)
                
                processed_rows += 1

                # Write chunks periodically
                if processed_rows % chunk_size == 0:
                    chunk_table = pa.table(chunk_data)
                    
                    if processed_rows == chunk_size:
                        pq.write_table(chunk_table, output_path)
                    else:
                        pq.write_table(chunk_table, output_path, append=True)
                    
                    # Reset chunk
                    chunk_data = defaultdict(list)

        # Write final chunk if exists
        if chunk_data:
            chunk_table = pa.table(chunk_data)
            pq.write_table(chunk_table, output_path, append=True)

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    print(f"Total unique rows processed: {processed_rows}")

    return output_path

# This one is working!
def preprocess_occurrences_stream_v4(config, occurrences_path: Path):
    start_time = time.time()

    assert occurrences_path is not None, "No occurrence path provided"
    if config['format'].lower() != "dwca":
        raise ValueError(f"Unknown format: {config['format'].lower()}")

    seen_identifiers = set()
    species_counts = defaultdict(int)
    max_img_per_species = config.get('max_img_spc', float('inf'))

    chunk_size = 100
    chunk_data = defaultdict(list)
    processed_rows = 0

    output_path = occurrences_path.with_suffix(".parquet")
    parquet_writer = None

    with DwCAReader(occurrences_path) as dwca:
        for row in dwca:
            extensions = row.extensions[:-1]
            
            for e in extensions:
                identifier = e.data['http://purl.org/dc/terms/identifier']
                
                if not identifier:
                    continue

                identifier_hash = mmh3.hash(identifier)
                if identifier_hash in seen_identifiers:
                    continue
                seen_identifiers.add(identifier_hash)

                metadata = {k.split('/')[-1]: v for k, v in row.data.items() 
                            if k.split('/')[-1] in KEYS_OCC + KEYS_GBIF}
                
                metadata.update({
                    k.split('/')[-1]: v for k, v in e.data.items() 
                    if k.split('/')[-1] in KEYS_MULT
                })

                if any(not metadata.get(key) for key in KEYS_GBIF):
                    continue

                taxon_key = metadata.get('taxonKey')
                species_counts[taxon_key] += 1
                
                if species_counts[taxon_key] > max_img_per_species:
                    continue

                # Accumulate data in chunk
                for k, v in metadata.items():
                    chunk_data[k].append(v)
                
                processed_rows += 1

                # Write chunk when full
                if processed_rows % chunk_size == 0:
                    chunk_table = pa.table(chunk_data)
                    
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(output_path, chunk_table.schema)
                    
                    parquet_writer.write_table(chunk_table)
                    chunk_data = defaultdict(list)

        # Write final chunk if exists
        if chunk_data:
            chunk_table = pa.table(chunk_data)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_path, chunk_table.schema)
            parquet_writer.write_table(chunk_table)

        if parquet_writer:
            parquet_writer.close()

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    print(f"Total unique rows processed: {processed_rows}")

    return output_path

import psutil
def get_memory_usage():
    """Get current process memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def preprocess_occurrences_stream(config, occurrences_path: Path):
    start_time = time.time()
    
    # Memory tracking setup
    memory_log = []
    def log_memory(stage):
        current_memory = get_memory_usage()
        memory_log.append((stage, current_memory))
        print(f"{stage}: {current_memory:.2f} MB")

    log_memory("Start")

    assert occurrences_path is not None, "No occurrence path provided"
    if config['format'].lower() != "dwca":
        raise ValueError(f"Unknown format: {config['format'].lower()}")

    seen_identifiers = set()
    species_counts = defaultdict(int)
    max_img_per_species = config.get('max_img_spc', float('inf'))

    chunk_size = 100
    chunk_data = defaultdict(list)
    processed_rows = 0

    output_path = occurrences_path.with_suffix(".parquet")
    parquet_writer = None

    log_memory("Before processing")

    with DwCAReader(occurrences_path) as dwca:
        for row in dwca:
            extensions = row.extensions[:-1]
            
            for e in extensions:
                identifier = e.data['http://purl.org/dc/terms/identifier']
                
                if not identifier:
                    continue

                identifier_hash = mmh3.hash(identifier)
                if identifier_hash in seen_identifiers:
                    continue
                seen_identifiers.add(identifier_hash)

                metadata = {k.split('/')[-1]: v for k, v in row.data.items() 
                            if k.split('/')[-1] in KEYS_OCC + KEYS_GBIF}
                
                metadata.update({
                    k.split('/')[-1]: v for k, v in e.data.items() 
                    if k.split('/')[-1] in KEYS_MULT
                })

                if any(not metadata.get(key) for key in KEYS_GBIF):
                    continue

                taxon_key = metadata.get('taxonKey')
                species_counts[taxon_key] += 1
                
                if species_counts[taxon_key] > max_img_per_species:
                    continue

                # Accumulate data in chunk
                for k, v in metadata.items():
                    chunk_data[k].append(v)
                
                processed_rows += 1

                # Write chunk when full
                if processed_rows % chunk_size == 0:
                    chunk_table = pa.table(chunk_data)
                    
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(output_path, chunk_table.schema)
                    
                    parquet_writer.write_table(chunk_table)
                    chunk_data = defaultdict(list)
                    
                    log_memory(f"After processing {processed_rows} rows")

        # Write final chunk if exists
        if chunk_data:
            chunk_table = pa.table(chunk_data)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_path, chunk_table.schema)
            parquet_writer.write_table(chunk_table)

        if parquet_writer:
            parquet_writer.close()

    log_memory("End of processing")

    # Performance and memory summary
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    print(f"Total unique rows processed: {processed_rows}")
    
    # Optional: Print detailed memory log
    print("\nMemory Usage Log:")
    for stage, memory in memory_log:
        print(f"{stage}: {memory:.2f} MB")

    return output_path