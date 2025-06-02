# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module for loading media manifest files."""

import json
import logging
import re
from io import StringIO
from pathlib import Path

import pandas as pd

from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.util import load_files # Assuming this can be reused
from graphrag.logger.base import ProgressLogger
from graphrag.storage.pipeline_storage import PipelineStorage
from graphrag.config.enums import InputFileType

log = logging.getLogger(__name__)


async def _load_single_media_manifest_file(
    path: str,
    group: dict | None, # group is currently not used, but part of load_files callback signature
    config: InputConfig,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """Load and parse a single media manifest file (JSON or CSV)."""
    if not config.media_url_column:
        msg = "media_url_column must be specified in InputConfig when loading media manifests."
        log.error(msg)
        raise ValueError(msg)

    file_content = await storage.get(path, encoding=config.encoding)
    file_extension = Path(path).suffix.lower()

    rows = []
    if file_extension == ".json":
        data = json.loads(file_content)
        if isinstance(data, dict): # Single object
            rows.append(data)
        elif isinstance(data, list): # Array of objects
            rows.extend(data)
        else:
            log.warning(f"Unsupported JSON structure in manifest {path}. Expected object or array of objects.")
            return pd.DataFrame()
    elif file_extension == ".csv":
        csv_file = StringIO(file_content)
        df_csv = pd.read_csv(csv_file)
        rows = df_csv.to_dict(orient="records")
    else:
        log.warning(f"Unsupported manifest file type: {path}. Only .json and .csv are supported for media manifests.")
        return pd.DataFrame()

    if not rows:
        log.info(f"No data rows found in manifest file: {path}")
        return pd.DataFrame()

    processed_rows = []
    for row in rows:
        media_url = row.get(config.media_url_column)
        if not media_url:
            log.warning(f"Skipping row in {path} due to missing media_url_column '{config.media_url_column}'. Row: {row}")
            continue

        # Determine media_type from InputConfig.file_type
        media_type_str = None
        if config.file_type == InputFileType.image:
            media_type_str = "image"
        elif config.file_type == InputFileType.video:
            media_type_str = "video"
        else:
            # This case should ideally not be reached if load_media_manifest is called correctly
            log.warning(f"Cannot determine media type from config.file_type: {config.file_type} for manifest {path}")
            # Defaulting to 'unknown', or could skip
            media_type_str = "unknown"


        # Title: from configured title_column, else from media_url filename, else from manifest filename
        title = None
        if config.title_column and row.get(config.title_column):
            title = row.get(config.title_column)
        if not title:
            try:
                title = Path(str(media_url)).name
            except Exception: # Handle potential errors if media_url is not a valid path-like string
                pass
        if not title:
            title = Path(path).name


        entry = {
            "s3_url": media_url, # Assuming S3 for now as per issue, but could be any URL
            "title": title,
            "media_type": media_type_str,
            "manifest_path": path, # For traceability
        }

        # Add other metadata attributes
        if config.media_attributes_columns:
            for attr_col in config.media_attributes_columns:
                if attr_col in row:
                    entry[attr_col] = row[attr_col]
                else:
                    log.warning(f"media_attributes_column '{attr_col}' not found in row in {path}. Row: {row}")

        # Store all other columns from the manifest as attributes, prefixed to avoid collision
        for key, value in row.items():
            if key not in [config.media_url_column, config.title_column] and (not config.media_attributes_columns or key not in config.media_attributes_columns) :
                entry[f"manifest_attr_{key}"] = value

        processed_rows.append(entry)

    if not processed_rows:
        return pd.DataFrame()

    df = pd.DataFrame(processed_rows)
    # Add creation date of the manifest file itself
    creation_date = await storage.get_creation_date(path)
    df["manifest_creation_date"] = creation_date

    return df


async def load_media_manifest(
    config: InputConfig,
    progress: ProgressLogger | None,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """Load media manifest inputs from a directory.

    Manifest files themselves can be JSON or CSV.
    The config.file_type is expected to be InputFileType.image or InputFileType.video,
    which dictates the 'media_type' of the items within the manifest.
    The config.file_pattern should point to the manifest files themselves (e.g., '*.manifest.json').
    """
    log.info(f"Loading media manifest files from {config.base_dir} using pattern {config.file_pattern}")

    if config.file_type not in [InputFileType.image, InputFileType.video]:
        msg = f"load_media_manifest expects config.file_type to be 'image' or 'video', but got {config.file_type}"
        log.error(msg)
        raise ValueError(msg)

    # Curry the config and storage into the single file loader
    async def file_loader_fn(path: str, group: dict | None):
        return await _load_single_media_manifest_file(path, group, config, storage)

    # The load_files utility expects a loader function that takes (path, group)
    # We need to pass config and storage to _load_single_media_manifest_file
    # So, we create a wrapper or ensure load_files can pass through extra args if it supports it.
    # For simplicity here, we'll assume load_files passes only path and group.
    # A more robust way would be to use functools.partial or a small lambda if load_files is strict.
    # However, looking at load_files signature, it takes `loader: Any`.
    # The json.py and csv.py pass `load_file` which internally uses config and storage from its closure.
    # Let's replicate that pattern for consistency with existing input loaders.

    # Replicating pattern from json.py and csv.py for load_files:
    # We define the loader for a single file within this scope to capture config and storage.

    # This inner function will be passed to load_files
    async def _bound_load_single_manifest(path: str, group: dict | None) -> pd.DataFrame:
        return await _load_single_media_manifest_file(path, group, config, storage)

    all_manifest_data = await load_files( # type: ignore (complaining about _bound_load_single_manifest)
        _bound_load_single_manifest,
        config, # config here is used by load_files for file_pattern, file_filter etc.
        storage,
        progress,
    )

    if all_manifest_data is None or all_manifest_data.empty:
        log.warning(
            f"No media items loaded from manifests in {config.base_dir} with pattern {config.file_pattern}. "
            f"Ensure 'media_url_column' ('{config.media_url_column}') is correctly specified and present in your manifest files."
        )
        return pd.DataFrame()

    log.info(f"Successfully loaded {len(all_manifest_data)} media items from manifest files.")
    return all_manifest_data
