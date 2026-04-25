# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File utilities for integration tests
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def get_test_files(
    data_dir: Path,
    count: int = 3,
    collection_type: str = "with_metadata",
    files_with_metadata: list[str] | None = None,
    files_without_metadata: list[str] | None = None,
) -> list[str]:
    """Get test files from the data directory or specified files"""
    # If specific files are provided for this collection type, use them
    if collection_type == "with_metadata" and files_with_metadata:
        files = []
        for file_path in files_with_metadata:
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                logger.warning(f"⚠️ Specified file not found: {file_path}")
        if files:
            logger.info(
                f"Using {len(files)} specified files for collection with metadata: {files}"
            )
            return files
    elif collection_type == "without_metadata" and files_without_metadata:
        files = []
        for file_path in files_without_metadata:
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                logger.warning(f"⚠️ Specified file not found: {file_path}")
        if files:
            logger.info(
                f"Using {len(files)} specified files for collection without metadata: {files}"
            )
            return files

    # Default file selection based on collection type
    if collection_type == "with_metadata":
        default_files = ["multimodal_test.pdf", "woods_frost.docx"]
    else:  # without_metadata
        default_files = ["table_test.pdf", "embedded_table.pdf"]

    # Try to find default files in data directory
    files = []
    for filename in default_files:
        file_path = data_dir / filename
        if file_path.exists():
            files.append(str(file_path))
        else:
            logger.warning(f"⚠️ Default file not found: {file_path}")

    if files:
        logger.info(
            f"Using {len(files)} default files for collection {collection_type}: {files}"
        )
        return files

    # Fallback to data directory if no specific files provided or files not found
    if not data_dir.exists():
        logger.error(f"❌ Data directory {data_dir} does not exist")
        return []

    # Get all PDF and DOCX files
    files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.docx"))
    if len(files) < count:
        logger.warning(f"⚠️ Only {len(files)} files found, using all available")
        count = len(files)

    logger.info(
        f"Using fallback files from data directory: {[str(f) for f in files[:count]]}"
    )
    return [str(f) for f in files[:count]]