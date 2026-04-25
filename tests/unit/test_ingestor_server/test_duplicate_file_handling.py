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
Unit tests for duplicate file detection and handling in the ingestor server.
Tests cover the process_file_paths function and validation error propagation.
"""

import inspect
import os
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nvidia_rag.ingestor_server.main import NvidiaRAGIngestor
from nvidia_rag.ingestor_server.server import process_file_paths
from nvidia_rag.utils.vdb.vdb_base import VDBRag


class MockUploadFile:
    """Mock for FastAPI UploadFile"""

    def __init__(self, filename: str, content: bytes = b"test content"):
        self.filename = filename
        self.file = BytesIO(content)

    def __repr__(self):
        return f"MockUploadFile({self.filename})"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config(temp_dir):
    """Mock the CONFIG object with a temporary directory"""
    with patch("nvidia_rag.ingestor_server.server.CONFIG") as mock_cfg:
        mock_cfg.temp_dir = temp_dir
        yield mock_cfg


class TestProcessFilePathsNoDuplicates:
    """Test process_file_paths with unique files"""

    @pytest.mark.asyncio
    async def test_single_file_no_duplicates(self, mock_config):
        """Test processing a single file returns correct paths and no errors"""

        files = [MockUploadFile("test1.txt")]
        collection_name = "test_collection"

        file_paths, duplicate_errors = await process_file_paths(files, collection_name)

        assert len(file_paths) == 1
        assert len(duplicate_errors) == 0
        assert "test1.txt" in file_paths[0]
        assert os.path.exists(file_paths[0])

    @pytest.mark.asyncio
    async def test_multiple_unique_files_no_duplicates(self, mock_config):
        """Test processing multiple unique files"""

        files = [
            MockUploadFile("file1.txt"),
            MockUploadFile("file2.pdf"),
            MockUploadFile("file3.docx"),
        ]
        collection_name = "test_collection"

        file_paths, duplicate_errors = await process_file_paths(files, collection_name)

        assert len(file_paths) == 3
        assert len(duplicate_errors) == 0
        assert all(os.path.exists(path) for path in file_paths)
        assert "file1.txt" in file_paths[0]
        assert "file2.pdf" in file_paths[1]
        assert "file3.docx" in file_paths[2]

    @pytest.mark.asyncio
    async def test_empty_file_list(self, mock_config):
        """Test processing empty file list"""

        files = []
        collection_name = "test_collection"

        file_paths, duplicate_errors = await process_file_paths(files, collection_name)

        assert len(file_paths) == 0
        assert len(duplicate_errors) == 0


class TestProcessFilePathsWithDuplicates:
    """Test process_file_paths with duplicate files"""

    @pytest.mark.asyncio
    async def test_two_identical_files(self, mock_config):
        """Test that duplicate files are detected and only one is processed"""

        files = [
            MockUploadFile("duplicate.txt", b"content1"),
            MockUploadFile("duplicate.txt", b"content2"),
        ]
        collection_name = "test_collection"

        file_paths, duplicate_errors = await process_file_paths(files, collection_name)

        # Only one file should be processed
        assert len(file_paths) == 1
        assert "duplicate.txt" in file_paths[0]

        # Should have one validation error
        assert len(duplicate_errors) == 1
        error = duplicate_errors[0]
        assert "duplicate.txt" in error["error"]
        assert "1 duplicate(s) found" in error["error"]
        assert error["metadata"]["filename"] == "duplicate.txt"
        assert error["metadata"]["duplicate_count"] == 1
        assert error["metadata"]["total_occurrences"] == 2

    @pytest.mark.asyncio
    async def test_three_identical_files(self, mock_config):
        """Test that three identical files result in only one being processed"""

        files = [
            MockUploadFile("triple.txt", b"content1"),
            MockUploadFile("triple.txt", b"content2"),
            MockUploadFile("triple.txt", b"content3"),
        ]
        collection_name = "test_collection"

        file_paths, duplicate_errors = await process_file_paths(files, collection_name)

        assert len(file_paths) == 1
        assert len(duplicate_errors) == 1

        error = duplicate_errors[0]
        assert "triple.txt" in error["error"]
        assert "2 duplicate(s) found" in error["error"]
        assert error["metadata"]["duplicate_count"] == 2
        assert error["metadata"]["total_occurrences"] == 3

    @pytest.mark.asyncio
    async def test_multiple_sets_of_duplicates(self, mock_config):
        """Test multiple different files with duplicates"""

        files = [
            MockUploadFile("file1.txt"),
            MockUploadFile("file1.txt"),
            MockUploadFile("file2.pdf"),
            MockUploadFile("file2.pdf"),
            MockUploadFile("file2.pdf"),
            MockUploadFile("unique.docx"),
        ]
        collection_name = "test_collection"

        file_paths, duplicate_errors = await process_file_paths(files, collection_name)

        # Should process 3 unique files
        assert len(file_paths) == 3

        # Should have 2 validation errors (one for each duplicate set)
        assert len(duplicate_errors) == 2

        # Check file1.txt duplicate error
        file1_error = next(e for e in duplicate_errors if "file1.txt" in e["error"])
        assert file1_error["metadata"]["duplicate_count"] == 1
        assert file1_error["metadata"]["total_occurrences"] == 2

        # Check file2.pdf duplicate error
        file2_error = next(e for e in duplicate_errors if "file2.pdf" in e["error"])
        assert file2_error["metadata"]["duplicate_count"] == 2
        assert file2_error["metadata"]["total_occurrences"] == 3

    @pytest.mark.asyncio
    async def test_mixed_unique_and_duplicate_files(self, mock_config):
        """Test mixture of unique and duplicate files"""

        files = [
            MockUploadFile("unique1.txt"),
            MockUploadFile("duplicate.pdf"),
            MockUploadFile("unique2.docx"),
            MockUploadFile("duplicate.pdf"),
            MockUploadFile("unique3.xlsx"),
        ]
        collection_name = "test_collection"

        file_paths, duplicate_errors = await process_file_paths(files, collection_name)

        assert len(file_paths) == 4  # 3 unique + 1 duplicate set
        assert len(duplicate_errors) == 1

        error = duplicate_errors[0]
        assert "duplicate.pdf" in error["error"]
        assert error["metadata"]["duplicate_count"] == 1


class TestProcessFilePathsErrorHandling:
    """Test error handling in process_file_paths"""

    @pytest.mark.asyncio
    async def test_empty_filename_raises_error(self, mock_config):
        """Test that empty filename raises RuntimeError"""

        files = [MockUploadFile("")]
        collection_name = "test_collection"

        with pytest.raises(RuntimeError, match="Error parsing uploaded filename"):
            await process_file_paths(files, collection_name)

    @pytest.mark.asyncio
    async def test_file_with_special_characters(self, mock_config):
        """Test handling files with special characters in name"""

        files = [
            MockUploadFile("file-with-dashes.txt"),
            MockUploadFile("file_with_underscores.txt"),
            MockUploadFile("file with spaces.txt"),
        ]
        collection_name = "test_collection"

        file_paths, duplicate_errors = await process_file_paths(files, collection_name)

        assert len(file_paths) == 3
        assert len(duplicate_errors) == 0


class TestProcessFilePathsDirectoryCreation:
    """Test that directories are created correctly"""

    @pytest.mark.asyncio
    async def test_collection_directory_created(self, mock_config):
        """Test that collection-specific directory is created"""

        files = [MockUploadFile("test.txt")]
        collection_name = "new_collection"

        _file_paths, _ = await process_file_paths(files, collection_name)

        expected_dir = Path(mock_config.temp_dir) / "uploaded_files" / collection_name
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    @pytest.mark.asyncio
    async def test_nested_collection_name(self, mock_config):
        """Test handling collection names with special characters"""

        files = [MockUploadFile("test.txt")]
        collection_name = "my_collection_v1"

        file_paths, _ = await process_file_paths(files, collection_name)

        assert len(file_paths) == 1
        assert "my_collection_v1" in file_paths[0]


class TestValidationErrorFormat:
    """Test the format of validation errors"""

    @pytest.mark.asyncio
    async def test_validation_error_structure(self, mock_config):
        """Test that validation errors have correct structure"""

        files = [
            MockUploadFile("dup.txt"),
            MockUploadFile("dup.txt"),
        ]
        collection_name = "test_collection"

        _, duplicate_errors = await process_file_paths(files, collection_name)

        assert len(duplicate_errors) == 1
        error = duplicate_errors[0]

        # Check top-level structure
        assert "error" in error
        assert "metadata" in error
        assert isinstance(error["error"], str)
        assert isinstance(error["metadata"], dict)

        # Check metadata structure
        metadata = error["metadata"]
        assert "filename" in metadata
        assert "duplicate_count" in metadata
        assert "total_occurrences" in metadata
        assert isinstance(metadata["filename"], str)
        assert isinstance(metadata["duplicate_count"], int)
        assert isinstance(metadata["total_occurrences"], int)

    @pytest.mark.asyncio
    async def test_validation_error_message_format(self, mock_config):
        """Test that error message is properly formatted"""

        files = [
            MockUploadFile("test.pdf"),
            MockUploadFile("test.pdf"),
            MockUploadFile("test.pdf"),
        ]
        collection_name = "test_collection"

        _, duplicate_errors = await process_file_paths(files, collection_name)

        error = duplicate_errors[0]
        error_msg = error["error"]

        assert "test.pdf" in error_msg
        assert "2 duplicate(s) found" in error_msg
        assert "Duplicates were discarded" in error_msg
        assert "1 file is being processed" in error_msg


class TestAdditionalValidationErrorsPropagation:
    """Test that additional_validation_errors parameter works correctly"""

    @pytest.mark.asyncio
    async def test_upload_documents_signature_includes_additional_errors(self):
        """Test that upload_documents method accepts additional_validation_errors parameter"""

        # Check that the parameter exists in the signature
        sig = inspect.signature(NvidiaRAGIngestor.upload_documents)
        assert "additional_validation_errors" in sig.parameters

        # Check default value
        param = sig.parameters["additional_validation_errors"]
        assert param.default is None

    @pytest.mark.asyncio
    async def test_update_documents_signature_includes_additional_errors(self):
        """Test that update_documents method accepts additional_validation_errors parameter"""

        # Check that the parameter exists in the signature
        sig = inspect.signature(NvidiaRAGIngestor.update_documents)
        assert "additional_validation_errors" in sig.parameters

        # Check default value
        param = sig.parameters["additional_validation_errors"]
        assert param.default is None

    @pytest.mark.asyncio
    async def test_additional_validation_errors_parameter_documented(self):
        """Test that additional_validation_errors is documented in docstrings"""

        # Check upload_documents docstring
        upload_doc = NvidiaRAGIngestor.upload_documents.__doc__
        assert upload_doc is not None
        assert "additional_validation_errors" in upload_doc

        # Check update_documents docstring
        update_doc = NvidiaRAGIngestor.update_documents.__doc__
        assert update_doc is not None


class TestLoggingBehavior:
    """Test that duplicate detection logs appropriate warnings"""

    @pytest.mark.asyncio
    async def test_duplicate_logging(self, mock_config, caplog):
        """Test that duplicates are logged with warnings"""

        files = [
            MockUploadFile("test.txt"),
            MockUploadFile("test.txt"),
        ]
        collection_name = "test_collection"

        with caplog.at_level("WARNING"):
            await process_file_paths(files, collection_name)

        # Check that warning was logged
        assert any(
            "Duplicate files detected" in record.message for record in caplog.records
        )
        assert any("test.txt" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_no_logging_for_unique_files(self, mock_config, caplog):
        """Test that no warnings are logged for unique files"""
        files = [
            MockUploadFile("file1.txt"),
            MockUploadFile("file2.txt"),
        ]
        collection_name = "test_collection"

        with caplog.at_level("WARNING"):
            await process_file_paths(files, collection_name)

        # Check that no duplicate warnings were logged
        assert not any(
            "Duplicate files detected" in record.message for record in caplog.records
        )


class TestUpdateDocumentsWithDuplicates:
    """Test that update_documents also handles duplicates correctly"""

    @pytest.mark.asyncio
    async def test_update_documents_with_additional_validation_errors(self):
        """Test that update_documents passes additional_validation_errors"""

        mock_vdb_op = Mock(spec=VDBRag)
        mock_vdb_op.check_collection_exists.return_value = True
        mock_vdb_op.get_metadata_schema.return_value = None

        ingestor = NvidiaRAGIngestor(vdb_op=mock_vdb_op)

        # Mock upload_documents to capture parameters
        captured_kwargs = {}

        async def mock_upload(**kwargs):
            captured_kwargs.update(kwargs)
            return {"message": "Success", "validation_errors": []}

        ingestor.upload_documents = mock_upload

        # Call update_documents with additional_validation_errors
        test_errors = [{"error": "Test duplicate error"}]
        await ingestor.update_documents(
            filepaths=[],
            collection_name="test",
            blocking=True,
            additional_validation_errors=test_errors,
        )

        # Verify the parameter was passed through
        assert "additional_validation_errors" in captured_kwargs
        assert captured_kwargs["additional_validation_errors"] == test_errors
