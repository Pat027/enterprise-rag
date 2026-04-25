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
Document upload test module
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.file_utils import get_test_files

logger = logging.getLogger(__name__)


class DocumentUploadModule(BaseTestModule):
    """Document upload test module"""

    def __init__(self, test_runner):
        super().__init__(test_runner)
        self._task_id = None
        self._test_files_without_metadata = []
        self._metadata_update_result = None

    @test_case(4, "Upload Documents with Metadata")
    async def _test_upload_with_metadata(self) -> bool:
        """Test uploading documents with metadata"""
        logger.info("\n=== Test 4: Upload Documents with Metadata and Summary ===")
        upload_start = time.time()
        test_files = self._get_test_files(3, "with_metadata")
        if not test_files:
            self.add_test_result(
                self._test_upload_with_metadata.test_number,
                self._test_upload_with_metadata.test_name,
                f"Upload test documents with custom metadata and generate summaries (non-blocking mode). Collection: {self.collections['with_metadata']}. Supports default file selection (multimodal_test.pdf, woods_frost.docx) or custom files via --files-with-metadata argument.",
                ["POST /v1/documents"],
                [
                    "collection_name",
                    "blocking",
                    "custom_metadata",
                    "generate_summary",
                    "split_options",
                ],
                time.time() - upload_start,
                TestStatus.FAILURE,
                "No test files found",
            )
            return False

        # Store test files in instance variable for use in other methods
        self.test_runner.test_files = test_files

        custom_metadata = [
            {
                "filename": os.path.basename(test_files[0]),
                "metadata": self.test_runner.original_metadata,
            }
        ]

        task_id = await self._upload_documents(
            self.collections["with_metadata"],
            test_files,
            blocking=False,
            custom_metadata=custom_metadata,
            generate_summary=True,
        )
        upload_time = time.time() - upload_start

        # Get file names for description
        file_names = [os.path.basename(f) for f in test_files]

        if task_id:
            self.add_test_result(
                self._test_upload_with_metadata.test_number,
                self._test_upload_with_metadata.test_name,
                f"Upload test documents with custom metadata and generate summaries (non-blocking mode). Collection: {self.collections['with_metadata']}. Files: {', '.join(file_names)}. Metadata: {self.test_runner.original_metadata}. Supports default file selection (multimodal_test.pdf, woods_frost.docx) or custom files via --files-with-metadata argument.",
                ["POST /v1/documents"],
                [
                    "collection_name",
                    "blocking",
                    "custom_metadata",
                    "generate_summary",
                    "split_options",
                ],
                upload_time,
                TestStatus.SUCCESS,
            )
            # Store task_id for next test
            self._task_id = task_id
            return True
        else:
            self.add_test_result(
                self._test_upload_with_metadata.test_number,
                self._test_upload_with_metadata.test_name,
                f"Upload test documents with custom metadata and generate summaries (non-blocking mode). Collection: {self.collections['with_metadata']}. Files: {', '.join(file_names)}. Metadata: {self.test_runner.original_metadata}. Supports default file selection (multimodal_test.pdf, woods_frost.docx) or custom files via --files-with-metadata argument.",
                ["POST /v1/documents"],
                [
                    "collection_name",
                    "blocking",
                    "custom_metadata",
                    "generate_summary",
                    "split_options",
                ],
                upload_time,
                TestStatus.FAILURE,
                "Failed to upload documents",
            )
            return False

    @test_case(5, "Wait for Ingestion Completion")
    async def _test_wait_for_ingestion(self) -> bool:
        """Test waiting for ingestion completion"""
        logger.info("\n=== Test 5: Wait for Ingestion Completion ===")
        wait_start = time.time()
        wait_success = await self._wait_for_task_completion(self._task_id)
        wait_time = time.time() - wait_start

        if wait_success:
            self.add_test_result(
                self._test_wait_for_ingestion.test_number,
                self._test_wait_for_ingestion.test_name,
                "Poll task status until ingestion is completed successfully",
                ["GET /v1/status"],
                ["task_id", "state"],
                wait_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_wait_for_ingestion.test_number,
                self._test_wait_for_ingestion.test_name,
                "Poll task status until ingestion is completed successfully",
                ["GET /v1/status"],
                ["task_id", "state"],
                wait_time,
                TestStatus.FAILURE,
                "Task failed or timed out",
            )
            return False

    @test_case(6, "Upload Documents without Metadata")
    async def _test_upload_without_metadata(self) -> bool:
        """Test uploading documents without metadata"""
        logger.info("\n=== Test 6: Upload Documents without Metadata (Blocking) ===")
        blocking_start = time.time()
        test_files_without_metadata = self._get_test_files(3, "without_metadata")
        if not test_files_without_metadata:
            self.add_test_result(
                self._test_upload_without_metadata.test_number,
                self._test_upload_without_metadata.test_name,
                f"Upload documents without metadata using blocking mode for immediate completion. Collection: {self.collections['without_metadata']}. Supports default file selection (table_test.pdf, embedded_table.pdf) or custom files via --files-without-metadata argument.",
                ["POST /v1/documents"],
                ["collection_name", "blocking", "split_options"],
                time.time() - blocking_start,
                TestStatus.FAILURE,
                "No test files found for collection without metadata",
            )
            return False

        result = await self._upload_documents(
            self.collections["without_metadata"],
            test_files_without_metadata,
            blocking=True,
        )
        blocking_time = time.time() - blocking_start

        # Get file names for description
        file_names_without_metadata = [
            os.path.basename(f) for f in test_files_without_metadata
        ]

        if result == "BLOCKING_COMPLETED":
            self.add_test_result(
                self._test_upload_without_metadata.test_number,
                self._test_upload_without_metadata.test_name,
                f"Upload documents without metadata using blocking mode for immediate completion. Collection: {self.collections['without_metadata']}. Files: {', '.join(file_names_without_metadata)}. Supports default file selection (table_test.pdf, embedded_table.pdf) or custom files via --files-without-metadata argument.",
                ["POST /v1/documents"],
                ["collection_name", "blocking", "split_options"],
                blocking_time,
                TestStatus.SUCCESS,
            )
            # Store files for later tests
            self._test_files_without_metadata = test_files_without_metadata
            return True
        else:
            self.add_test_result(
                self._test_upload_without_metadata.test_number,
                self._test_upload_without_metadata.test_name,
                f"Upload documents without metadata using blocking mode for immediate completion. Collection: {self.collections['without_metadata']}. Files: {', '.join(file_names_without_metadata)}. Supports default file selection (table_test.pdf, embedded_table.pdf) or custom files via --files-without-metadata argument.",
                ["POST /v1/documents"],
                ["collection_name", "blocking", "split_options"],
                blocking_time,
                TestStatus.FAILURE,
                "Blocking upload failed",
            )
            return False

    @test_case(7, "Update Document Metadata")
    async def _test_update_metadata(self) -> bool:
        """Test updating document metadata"""
        logger.info("\n=== Test 7: Update Document Metadata ===")
        update_start = time.time()
        filename = os.path.basename(self.test_runner.test_files[0])

        result = await self._update_document_metadata(
            self.collections["with_metadata"],
            filename,
            self.test_runner.updated_metadata,
            blocking=False,
        )
        update_time = time.time() - update_start

        if result:
            self.add_test_result(
                self._test_update_metadata.test_number,
                self._test_update_metadata.test_name,
                f"Update metadata for existing documents in the collection using PATCH endpoint. Collection: {self.collections['with_metadata']}. File: {filename}. Updated metadata: {self.test_runner.updated_metadata}. Supports both blocking and non-blocking modes with automatic summary regeneration.",
                ["PATCH /v1/documents"],
                ["collection_name", "custom_metadata", "generate_summary"],
                update_time,
                TestStatus.SUCCESS,
            )
            # Store result for next test
            self._metadata_update_result = result
            return True
        else:
            self.add_test_result(
                self._test_update_metadata.test_number,
                self._test_update_metadata.test_name,
                f"Update metadata for existing documents in the collection using PATCH endpoint. Collection: {self.collections['with_metadata']}. File: {filename}. Updated metadata: {self.test_runner.updated_metadata}. Supports both blocking and non-blocking modes with automatic summary regeneration.",
                ["PATCH /v1/documents"],
                ["collection_name", "custom_metadata", "generate_summary"],
                update_time,
                TestStatus.FAILURE,
                "Failed to update document metadata",
            )
            return False

    @test_case(8, "Wait for Metadata Update")
    async def _test_wait_for_metadata_update(self) -> bool:
        """Test waiting for metadata update completion"""
        logger.info("\n=== Test 8: Wait for Metadata Update Completion ===")
        if self._metadata_update_result == "BLOCKING_COMPLETED":
            self.add_test_result(
                self._test_wait_for_metadata_update.test_number,
                self._test_wait_for_metadata_update.test_name,
                "Metadata update completed in blocking mode",
                ["PATCH /v1/documents (blocking)"],
                ["blocking"],
                0.0,
                TestStatus.SUCCESS,
            )
            return True
        else:
            # For non-blocking updates, wait for task completion
            wait_update_start = time.time()
            wait_update_success = await self._wait_for_task_completion(
                self._metadata_update_result
            )
            wait_update_time = time.time() - wait_update_start

            if wait_update_success:
                self.add_test_result(
                    self._test_wait_for_metadata_update.test_number,
                    self._test_wait_for_metadata_update.test_name,
                    "Poll task status until metadata update is completed",
                    ["GET /v1/status"],
                    ["task_id", "state"],
                    wait_update_time,
                    TestStatus.SUCCESS,
                )
                return True
            else:
                self.add_test_result(
                    self._test_wait_for_metadata_update.test_number,
                    self._test_wait_for_metadata_update.test_name,
                    "Poll task status until metadata update is completed",
                    ["GET /v1/status"],
                    ["task_id", "state"],
                    wait_update_time,
                    TestStatus.FAILURE,
                    "Metadata update task failed or timed out",
                )
                return False

    @test_case(66, "Upload Duplicate Files")
    async def _test_upload_duplicate_files(self) -> bool:
        """Test uploading duplicate files to verify deduplication"""
        logger.info("\n=== Test 66: Upload Duplicate Files ===")
        duplicate_start = time.time()

        # Get a single test file and upload it twice
        test_files = self._get_test_files(1, "without_metadata")
        if not test_files:
            self.add_test_result(
                self._test_upload_duplicate_files.test_number,
                self._test_upload_duplicate_files.test_name,
                f"Upload same file multiple times to verify duplicate detection and handling. Collection: {self.collections['without_metadata']}",
                ["POST /v1/documents"],
                ["validation_errors"],
                time.time() - duplicate_start,
                TestStatus.FAILURE,
                "No test files found",
            )
            return False

        # Upload the same file 3 times (duplicates)
        duplicate_files = [test_files[0], test_files[0], test_files[0]]

        result = await self._upload_documents_with_duplicates(
            self.collections["without_metadata"],
            duplicate_files,
            blocking=True,
        )
        duplicate_time = time.time() - duplicate_start

        file_name = os.path.basename(test_files[0])

        if result and result.get("validation_errors"):
            # Check if we got the expected duplicate warning
            validation_errors = result.get("validation_errors", [])
            has_duplicate_error = any(
                file_name in error.get("error", "")
                and "duplicate" in error.get("error", "").lower()
                for error in validation_errors
            )

            if has_duplicate_error:
                total_docs = result.get("total_documents", 0)
                logger.info(
                    f"✅ Duplicate detection working: {len(validation_errors)} validation error(s), {total_docs} document(s) processed"
                )
                self.add_test_result(
                    self._test_upload_duplicate_files.test_number,
                    self._test_upload_duplicate_files.test_name,
                    f"Upload same file multiple times to verify duplicate detection and handling. Collection: {self.collections['without_metadata']}. File: {file_name} (uploaded 3 times, expected to process only 1). Validates that validation_errors array contains duplicate warnings with proper metadata.",
                    ["POST /v1/documents"],
                    [
                        "validation_errors",
                        "validation_errors[].metadata.duplicate_count",
                    ],
                    duplicate_time,
                    TestStatus.SUCCESS,
                )
                return True
            else:
                logger.error("❌ No duplicate error found in validation_errors")
                self.add_test_result(
                    self._test_upload_duplicate_files.test_number,
                    self._test_upload_duplicate_files.test_name,
                    f"Upload same file multiple times to verify duplicate detection and handling. Collection: {self.collections['without_metadata']}. File: {file_name} (uploaded 3 times, expected to process only 1). Validates that validation_errors array contains duplicate warnings with proper metadata.",
                    ["POST /v1/documents"],
                    [
                        "validation_errors",
                        "validation_errors[].metadata.duplicate_count",
                    ],
                    duplicate_time,
                    TestStatus.FAILURE,
                    "Expected duplicate validation error not found",
                )
                return False
        elif result and not result.get("validation_errors"):
            logger.error("❌ No validation errors returned for duplicate files")
            self.add_test_result(
                self._test_upload_duplicate_files.test_number,
                self._test_upload_duplicate_files.test_name,
                f"Upload same file multiple times to verify duplicate detection and handling. Collection: {self.collections['without_metadata']}. File: {file_name} (uploaded 3 times, expected to process only 1). Validates that validation_errors array contains duplicate warnings with proper metadata.",
                ["POST /v1/documents"],
                ["validation_errors", "validation_errors[].metadata.duplicate_count"],
                duplicate_time,
                TestStatus.FAILURE,
                "No validation_errors in response",
            )
            return False
        else:
            logger.error("❌ Failed to upload duplicate files")
            self.add_test_result(
                self._test_upload_duplicate_files.test_number,
                self._test_upload_duplicate_files.test_name,
                f"Upload same file multiple times to verify duplicate detection and handling. Collection: {self.collections['without_metadata']}. File: {file_name} (uploaded 3 times, expected to process only 1). Validates that validation_errors array contains duplicate warnings with proper metadata.",
                ["POST /v1/documents"],
                ["validation_errors", "validation_errors[].metadata.duplicate_count"],
                duplicate_time,
                TestStatus.FAILURE,
                "Upload failed",
            )
            return False

    @test_case(9, "Verify Documents in Collections")
    async def _test_verify_documents(self) -> bool:
        """Test verifying documents in collections"""
        logger.info("\n=== Test 9: Verify Documents in Collections ===")
        verify_docs_start = time.time()

        # Get expected document names from test files for each collection
        expected_document_names_with_metadata = [
            os.path.basename(f) for f in self.test_runner.test_files
        ]
        expected_document_names_without_metadata = [
            os.path.basename(f) for f in self._test_files_without_metadata
        ]

        # Verify documents with metadata and check that the metadata was updated correctly
        filename = os.path.basename(self.test_runner.test_files[0])
        expected_metadata = {
            filename: self.test_runner.updated_metadata  # This is the updated metadata from Test 7
        }
        verify_with_metadata = await self._verify_documents(
            self.collections["with_metadata"],
            expected_metadata,
            expected_document_names_with_metadata,
        )
        verify_without_metadata = await self._verify_documents(
            self.collections["without_metadata"],
            None,
            expected_document_names_without_metadata,
        )
        verify_docs_time = time.time() - verify_docs_start

        if verify_with_metadata and verify_without_metadata:
            self.add_test_result(
                self._test_verify_documents.test_number,
                self._test_verify_documents.test_name,
                f"Verify documents are properly ingested and metadata is correctly stored. Collections: {self.collections['with_metadata']} (files: {', '.join(expected_document_names_with_metadata)}), {self.collections['without_metadata']} (files: {', '.join(expected_document_names_without_metadata)}). Validates document list, metadata schema compliance, updated metadata values from previous operations, and filename metadata presence even without custom metadata.",
                ["GET /v1/documents"],
                [
                    "collection_name",
                    "documents[].document_name",
                    "documents[].metadata",
                ],
                verify_docs_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_verify_documents.test_number,
                self._test_verify_documents.test_name,
                f"Verify documents are properly ingested and metadata is correctly stored. Collections: {self.collections['with_metadata']} (files: {', '.join(expected_document_names_with_metadata)}), {self.collections['without_metadata']} (files: {', '.join(expected_document_names_without_metadata)}). Validates document list, metadata schema compliance, updated metadata values from previous operations, and filename metadata presence even without custom metadata.",
                ["GET /v1/documents"],
                [
                    "collection_name",
                    "documents[].document_name",
                    "documents[].metadata",
                ],
                verify_docs_time,
                TestStatus.FAILURE,
                "Document verification failed for one or both collections",
            )
            return False

    def _get_test_files(
        self, count: int = 3, collection_type: str = "with_metadata"
    ) -> list[str]:
        """Get test files from the data directory or specified files"""
        return get_test_files(
            self.test_runner.data_dir,
            count,
            collection_type,
            self.test_runner.files_with_metadata,
            self.test_runner.files_without_metadata,
        )

    async def _upload_documents_with_duplicates(
        self,
        collection_name: str,
        files: list[str],
        blocking: bool = False,
        custom_metadata: list[dict[str, Any]] | None = None,
        generate_summary: bool = False,
    ) -> dict[str, Any] | None:
        """Upload documents to a collection and return full response (for duplicate testing)"""
        data = {
            "collection_name": collection_name,
            "blocking": blocking,
            "split_options": {"chunk_size": 512, "chunk_overlap": 150},
            "custom_metadata": custom_metadata or [],
            "generate_summary": generate_summary,
        }

        form_data = aiohttp.FormData()
        for file_path in files:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    file_content = f.read()
                form_data.add_field(
                    "documents",
                    file_content,
                    filename=os.path.basename(file_path),
                    content_type="application/octet-stream",
                )

        form_data.add_field("data", json.dumps(data), content_type="application/json")

        async with aiohttp.ClientSession() as session:
            try:
                logger.info(
                    f"📤 Uploading {len(files)} documents (including duplicates) to collection '{collection_name}'"
                )
                logger.info(f"📁 Files: {[os.path.basename(f) for f in files]}")

                async with session.post(
                    f"{self.ingestor_server_url}/v1/documents", data=form_data
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Upload request successful. Response:\n{json.dumps(result, indent=2)}"
                        )
                        return result
                    else:
                        logger.error(
                            f"❌ Failed to upload documents. Status: {response.status}"
                        )
                        logger.error(f"❌ Response:\n{json.dumps(result, indent=2)}")
                        return None
            except Exception:
                logger.exception("❌ Error uploading documents")
                return None

    async def _upload_documents(
        self,
        collection_name: str,
        files: list[str],
        blocking: bool = False,
        custom_metadata: list[dict[str, Any]] = None,
        generate_summary: bool = False,
    ) -> str | None:
        """Upload documents to a collection"""
        data = {
            "collection_name": collection_name,
            "blocking": blocking,
            "split_options": {"chunk_size": 512, "chunk_overlap": 150},
            "custom_metadata": custom_metadata or [],
            "generate_summary": generate_summary,
        }

        form_data = aiohttp.FormData()
        for file_path in files:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    file_content = f.read()
                form_data.add_field(
                    "documents",
                    file_content,
                    filename=os.path.basename(file_path),
                    content_type="application/octet-stream",
                )

        form_data.add_field("data", json.dumps(data), content_type="application/json")

        async with aiohttp.ClientSession() as session:
            try:
                logger.info(
                    f"📤 Uploading {len(files)} documents to collection '{collection_name}'"
                )
                logger.info(f"📁 Files: {[os.path.basename(f) for f in files]}")
                logger.info(f"📋 Upload data: {json.dumps(data, indent=2)}")

                async with session.post(
                    f"{self.ingestor_server_url}/v1/documents", data=form_data
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Upload request successful. Response:\n{json.dumps(result, indent=2)}"
                        )
                        if blocking:
                            # For blocking uploads, the API returns completion result directly
                            total_documents = result.get("total_documents", 0)
                            failed_documents = result.get("failed_documents", [])
                            logger.info(
                                f"✅ Documents uploaded successfully (blocking). Total: {total_documents}, Failed: {len(failed_documents)}"
                            )
                            if failed_documents:
                                logger.warning(
                                    f"⚠️ Failed documents: {failed_documents}"
                                )
                            # Return a special value to indicate blocking completion
                            return "BLOCKING_COMPLETED"
                        else:
                            # For non-blocking uploads, return the task_id
                            task_id = result.get("task_id")
                            logger.info(
                                f"✅ Documents uploaded successfully. Task ID: {task_id}"
                            )
                            return task_id
                    else:
                        logger.error(
                            f"❌ Failed to upload documents. Status: {response.status}"
                        )
                        logger.error(f"❌ Response:\n{json.dumps(result, indent=2)}")
                        return None
            except Exception as e:
                logger.error(f"❌ Error uploading documents: {e}")
                return None

    async def _wait_for_task_completion(self, task_id: str) -> bool:
        """Wait for task completion by polling status endpoint"""
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            async with aiohttp.ClientSession() as session:
                try:
                    params = {"task_id": task_id}
                    async with session.get(
                        f"{self.ingestor_server_url}/v1/status", params=params
                    ) as response:
                        result = await response.json()
                        if response.status == 200:
                            state = result.get("state")
                            if state == "FINISHED":
                                logger.info(f"✅ Task {task_id} completed successfully")
                                logger.info(
                                    f"Task result:\n{json.dumps(result, indent=2)}"
                                )
                                return True
                            elif state == "FAILED":
                                logger.error(f"❌ Task {task_id} failed")
                                logger.error(
                                    f"Task failure details:\n{json.dumps(result, indent=2)}"
                                )
                                return False
                            else:
                                logger.info(f"⏳ Task {task_id} state: {state}")
                                # Log additional task details for debugging
                                if "progress" in result:
                                    logger.info(
                                        f"   Progress: {result.get('progress')}"
                                    )
                                if "message" in result:
                                    logger.info(f"   Message: {result.get('message')}")

                except Exception as e:
                    logger.error(f"❌ Error checking task status: {e}")

            await asyncio.sleep(self.poll_interval)

        logger.error(f"❌ Task {task_id} timed out after {self.timeout} seconds")
        return False

    async def _update_document_metadata(
        self,
        collection_name: str,
        filename: str,
        new_metadata: dict[str, Any],
        blocking: bool = False,
        generate_summary: bool = True,
    ) -> str | None:
        """Update document metadata using PATCH /documents"""
        # Find the original file path from test_files
        original_file_path = None
        for test_file in self.test_runner.test_files:
            if os.path.basename(test_file) == filename:
                original_file_path = test_file
                break

        if not original_file_path or not os.path.exists(original_file_path):
            logger.error(f"❌ Original file {filename} not found for metadata update")
            return None

        data = {
            "collection_name": collection_name,
            "blocking": blocking,
            "custom_metadata": [{"filename": filename, "metadata": new_metadata}],
            "generate_summary": generate_summary,
        }

        form_data = aiohttp.FormData()
        # Add the original file to satisfy the API schema requirement
        with open(original_file_path, "rb") as f:
            file_content = f.read()
        form_data.add_field(
            "documents",
            file_content,
            filename=filename,
            content_type="application/octet-stream",
        )
        form_data.add_field("data", json.dumps(data), content_type="application/json")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.patch(
                    f"{self.ingestor_server_url}/v1/documents", data=form_data
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        if blocking:
                            # For blocking updates, the API returns completion result directly
                            logger.info(
                                "✅ Document metadata updated successfully (blocking)"
                            )
                            return "BLOCKING_COMPLETED"
                        else:
                            # For non-blocking updates, return the task_id
                            task_id = result.get("task_id")
                            logger.info(
                                f"✅ Document metadata updated successfully. Task ID: {task_id}"
                            )
                            return task_id
                    else:
                        logger.error("❌ Failed to update document metadata")
                        return None
            except Exception as e:
                logger.error(f"❌ Error updating document metadata: {e}")
                return None

    async def _verify_documents(
        self,
        collection_name: str,
        expected_metadata: dict[str, dict[str, Any]] | None = None,
        expected_documents: list[str] | None = None,
    ) -> bool:
        """Verify documents are ingested in the collection and optionally verify metadata and document list"""
        async with aiohttp.ClientSession() as session:
            try:
                params = {"collection_name": collection_name}
                async with session.get(
                    f"{self.ingestor_server_url}/v1/documents", params=params
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        documents = result.get("documents", [])
                        logger.info(
                            f"Found {len(documents)} documents in collection '{collection_name}'"
                        )

                        # Extract actual document names from the collection
                        actual_document_names = [
                            doc.get("document_name") for doc in documents
                        ]
                        logger.info(
                            f"Actual documents in collection: {actual_document_names}"
                        )

                        # Verify that only expected documents are present (if expected_documents provided)
                        if expected_documents is not None:
                            expected_doc_set = set(expected_documents)
                            actual_doc_set = set(actual_document_names)

                            # Check for unexpected documents
                            unexpected_documents = actual_doc_set - expected_doc_set
                            if unexpected_documents:
                                logger.error(
                                    f"❌ Unexpected documents found in collection '{collection_name}': {unexpected_documents}"
                                )
                                logger.error(
                                    f"Expected documents: {expected_documents}"
                                )
                                logger.error(
                                    f"Actual documents: {actual_document_names}"
                                )
                                return False

                            # Check for missing documents
                            missing_documents = expected_doc_set - actual_doc_set
                            if missing_documents:
                                logger.error(
                                    f"❌ Missing documents in collection '{collection_name}': {missing_documents}"
                                )
                                logger.error(
                                    f"Expected documents: {expected_documents}"
                                )
                                logger.error(
                                    f"Actual documents: {actual_document_names}"
                                )
                                return False

                            logger.info(
                                f"✅ Document list verification passed for collection '{collection_name}'"
                            )

                        # Check if metadata schema is available for collection with metadata
                        if collection_name == self.collections["with_metadata"]:
                            for doc in documents:
                                filename = doc.get("document_name")
                                if "metadata" in doc:
                                    logger.info(
                                        f"✅ Document '{filename}' has metadata"
                                    )

                                    # Verify expected metadata if provided
                                    if (
                                        expected_metadata
                                        and filename in expected_metadata
                                    ):
                                        expected = expected_metadata[filename]
                                        actual = doc.get("metadata", {})

                                        # Check if all expected metadata fields match
                                        metadata_matches = True
                                        for key, expected_value in expected.items():
                                            if key not in actual:
                                                logger.error(
                                                    f"❌ Document '{filename}' missing expected metadata field '{key}'"
                                                )
                                                metadata_matches = False
                                            elif actual[key] != expected_value:
                                                logger.error(
                                                    f"❌ Document '{filename}' metadata field '{key}' mismatch. Expected: {expected_value}, Got: {actual[key]}"
                                                )
                                                metadata_matches = False

                                        if metadata_matches:
                                            logger.info(
                                                f"✅ Document '{filename}' metadata verified successfully"
                                            )
                                        else:
                                            logger.error(
                                                f"❌ Document '{filename}' metadata verification failed"
                                            )
                                            return False
                                else:
                                    logger.warning(
                                        f"⚠️ Document '{filename}' missing metadata"
                                    )

                        # Check filename metadata for collection without custom metadata
                        elif collection_name == self.collections["without_metadata"]:
                            for doc in documents:
                                filename = doc.get("document_name")
                                metadata = doc.get("metadata", {})

                                # Verify that filename is present and not None in metadata even without custom_metadata
                                if (
                                    "filename" not in metadata
                                    or metadata["filename"] is None
                                ):
                                    logger.error(
                                        f"❌ Document '{filename}' missing or null filename in metadata"
                                    )
                                    return False
                                else:
                                    logger.info(
                                        f"✅ Document '{filename}' has filename metadata: {metadata['filename']}"
                                    )

                        # For deletion verification, we want to return True when documents are empty
                        # For normal verification, we want to return True when documents exist
                        if (
                            expected_documents is not None
                            and len(expected_documents) == 0
                        ):
                            # This is a deletion verification - return True if no documents found
                            return len(documents) == 0
                        else:
                            # This is a normal verification - return True if documents exist
                            return len(documents) > 0
                    else:
                        logger.error("❌ Failed to verify documents")
                        return False
            except Exception as e:
                logger.error(f"❌ Error verifying documents: {e}")
                return False
