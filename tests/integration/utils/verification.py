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
Verification utilities for integration tests
"""

import logging
from typing import Any, Dict, List

import aiohttp

logger = logging.getLogger(__name__)


def verify_response_content(
    response_text: str, expected_keywords: list[str], min_matches: int = 1
) -> bool:
    """Verify that response contains expected keywords"""
    if not response_text:
        logger.warning("⚠️ Empty response text for verification")
        return False

    response_lower = response_text.lower()
    matches = sum(
        1 for keyword in expected_keywords if keyword.lower() in response_lower
    )

    logger.info(
        f"Response verification: found {matches}/{len(expected_keywords)} expected keywords"
    )
    logger.info(f"Expected keywords: {expected_keywords}")
    logger.info(f"Response preview: {response_text[:200]}...")

    return matches >= min_matches


def verify_summary_content(summary_text: str, filename: str) -> bool:
    """Verify that summary contains expected keywords based on the file content"""
    if not summary_text:
        logger.warning(f"⚠️ Empty summary text for verification of {filename}")
        return False

    # Define expected keywords for default files
    expected_keywords_map = {
        "multimodal_test.pdf": [
            "table",
            "chart",
            "animal",
            "gadget",
            "speaker",
            "bullet",
            "testing",
            "document",
        ],
        "woods_frost.docx": [
            "frost",
            "woods",
            "snowy",
            "poem",
            "promises",
            "miles",
            "collections",
            "boy's will",
        ],
        "table_test.pdf": ["table", "data", "information", "structured", "content"],
        "embedded_table.pdf": [
            "table",
            "embedded",
            "data",
            "information",
            "structured",
        ],
    }

    # Only verify for default files
    if filename not in expected_keywords_map:
        logger.info(
            f"⚠️ Skipping summary verification for non-default file: {filename}"
        )
        return True

    expected_keywords = expected_keywords_map[filename]
    summary_lower = summary_text.lower()
    matches = sum(
        1 for keyword in expected_keywords if keyword.lower() in summary_lower
    )

    logger.info(
        f"Summary verification for {filename}: found {matches}/{len(expected_keywords)} expected keywords"
    )
    logger.info(f"Expected keywords: {expected_keywords}")
    logger.info(f"Summary preview: {summary_text[:200]}...")

    if matches >= 2:  # Require at least 2 keyword matches for summary verification
        logger.info(f"✅ Summary content verification passed for {filename}")
        return True
    else:
        logger.error(
            f"❌ Summary content verification failed for {filename} - insufficient keyword matches"
        )
        return False


async def verify_citation_document_names(
    results: list[dict], collection_names: list[str], ingestor_server_url: str
) -> bool:
    """Verify that document names in citations are a subset of documents in the specified collections"""
    try:
        # Get all documents from the specified collections
        all_documents = []
        async with aiohttp.ClientSession() as session:
            for collection_name in collection_names:
                params = {"collection_name": collection_name}
                async with session.get(
                    f"{ingestor_server_url}/v1/documents", params=params
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        documents = result.get("documents", [])
                        all_documents.extend(documents)
                    else:
                        logger.error(
                            f"❌ Failed to get documents from collection {collection_name}"
                        )
                        return False

        # Extract document names from collections
        collection_document_names = {
            doc.get("document_name") for doc in all_documents
        }

        # Extract document names from citations
        citation_document_names = set()
        for result in results:
            citation_doc_name = result.get("document_name")
            if not citation_doc_name:
                logger.error(f"❌ Citation result missing document_name: {result}")
                return False
            citation_document_names.add(citation_doc_name)

        # Check if citation document names are a subset of collection document names
        if citation_document_names.issubset(collection_document_names):
            logger.info(
                f"✅ All {len(results)} citation document names are subset of collection documents"
            )
            logger.info(f"Citation documents: {sorted(citation_document_names)}")
            logger.info(
                f"Collection documents: {sorted(collection_document_names)}"
            )
            return True
        else:
            # Find which citation documents are not in collections
            missing_documents = citation_document_names - collection_document_names
            logger.error(
                f"❌ Citation documents not found in collections: {missing_documents}"
            )
            logger.error(f"Citation documents: {sorted(citation_document_names)}")
            logger.error(
                f"Collection documents: {sorted(collection_document_names)}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ Error verifying citation document names: {e}")
        return False


def verify_filtered_citations(
    results: list[dict], expected_file: str
) -> bool:
    """Verify that citations only come from the expected file when using filters"""
    try:
        # Extract document names from citations
        citation_document_names = set()
        for result in results:
            citation_doc_name = result.get("document_name")
            if not citation_doc_name:
                logger.error(f"❌ Citation result missing document_name: {result}")
                return False
            citation_document_names.add(citation_doc_name)

        # Check if all citations come from the expected file
        if (
            len(citation_document_names) == 1
            and expected_file in citation_document_names
        ):
            logger.info(
                f"✅ All {len(results)} citations come from expected file: {expected_file}"
            )
            return True
        else:
            logger.error(
                f"❌ Citations verification failed - expected only '{expected_file}', got: {sorted(citation_document_names)}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ Error verifying filtered citations: {e}")
        return False


def validate_metadata_schema(
    actual_schema: list[dict[str, Any]], collection_name: str, expected_metadata_schema: list[dict[str, Any]]
) -> bool:
    """Validate that the actual metadata schema matches the expected schema"""
    if not actual_schema:
        logger.error(
            f"❌ Metadata schema is empty for collection '{collection_name}'"
        )
        return False

    # Convert schemas to sets of field names for comparison
    expected_fields = {field["name"] for field in expected_metadata_schema}
    actual_fields = {field["name"] for field in actual_schema}

    # Check if all expected fields are present
    missing_fields = expected_fields - actual_fields
    if missing_fields:
        logger.error(
            f"❌ Missing metadata fields in collection '{collection_name}': {missing_fields}"
        )
        logger.error(f"Expected fields: {expected_fields}")
        logger.error(f"Actual fields: {actual_fields}")
        return False

    # Check if all actual fields are expected (no extra fields)
    extra_fields = actual_fields - expected_fields
    if extra_fields:
        if not extra_fields == {"filename"}:  # filename is always present
            logger.warning(
                f"⚠️ Extra metadata fields in collection '{collection_name}': {extra_fields}"
            )
        # This is a warning, not an error, as extra fields are acceptable

    # Validate field properties for each expected field
    for expected_field in expected_metadata_schema:
        field_name = expected_field["name"]
        actual_field = next(
            (f for f in actual_schema if f["name"] == field_name), None
        )

        if not actual_field:
            logger.error(
                f"❌ Field '{field_name}' not found in actual schema for collection '{collection_name}'"
            )
            return False

        # Validate field type
        if actual_field.get("type") != expected_field.get("type"):
            logger.error(
                f"❌ Field '{field_name}' type mismatch in collection '{collection_name}'. Expected: {expected_field.get('type')}, Got: {actual_field.get('type')}"
            )
            return False

        # Validate field description (optional check)
        if expected_field.get("description") and actual_field.get(
            "description"
        ) != expected_field.get("description"):
            logger.warning(
                f"⚠️ Field '{field_name}' description mismatch in collection '{collection_name}'. Expected: {expected_field.get('description')}, Got: {actual_field.get('description')}"
            )
            # This is a warning, not an error, as description might vary

    logger.info(
        f"✅ Metadata schema validation passed for collection '{collection_name}'"
    )
    logger.info(f"Validated fields: {sorted(expected_fields)}")
    return True