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
Custom prompt test module
"""

import json
import logging
import time
from typing import Any

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.response_handlers import extract_streaming_text, print_response
from ..utils.verification import verify_response_content

logger = logging.getLogger(__name__)


class CustomPromptModule(BaseTestModule):
    """Custom prompt test module"""

    COLLECTION_NAME = "test_collection_with_metadata"

    @test_case(31, "Generate without Knowledge Base")
    async def _test_generate_without_knowledge_base(self) -> bool:
        """Test /generate endpoint with use_knowledge_base=False"""
        logger.info("\n=== Test 31: Generate without Knowledge Base ===")
        generate_start = time.time()
        generate_success = await self.test_generate_without_knowledge_base()
        generate_time = time.time() - generate_start

        if generate_success:
            self.add_test_result(
                self._test_generate_without_knowledge_base.test_number,
                self._test_generate_without_knowledge_base.test_name,
                "Test RAG generation without knowledge base (use_knowledge_base=False). Query: 'Who are you?'. Expected to contain 'Foundational RAG' keyword in response.",
                ["POST /v1/generate"],
                ["messages", "use_knowledge_base"],
                generate_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_generate_without_knowledge_base.test_number,
                self._test_generate_without_knowledge_base.test_name,
                "Test RAG generation without knowledge base (use_knowledge_base=False). Query: 'Who are you?'. Expected to contain 'Foundational RAG' keyword in response.",
                ["POST /v1/generate"],
                ["messages", "use_knowledge_base"],
                generate_time,
                TestStatus.FAILURE,
                "Generate without knowledge base test failed",
            )
            return False

    @test_case(32, "Generate with Knowledge Base")
    async def _test_generate_with_knowledge_base(self) -> bool:
        """Test /generate endpoint with use_knowledge_base=True"""
        logger.info("\n=== Test 32: Generate with Knowledge Base ===")
        generate_start = time.time()
        generate_success = await self.test_generate_with_knowledge_base()
        generate_time = time.time() - generate_start

        if generate_success:
            self.add_test_result(
                self._test_generate_with_knowledge_base.test_number,
                self._test_generate_with_knowledge_base.test_name,
                "Test RAG generation with knowledge base (use_knowledge_base=True). Query: 'Who are you?'. Expected to contain 'Foundational RAG' keyword in response.",
                ["POST /v1/generate"],
                ["messages", "use_knowledge_base", "collection_names"],
                generate_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_generate_with_knowledge_base.test_number,
                self._test_generate_with_knowledge_base.test_name,
                "Test RAG generation with knowledge base (use_knowledge_base=True). Query: 'Who are you?'. Expected to contain 'Foundational RAG' keyword in response.",
                ["POST /v1/generate"],
                ["messages", "use_knowledge_base", "collection_names"],
                generate_time,
                TestStatus.FAILURE,
                "Generate with knowledge base test failed",
            )
            return False

    async def test_generate_without_knowledge_base(self) -> bool:
        """Test /generate endpoint with use_knowledge_base=False"""
        payload = {
            "messages": [{"role": "user", "content": "Who are you?"}],
            "use_knowledge_base": False,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
            "enable_reranker": False,
            "enable_query_rewriting": False,
            "enable_citations": False,
            "stop": [],
            "collection_names": [self.COLLECTION_NAME],
        }

        async with aiohttp.ClientSession() as session:
            try:
                logger.info("🤖 Generating response without knowledge base")
                logger.info(f"📋 Generate request payload:\n{json.dumps(payload, indent=2)}")

                async with session.post(
                    f"{self.rag_server_url}/v1/generate", json=payload
                ) as response:
                    result = await print_response(response)
                    if response.status == 200:
                        # Check if it's a streaming response
                        if result.get("streaming_response"):
                            logger.info(
                                "✅ Generate without knowledge base test passed (streaming response processed)"
                            )
                        else:
                            logger.info(
                                "✅ Generate without knowledge base test passed"
                            )

                        # Extract response text
                        response_text = ""
                        if result.get("streaming_response"):
                            # Extract text from streaming response
                            response_text = extract_streaming_text(result)
                        else:
                            # Extract text from regular response
                            choices = result.get("choices", [])
                            if choices:
                                response_text = (
                                    choices[0].get("message", {}).get("content", "")
                                )

                        # Verify response contains "Foundational RAG" keyword
                        if response_text:
                            expected_keywords = ["Foundational RAG"]
                            if verify_response_content(
                                response_text, expected_keywords, min_matches=1
                            ):
                                logger.info(
                                    "✅ Response content verification passed - found 'Foundational RAG' keyword"
                                )
                                return True
                            else:
                                logger.error(
                                    "⚠️ Response content verification failed - expected 'Foundational RAG' keyword not found"
                                )
                                logger.error(f"Response text: {response_text}")
                                return False
                        else:
                            logger.error(
                                "⚠️ No response text found for content verification"
                            )
                            return False
                    else:
                        logger.error(f"❌ Generate request failed with status: {response.status}")
                        logger.error(f"Response: {result}")
                        return False
            except Exception as e:
                logger.error(f"❌ Generate without knowledge base test error: {e}")
                return False

    async def test_generate_with_knowledge_base(self) -> bool:
        """Test /generate endpoint with use_knowledge_base=True"""

        payload = {
            "messages": [{"role": "user", "content": "Who are you?"}],
            "use_knowledge_base": True,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
            "reranker_top_k": 2,
            "vdb_top_k": 10,
            "enable_query_rewriting": False,
            "enable_reranker": False,
            "enable_citations": False,
            "stop": [],
            "collection_names": [self.COLLECTION_NAME],
        }

        async with aiohttp.ClientSession() as session:
            try:
                logger.info(f"🤖 Generating response with knowledge base using collection: {self.COLLECTION_NAME}")
                logger.info(f"📋 Generate request payload:\n{json.dumps(payload, indent=2)}")

                async with session.post(
                    f"{self.rag_server_url}/v1/generate", json=payload
                ) as response:
                    result = await print_response(response)
                    if response.status == 200:
                        # Check if it's a streaming response
                        if result.get("streaming_response"):
                            logger.info(
                                "✅ Generate with knowledge base test passed (streaming response processed)"
                            )
                        else:
                            logger.info(
                                "✅ Generate with knowledge base test passed"
                            )

                        # Extract response text
                        response_text = ""
                        if result.get("streaming_response"):
                            # Extract text from streaming response
                            response_text = extract_streaming_text(result)
                        else:
                            # Extract text from regular response
                            choices = result.get("choices", [])
                            if choices:
                                response_text = (
                                    choices[0].get("message", {}).get("content", "")
                                )

                        # Verify response contains "Foundational RAG" keyword
                        if response_text:
                            expected_keywords = ["Foundational RAG"]
                            if verify_response_content(
                                response_text, expected_keywords, min_matches=1
                            ):
                                logger.info(
                                    "✅ Response content verification passed - found 'Foundational RAG' keyword"
                                )
                                return True
                            else:
                                logger.error(
                                    "⚠️ Response content verification failed - expected 'Foundational RAG' keyword not found"
                                )
                                logger.error(f"Response text: {response_text}")
                                return False
                        else:
                            logger.error(
                                "⚠️ No response text found for content verification"
                            )
                            return False
                    else:
                        logger.error(f"❌ Generate request failed with status: {response.status}")
                        logger.error(f"Response: {result}")
                        return False
            except Exception as e:
                logger.error(f"❌ Generate with knowledge base test error: {e}")
                return False
