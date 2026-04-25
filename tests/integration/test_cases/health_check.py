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
Health check test module
"""

import json
import logging
import time
from typing import Any

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case

logger = logging.getLogger(__name__)


class HealthCheckModule(BaseTestModule):
    """Health check test module"""



    @test_case(1, "Health Checks")
    async def _test_health_checks(self) -> bool:
        """Test health checks"""
        logger.info("\n=== Test 1: Health Checks ===")
        health_start = time.time()

        # Run individual health checks
        rag_health = await self._check_rag_health()
        ingestor_health = await self._check_ingestor_health()

        health_time = time.time() - health_start

        if rag_health and ingestor_health:
            self.add_test_result(
                self._test_health_checks.test_number,
                self._test_health_checks.test_name,
                "Verify both RAG and Ingestion servers are healthy and dependencies are available",
                ["GET /v1/health (RAG)", "GET /v1/health (Ingestor)"],
                ["check_dependencies (RAG only)"],
                health_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_health_checks.test_number,
                self._test_health_checks.test_name,
                "Verify both RAG and Ingestion servers are healthy and dependencies are available",
                ["GET /v1/health (RAG)", "GET /v1/health (Ingestor)"],
                ["check_dependencies (RAG only)"],
                health_time,
                TestStatus.FAILURE,
                "One or both servers failed health check",
            )
            return False

    async def _check_rag_health(self) -> bool:
        """Check RAG server health"""
        try:
            params = {"check_dependencies": "True"}
            logger.info(f"🔍 Checking RAG server health with params: {json.dumps(params, indent=2)}")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.rag_server_url}/v1/health", params=params) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(f"✅ RAG server health check passed:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(f"❌ RAG server health check failed: {response.status}")
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ RAG server health check error: {e}")
            return False

    async def _check_ingestor_health(self) -> bool:
        """Check Ingestor server health"""
        try:
            params = {"check_dependencies": "True"}
            logger.info(f"🔍 Checking Ingestor server health with params: {json.dumps(params, indent=2)}")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ingestor_server_url}/v1/health", params=params) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(f"✅ Ingestor server health check passed:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(f"❌ Ingestor server health check failed: {response.status}")
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Ingestor server health check error: {e}")
            return False