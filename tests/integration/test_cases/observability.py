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
Integration tests for observability features.
Tests the metrics endpoint and Prometheus integration with actual RAG operations to verify metrics collection and observability.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import aiohttp

from ..base import BaseTestModule, test_case

logger = logging.getLogger(__name__)


class ObservabilityModule(BaseTestModule):
    """Integration tests for observability features including metrics endpoint and Prometheus integration"""

    def __init__(self, test_runner):
        super().__init__(test_runner)
        self.rag_base_url = f"{self.rag_server_url}/v1"
        self.metrics_url = f"{self.rag_server_url}/metrics"
        self.prometheus_url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")

        # Use existing collections created by tests 2, 4, 5
        # These will be available when observability tests run after basic setup
        self.test_collection = self.collections["with_metadata"]

    @test_case(52, "Metrics Endpoint After RAG Operations")
    async def _test_metrics_after_rag_operations(self) -> bool:
        """Test metrics endpoint after performing RAG operations"""
        logger.info("\n=== Test 52: Metrics Endpoint After RAG Operations ===")

        try:
            # First, get baseline metrics
            baseline_metrics = await self._get_metrics_content()
            baseline_count = self._count_metrics_lines(baseline_metrics)

            # Perform some RAG operations to generate metrics
            # Uses existing collection created by tests 2, 4, 5
            await self._perform_rag_operations()

            # Wait a moment for metrics to be updated
            # Allow time for metrics to be generated and available
            metrics_delay = int(os.environ.get("METRICS_UPDATE_DELAY", "3"))
            await asyncio.sleep(metrics_delay)

            # Get metrics after operations
            updated_metrics = await self._get_metrics_content()
            updated_count = self._count_metrics_lines(updated_metrics)

            # Define expected metrics for validation
            expected_metrics = [
                "api_requests_total",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "rag_ttft_ms",
                "llm_ttft_ms",
            ]

            # Parse and compare specific metric values for more reliable validation
            baseline_values = self._parse_metric_values(
                baseline_metrics, expected_metrics
            )
            updated_values = self._parse_metric_values(
                updated_metrics, expected_metrics
            )

            # Verify at least some metrics increased
            increased_metrics = 0
            for metric in expected_metrics:
                if metric in updated_values:
                    if metric in baseline_values:
                        # Metric existed before and after - check if it increased
                        if updated_values[metric] > baseline_values[metric]:
                            increased_metrics += 1
                            logger.info(
                                f"✓ {metric} increased: {baseline_values[metric]} -> {updated_values[metric]}"
                            )
                    else:
                        # Metric appeared after operations (wasn't there before)
                        increased_metrics += 1
                        logger.info(
                            f"✓ {metric} appeared after operations: {updated_values[metric]}"
                        )

            # If no metrics increased or appeared, fall back to the original line count logic
            if increased_metrics == 0:
                logger.warning(
                    "No metrics increased or appeared, falling back to line count validation"
                )
                assert updated_count >= baseline_count, (
                    f"Metrics count should increase after operations: {baseline_count} -> {updated_count}"
                )
            else:
                logger.info(
                    f"✓ {increased_metrics} metrics increased or appeared after operations"
                )

            found_metrics = []
            for metric in expected_metrics:
                if metric in updated_metrics:
                    found_metrics.append(metric)

            logger.info(
                f"✓ Found {len(found_metrics)}/{len(expected_metrics)} expected metrics: {found_metrics}"
            )
            logger.info(
                f"✓ Metrics lines increased from {baseline_count} to {updated_count}"
            )

            # Configurable minimum metrics threshold
            min_metrics_threshold = int(os.environ.get("MIN_METRICS_THRESHOLD", "3"))
            return len(found_metrics) >= min_metrics_threshold

        except Exception as e:
            logger.error(f"✗ Metrics after RAG operations test failed: {str(e)}")
            return False

    @test_case(53, "Metrics Endpoint Basic Functionality")
    async def _test_metrics_endpoint_basic(self) -> bool:
        """Test basic /metrics endpoint functionality"""
        logger.info("\n=== Test 53: Metrics Endpoint Basic Functionality ===")

        try:
            async with aiohttp.ClientSession() as session:
                # Make request to metrics endpoint
                async with session.get(self.metrics_url) as response:
                    assert response.status == 200, (
                        f"Expected status 200, got {response.status}"
                    )

                    # Verify content type
                    content_type = response.headers.get("content-type", "")
                    assert "text/plain" in content_type, (
                        f"Expected text/plain content type, got {content_type}"
                    )

                    # Get metrics content
                    metrics_content = await response.text()

                    # Verify metrics content (may be empty initially before RAG operations)
                    # This test now runs after RAG operations, so metrics should be present
                    if len(metrics_content) == 0:
                        logger.warning(
                            "⚠️ Metrics content is empty - this may indicate no RAG operations have been performed yet"
                        )

                    # Verify basic Prometheus format
                    if len(metrics_content) > 0:
                        assert "# HELP" in metrics_content, (
                            "Metrics should contain Prometheus HELP headers"
                        )
                        assert "# TYPE" in metrics_content, (
                            "Metrics should contain Prometheus TYPE headers"
                        )

                    # Verify that tracing is enabled by checking for OpenTelemetry metrics
                    # These should be present if tracing is properly configured
                    otel_indicators = [
                        "api_requests_total",
                        "input_tokens",
                        "output_tokens",
                    ]

                    found_otel_indicators = [
                        indicator
                        for indicator in otel_indicators
                        if indicator in metrics_content
                    ]

                    logger.info(
                        f"✓ Metrics endpoint returned {len(metrics_content)} bytes of data"
                    )
                    logger.info(f"✓ Content type: {content_type}")
                    logger.info(
                        f"✓ Found {len(found_otel_indicators)}/{len(otel_indicators)} OpenTelemetry indicators: {found_otel_indicators}"
                    )

                    return True

        except Exception as e:
            logger.error(f"✗ Metrics endpoint basic test failed: {str(e)}")
            return False

    @test_case(54, "Metrics Endpoint Error Handling")
    async def _test_metrics_endpoint_error_handling(self) -> bool:
        """Test metrics endpoint error handling"""
        logger.info("\n=== Test 54: Metrics Endpoint Error Handling ===")

        try:
            # Test with invalid endpoint path
            invalid_url = f"{self.test_runner.rag_server_url}/invalid-metrics"

            async with aiohttp.ClientSession() as session:
                async with session.get(invalid_url) as response:
                    # Should return 404 for invalid path
                    assert response.status == 404, (
                        f"Expected 404 for invalid path, got {response.status}"
                    )

            # Test metrics endpoint with different HTTP methods
            async with aiohttp.ClientSession() as session:
                # POST should not be allowed
                async with session.post(self.metrics_url) as response:
                    assert response.status == 405, (
                        f"Expected 405 for POST method, got {response.status}"
                    )

                # PUT should not be allowed
                async with session.put(self.metrics_url) as response:
                    assert response.status == 405, (
                        f"Expected 405 for PUT method, got {response.status}"
                    )

                # DELETE should not be allowed
                async with session.delete(self.metrics_url) as response:
                    assert response.status == 405, (
                        f"Expected 405 for DELETE method, got {response.status}"
                    )

                # PATCH should not be allowed
                async with session.patch(self.metrics_url) as response:
                    assert response.status == 405, (
                        f"Expected 405 for PATCH method, got {response.status}"
                    )

            logger.info("✓ Metrics endpoint error handling works correctly")
            return True

        except Exception as e:
            logger.error(f"✗ Metrics endpoint error handling test failed: {str(e)}")
            return False

    @test_case(55, "Metrics Content Validation")
    async def _test_metrics_content_validation(self) -> bool:
        """Test that metrics content follows Prometheus format"""
        logger.info("\n=== Test 55: Metrics Content Validation ===")

        try:
            metrics_content = await self._get_metrics_content()

            # Split into lines and validate format
            lines = metrics_content.strip().split("\n")
            valid_lines = 0
            help_lines = 0
            type_lines = 0

            # Look for specific Prometheus metrics that should be present
            expected_metric_patterns = [
                r"^# HELP\s+",
                r"^# TYPE\s+",
                r"^[a-zA-Z_:][a-zA-Z0-9_:]*\s+[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\s*$",
                r"^[a-zA-Z_:][a-zA-Z0-9_:]*\{[^}]*\}\s+[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\s*$",
            ]

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    if line.startswith("# HELP"):
                        help_lines += 1
                    elif line.startswith("# TYPE"):
                        type_lines += 1
                    continue

                # Check if line matches any expected metric pattern
                is_valid_metric = False
                for pattern in expected_metric_patterns:
                    if re.match(pattern, line):
                        is_valid_metric = True
                        break

                if is_valid_metric:
                    valid_lines += 1

            # Verify we have some valid metrics
            assert valid_lines > 0, (
                f"Should have at least some valid metric lines, found {valid_lines}"
            )
            assert help_lines > 0 or type_lines > 0, (
                f"Should have some HELP or TYPE lines, found {help_lines} HELP and {type_lines} TYPE"
            )

            # Check for specific metrics that should be present in a running RAG server
            rag_metrics_found = 0
            rag_metric_patterns = [
                "api_requests_total",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "rag_ttft_ms",
                "llm_ttft_ms",
                "avg_words_per_chunk",
            ]

            for pattern in rag_metric_patterns:
                if pattern in metrics_content:
                    rag_metrics_found += 1

            logger.info(f"✓ Found {valid_lines} valid metric lines")
            logger.info(f"✓ Found {help_lines} HELP lines and {type_lines} TYPE lines")
            logger.info(
                f"✓ Found {rag_metrics_found}/{len(rag_metric_patterns)} RAG-specific metrics"
            )
            logger.info(f"✓ Total lines processed: {len(lines)}")

            return True

        except Exception as e:
            logger.error(f"✗ Metrics content validation test failed: {str(e)}")
            return False

    @test_case(56, "Metrics Endpoint Performance")
    async def _test_metrics_endpoint_performance(self) -> bool:
        """Test metrics endpoint performance and response time"""
        logger.info("\n=== Test 56: Metrics Endpoint Performance ===")

        try:
            response_times = []

            # Make multiple requests to test performance
            for i in range(5):
                start_time = time.time()

                async with aiohttp.ClientSession() as session:
                    async with session.get(self.metrics_url) as response:
                        assert response.status == 200
                        content = await response.text()
                        assert len(content) > 0

                response_time = time.time() - start_time
                response_times.append(response_time)

                logger.info(f"Request {i + 1}: {response_time:.3f}s")

            # Calculate statistics
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)

            # Get thresholds from environment or use defaults
            avg_threshold = float(os.environ.get("METRICS_AVG_RESPONSE_TIME", "1.0"))
            max_threshold = float(os.environ.get("METRICS_MAX_RESPONSE_TIME", "2.0"))

            # Verify performance is reasonable
            assert avg_response_time < avg_threshold, (
                f"Average response time too slow: {avg_response_time:.3f}s (threshold: {avg_threshold}s)"
            )
            assert max_response_time < max_threshold, (
                f"Max response time too slow: {max_response_time:.3f}s (threshold: {max_threshold}s)"
            )

            logger.info(f"✓ Average response time: {avg_response_time:.3f}s")
            logger.info(f"✓ Min response time: {min_response_time:.3f}s")
            logger.info(f"✓ Max response time: {max_response_time:.3f}s")

            return True

        except Exception as e:
            logger.error(f"✗ Metrics endpoint performance test failed: {str(e)}")
            return False

    async def _get_metrics_content(self) -> str:
        """Helper method to get metrics content"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.metrics_url, timeout=10) as response:
                    assert response.status == 200, (
                        f"Failed to get metrics: {response.status}"
                    )
                    return await response.text()
        except Exception as e:
            logger.error(f"Failed to fetch metrics content: {e}")
            raise

    def _count_metrics_lines(self, metrics_content: str) -> int:
        """Helper method to count non-empty metric lines"""
        lines = [line.strip() for line in metrics_content.split("\n") if line.strip()]
        return len(lines)

    def _parse_metric_values(
        self, metrics_content: str, expected_metrics: list
    ) -> dict:
        """Helper method to parse specific metric values from Prometheus format"""
        metric_values = {}
        lines = metrics_content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse metric lines in format: metric_name{labels} value timestamp
            # or: metric_name value timestamp
            for metric in expected_metrics:
                if line.startswith(metric):
                    try:
                        # Extract the value (second to last part)
                        parts = line.split()
                        if len(parts) >= 2:
                            value_str = parts[-2] if parts[-1].isdigit() else parts[-1]
                            value = float(value_str)
                            metric_values[metric] = value
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
                    break

        return metric_values

    async def _perform_rag_operations(self):
        """Helper method to perform RAG operations that generate metrics using existing collection"""
        try:
            # Use existing collection created by tests 2, 4, 5
            collection_name = self.test_collection

            # Perform a search operation
            await self._perform_test_search(collection_name)

            # Perform a generate operation
            await self._perform_test_generate(collection_name)

            logger.info(
                f"✓ Performed RAG operations on existing collection: {collection_name}"
            )

        except Exception as e:
            logger.error(f"Failed to perform RAG operations: {e}")
            raise e

    async def _perform_test_search(self, collection_name: str):
        """Perform a test search operation"""
        url = f"{self.rag_base_url}/search"
        data = {
            "query": "test document",
            "collection_names": [collection_name],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                assert response.status == 200, f"Search failed: {response.status}"

    async def _perform_test_generate(self, collection_name: str):
        """Perform a test generate operation"""
        url = f"{self.rag_base_url}/generate"
        data = {
            "messages": [{"role": "user", "content": "What is this document about?"}],
            "collection_names": [collection_name],
            "use_knowledge_base": True,
            "temperature": 0.7,
            "reranker_top_k": 1,  # Only get 1 document
            "vdb_top_k": 1,  # Only retrieve 1 document
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                assert response.status == 200, f"Generate failed: {response.status}"

                # Read the streaming response
                response_chunks = []
                async for line in response.content:
                    decoded_line = line.decode("utf-8")
                    response_chunks.append(decoded_line)
                    if "[DONE]" in decoded_line:
                        break

                logger.info(
                    f"✓ Generate response received: {len(response_chunks)} chunks"
                )

    @test_case(57, "Prometheus API Health Check")
    async def _test_prometheus_health_check(self) -> bool:
        """Test that Prometheus API is accessible and healthy"""
        logger.info("\n=== Test 57: Prometheus API Health Check ===")

        try:
            # Wait for Prometheus to be ready
            logger.info("Waiting for Prometheus to be ready...")
            max_wait_time = int(os.environ.get("PROMETHEUS_READINESS_TIMEOUT", "60"))
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{self.prometheus_url}/api/v1/query?query=up", timeout=5
                        ) as response:
                            if response.status == 200:
                                logger.info("Prometheus is ready!")
                                break
                except Exception:
                    pass
                await asyncio.sleep(2)
            else:
                logger.error(
                    f"Prometheus failed to start within {max_wait_time} seconds"
                )
                return False

            # Wait for RAG server metrics to appear in Prometheus
            logger.info("Waiting for RAG server metrics to appear...")
            for i in range(10):
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.prometheus_url}/api/v1/query?query=api_requests_total"
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if (
                                data["status"] == "success"
                                and len(data["data"]["result"]) > 0
                            ):
                                logger.info("RAG server metrics detected!")
                                break
                await asyncio.sleep(2)

            # Test basic Prometheus API query
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.prometheus_url}/api/v1/query?query=up"
                ) as response:
                    assert response.status == 200, (
                        f"Prometheus API failed: {response.status}"
                    )

                    data = await response.json()
                    assert data["status"] == "success", (
                        f"Prometheus query failed: {data}"
                    )
                    assert len(data["data"]["result"]) > 0, (
                        "Prometheus is not collecting metrics"
                    )

                    logger.info("✓ Prometheus API is healthy and collecting metrics")
                    logger.info(f"✓ Found {len(data['data']['result'])} targets")

                    return True

        except Exception as e:
            logger.error(f"✗ Prometheus health check failed: {str(e)}")
            return False

    @test_case(58, "RAG Server Metrics in Prometheus")
    async def _test_rag_metrics_in_prometheus(self) -> bool:
        """Test that RAG server metrics are being collected by Prometheus"""
        logger.info("\n=== Test 58: RAG Server Metrics in Prometheus ===")

        try:
            # Wait for Prometheus to scrape metrics from RAG server
            # Prometheus typically has a scraping interval of 5-15 seconds
            scraping_delay = int(os.environ.get("PROMETHEUS_SCRAPING_DELAY", "10"))
            logger.info(
                f"Waiting {scraping_delay} seconds for Prometheus to scrape metrics..."
            )
            await asyncio.sleep(scraping_delay)

            # Test for RAG server metrics in Prometheus
            rag_metrics_queries = [
                "api_requests_total",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "rag_ttft_ms_count",
                "llm_ttft_ms_count",
            ]

            metrics_found = 0
            async with aiohttp.ClientSession() as session:
                for metric in rag_metrics_queries:
                    try:
                        async with session.get(
                            f"{self.prometheus_url}/api/v1/query?query={metric}",
                            timeout=10,
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                if (
                                    data["status"] == "success"
                                    and len(data["data"]["result"]) > 0
                                ):
                                    logger.info(f"✓ Found metric: {metric}")
                                    metrics_found += 1
                                else:
                                    logger.warning(
                                        f"⚠️  Metric not found or no data: {metric}"
                                    )
                            else:
                                logger.warning(
                                    f"❌ Failed to query metric {metric}: {response.status}"
                                )
                    except Exception as e:
                        logger.warning(f"❌ Error querying metric {metric}: {e}")

            logger.info(
                f"Found {metrics_found}/{len(rag_metrics_queries)} expected RAG server metrics"
            )

            # We should find at least 3 metrics (api_requests_total, input_tokens, output_tokens)
            return metrics_found >= 3

        except Exception as e:
            logger.error(f"✗ RAG metrics in Prometheus test failed: {str(e)}")
            return False

    @test_case(59, "Prometheus Metrics Endpoint Integration")
    async def _test_prometheus_metrics_endpoint_integration(self) -> bool:
        """Test that RAG server metrics endpoint is properly scraped by Prometheus"""
        logger.info("\n=== Test 59: Prometheus Metrics Endpoint Integration ===")

        try:
            # Test that Prometheus can successfully scrape the RAG server metrics endpoint
            async with aiohttp.ClientSession() as session:
                # Check Prometheus targets to ensure RAG server is being scraped
                async with session.get(
                    f"{self.prometheus_url}/api/v1/targets"
                ) as response:
                    assert response.status == 200, (
                        f"Failed to get Prometheus targets: {response.status}"
                    )

                    data = await response.json()
                    assert data["status"] == "success", "Prometheus targets API failed"

                    # Find RAG server target
                    rag_target_found = False
                    rag_instance_pattern = os.environ.get(
                        "RAG_INSTANCE_PATTERN", "rag-server"
                    )
                    for target in data["data"]["activeTargets"]:
                        if rag_instance_pattern in target.get("labels", {}).get(
                            "instance", ""
                        ):
                            rag_target_found = True
                            assert target["health"] == "up", (
                                f"RAG server target is not healthy: {target['health']}"
                            )
                            logger.info(
                                f"✓ RAG server target found: {target['labels']['instance']}"
                            )
                            break

                    assert rag_target_found, "RAG server target not found in Prometheus"

                    # Test that metrics are being scraped by querying for a specific metric
                    async with session.get(
                        f'{self.prometheus_url}/api/v1/query?query=up{{job="rag-app"}}'
                    ) as response:
                        assert response.status == 200, (
                            f"Failed to query RAG server metrics: {response.status}"
                        )

                        data = await response.json()
                        assert data["status"] == "success", "Prometheus query failed"
                        assert len(data["data"]["result"]) > 0, (
                            "No RAG server metrics found in Prometheus"
                        )

                        logger.info(
                            "✓ RAG server metrics successfully scraped by Prometheus"
                        )
                        logger.info(
                            f"✓ Found {len(data['data']['result'])} RAG server targets"
                        )

                    return True

        except Exception as e:
            logger.error(
                f"✗ Prometheus metrics endpoint integration test failed: {str(e)}"
            )
            return False

    @test_case(60, "Prometheus Query Performance")
    async def _test_prometheus_query_performance(self) -> bool:
        """Test Prometheus query performance"""
        logger.info("\n=== Test 60: Prometheus Query Performance ===")

        try:
            query_times = []
            test_queries = ["up", "api_requests_total", "input_tokens", "output_tokens"]

            async with aiohttp.ClientSession() as session:
                for query in test_queries:
                    start_time = time.time()

                    try:
                        async with session.get(
                            f"{self.prometheus_url}/api/v1/query?query={query}",
                            timeout=10,
                        ) as response:
                            if response.status == 200:
                                # Consume response but don't store
                                await response.json()
                                query_time = time.time() - start_time
                                query_times.append(query_time)
                                logger.info(f"Query '{query}': {query_time:.3f}s")
                            else:
                                logger.warning(
                                    f"Query '{query}' failed: {response.status}"
                                )
                    except Exception as e:
                        logger.warning(f"Query '{query}' error: {e}")

            if query_times:
                avg_time = sum(query_times) / len(query_times)
                max_time = max(query_times)

                # Get thresholds from environment or use defaults
                avg_threshold = float(
                    os.environ.get("PROMETHEUS_AVG_QUERY_TIME", "2.0")
                )
                max_threshold = float(
                    os.environ.get("PROMETHEUS_MAX_QUERY_TIME", "5.0")
                )

                # Verify performance is reasonable
                assert avg_time < avg_threshold, (
                    f"Average query time too slow: {avg_time:.3f}s (threshold: {avg_threshold}s)"
                )
                assert max_time < max_threshold, (
                    f"Max query time too slow: {max_time:.3f}s (threshold: {max_threshold}s)"
                )

                logger.info(f"✓ Average query time: {avg_time:.3f}s")
                logger.info(f"✓ Max query time: {max_time:.3f}s")

                return True
            else:
                logger.warning("No successful queries to measure performance")
                return False

        except Exception as e:
            logger.error(f"✗ Prometheus query performance test failed: {str(e)}")
            return False
