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
Base classes and common utilities for integration tests
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Enum for test status"""

    SUCCESS = "Success"
    FAILURE = "Failure"
    NOT_EXECUTED = "Not Executed"


@dataclass
class TestResult:
    """Data class to store test results"""

    test_number: int
    test_name: str
    description: str
    api_endpoints: list[str]
    payload_fields: list[str]
    time_taken: float
    status: TestStatus
    error_message: str | None = None
    test_phase: str | None = None  # "pre-sequence", "main", "post-sequence"


def test_case(test_number: int, test_name: str):
    """Decorator to register individual test case with number and name"""
    def decorator(func):
        func.test_number = test_number
        func.test_name = test_name
        return func
    return decorator


class BaseTestModule:
    """Base class for test cases"""

    def __init__(self, test_runner):
        self.test_runner = test_runner
        self.rag_server_url = test_runner.rag_server_url
        self.ingestor_server_url = test_runner.ingestor_server_url
        self.collections = test_runner.collections
        self.timeout = test_runner.timeout
        self.poll_interval = test_runner.poll_interval

        self.test_files = test_runner.test_files

    def add_test_result(
        self,
        test_number: int,
        test_name: str,
        description: str,
        api_endpoints: list[str],
        payload_fields: list[str],
        time_taken: float,
        status: TestStatus,
        error_message: str | None = None,
        test_phase: str | None = None,
    ):
        """Add a test result to the results list"""
        self.test_runner.add_test_result(
            test_number, test_name, description, api_endpoints,
            payload_fields, time_taken, status, error_message, test_phase
        )