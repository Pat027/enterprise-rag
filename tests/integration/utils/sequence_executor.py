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
Test sequence execution engine for integration tests
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import yaml, provide fallback if not available
try:
    import yaml
except ImportError:
    try:
        import PyYAML as yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install PyYAML")

from .discovery import discover_test_cases

logger = logging.getLogger(__name__)


class TestSequenceExecutor:
    """Execute test sequences based on configuration"""

    def __init__(self, config_file: str = "test_sequences.yaml"):
        self.config_file = Path(__file__).parent.parent / config_file
        self.config = self._load_test_sequences()
        self.test_registry = discover_test_cases()

    def _load_test_sequences(self) -> Dict[str, Any]:
        """Load test sequences from configuration file"""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded test sequences from {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load test sequences from {self.config_file}: {e}")
            return {"sequences": {}}

    def get_available_sequences(self) -> Dict[str, Dict[str, Any]]:
        """Get all available test sequences"""
        return self.config.get("sequences", {})

    def resolve_test_numbers(self, sequence: Dict[str, Any]) -> List[int]:
        """Resolve test numbers based on sequence configuration"""
        test_numbers = sequence.get("test_numbers", [])

        if test_numbers == "all":
            # Return all available test numbers in sorted order
            return sorted(list(self.test_registry.keys()))

        # Return test_numbers as-is to preserve execution order
        return test_numbers

    def get_sequence_test_numbers(self, sequence_name: str) -> List[int]:
        """Get test numbers for a specific sequence in execution order"""
        sequences = self.get_available_sequences()
        if sequence_name not in sequences:
            logger.error(f"Sequence '{sequence_name}' not found")
            return []

        sequence = sequences[sequence_name]
        # Return test numbers in the order specified in the config (test_numbers list)
        # This preserves the execution order defined in the configuration
        return self.resolve_test_numbers(sequence)

    def get_sequence_info(self, sequence_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific sequence"""
        sequences = self.get_available_sequences()
        if sequence_name not in sequences:
            return None

        sequence = sequences[sequence_name]
        test_numbers = self.get_sequence_test_numbers(sequence_name)

        return {
            "name": sequence.get("name", sequence_name),
            "description": sequence.get("description", ""),
            "test_numbers": test_numbers,
            "test_count": len(test_numbers)
        }

    def list_sequences(self) -> None:
        """List all available test sequences"""
        sequences = self.get_available_sequences()

        logger.info("\n" + "=" * 80)
        logger.info("AVAILABLE TEST SEQUENCES")
        logger.info("=" * 80)

        for sequence_name, sequence in sequences.items():
            info = self.get_sequence_info(sequence_name)
            if info:
                logger.info(f"\n{sequence_name.upper()}:")
                logger.info(f"  Name: {info['name']}")
                logger.info(f"  Description: {info['description']}")
                logger.info(f"  Test Count: {info['test_count']}")
                logger.info(f"  Test Numbers: {info['test_numbers']}")

                # Show pre-sequence tests
                pre_tests = self.get_pre_sequence_tests(sequence_name)
                if pre_tests:
                    logger.info(f"  Pre-sequence Tests: {pre_tests}")

                # Show post-sequence tests
                post_tests = self.get_post_sequence_tests(sequence_name)
                if post_tests:
                    logger.info(f"  Post-sequence Tests: {post_tests}")

        logger.info("\n" + "=" * 80)

    def validate_sequence(self, sequence_name: str) -> bool:
        """Validate that a sequence exists and has valid test numbers"""
        sequences = self.get_available_sequences()
        if sequence_name not in sequences:
            logger.error(f"Sequence '{sequence_name}' not found")
            return False

        test_numbers = self.get_sequence_test_numbers(sequence_name)
        if not test_numbers:
            logger.error(f"Sequence '{sequence_name}' has no valid test numbers")
            return False

        # Check if all test numbers exist in registry
        missing_tests = []
        for test_num in test_numbers:
            if test_num not in self.test_registry:
                missing_tests.append(test_num)

        if missing_tests:
            logger.error(f"Sequence '{sequence_name}' references non-existent test numbers: {missing_tests}")
            return False

        logger.info(f"Sequence '{sequence_name}' is valid with {len(test_numbers)} tests")
        return True

    def get_pre_sequence_tests(self, sequence_name: str) -> List[int]:
        """Get pre-sequence test numbers for a specific sequence"""
        sequences = self.get_available_sequences()
        if sequence_name not in sequences:
            return []

        sequence = sequences[sequence_name]
        pre_tests = sequence.get("pre_sequence", [])
        return self.resolve_test_numbers({"test_numbers": pre_tests})

    def get_post_sequence_tests(self, sequence_name: str) -> List[int]:
        """Get post-sequence test numbers for a specific sequence"""
        sequences = self.get_available_sequences()
        if sequence_name not in sequences:
            return []

        sequence = sequences[sequence_name]
        post_tests = sequence.get("post_sequence", [])
        return self.resolve_test_numbers({"test_numbers": post_tests})