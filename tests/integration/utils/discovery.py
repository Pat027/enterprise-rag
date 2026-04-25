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
Auto-discovery utilities for integration tests
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

from tests.integration.base import BaseTestModule

logger = logging.getLogger(__name__)


def discover_test_modules() -> List[Type[BaseTestModule]]:
    """Auto-discover all test modules extending BaseTestModule"""
    test_modules = []
    test_cases_dir = Path(__file__).parent.parent / "test_cases"

    # Scan all .py files in test_cases directory
    for py_file in test_cases_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue

        module_name = py_file.stem
        try:
            # Import the module using absolute import
            module_path = f"tests.integration.test_cases.{module_name}"
            module = importlib.import_module(module_path)

            # Find classes that extend BaseTestModule
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseTestModule) and
                    obj != BaseTestModule and
                    obj.__module__ == module.__name__):
                    test_modules.append(obj)
                    logger.debug(f"Discovered test module: {obj.__name__}")

        except Exception as e:
            logger.warning(f"Failed to import module {module_name}: {e}")

    logger.info(f"Discovered {len(test_modules)} test modules")
    return test_modules


def discover_test_cases() -> Dict[int, Tuple[Type[BaseTestModule], str, str]]:
    """Auto-discover all test cases using @test_case decorator"""
    test_registry = {}
    test_modules = discover_test_modules()

    for module_class in test_modules:
        # Get all methods of the module class
        for method_name, method_obj in inspect.getmembers(module_class, inspect.isfunction):
            # Check if method has test_case decorator attributes
            if hasattr(method_obj, 'test_number') and hasattr(method_obj, 'test_name'):
                test_number = method_obj.test_number
                test_name = method_obj.test_name

                if test_number in test_registry:
                    existing_module, existing_method, existing_name = test_registry[test_number]
                    logger.warning(f"Duplicate test number {test_number}: "
                                 f"{existing_module.__name__}.{existing_method} vs {module_class.__name__}.{method_name}")
                else:
                    test_registry[test_number] = (module_class, method_name, test_name)
                    logger.debug(f"Discovered test case {test_number}: {test_name} in {module_class.__name__}.{method_name}")

    logger.info(f"Discovered {len(test_registry)} test cases")
    return test_registry



