# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE-KKR -- KKR solver module for ODAT-SE
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Unit tests for odatse_kkr module.
"""

import os
import sys
import tempfile
from pathlib import Path

SOURCE_PATH = os.path.join(os.path.dirname(__file__), "../../src")
sys.path.insert(0, SOURCE_PATH)

import pytest


class TestConvergenceError:
    """Tests for ConvergenceError exception."""

    def test_basic_error(self):
        """Test basic ConvergenceError creation."""
        from odatse_kkr import ConvergenceError

        error = ConvergenceError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.ewidth_tried == []
        assert error.attempts == 0

    def test_error_with_ewidth_list(self):
        """Test ConvergenceError with ewidth list."""
        from odatse_kkr import ConvergenceError

        error = ConvergenceError(
            "Convergence failed",
            ewidth_tried=[2.0, 2.5, 3.0],
            attempts=3,
        )
        assert "tried ewidth values: [2.0, 2.5, 3.0]" in str(error)
        assert "after 3 attempt(s)" in str(error)
        assert error.ewidth_tried == [2.0, 2.5, 3.0]
        assert error.attempts == 3


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_basic_config(self):
        """Test basic RetryConfig creation."""
        from odatse_kkr import RetryConfig

        config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0])
        assert config.ewidth_list == [2.0, 2.5, 3.0]
        assert config.max_retries == 2  # len(ewidth_list) - 1
        assert config.change_record_on_retry is True

    def test_config_with_options(self):
        """Test RetryConfig with custom options."""
        from odatse_kkr import RetryConfig

        config = RetryConfig(
            ewidth_list=[2.0, 2.5, 3.0, 3.5],
            max_retries=5,
            change_record_on_retry=False,
        )
        assert config.max_retries == 5
        assert config.change_record_on_retry is False

    def test_empty_ewidth_list_raises(self):
        """Test that empty ewidth_list raises ValueError."""
        from odatse_kkr import RetryConfig

        with pytest.raises(ValueError, match="ewidth_list must not be empty"):
            RetryConfig(ewidth_list=[])

    def test_from_config(self):
        """Test RetryConfig.from_config class method."""
        from odatse_kkr import RetryConfig

        config = {
            "kkr": {
                "retry": {
                    "ewidth_list": [2.0, 2.5, 3.0],
                    "change_record_on_retry": False,
                }
            }
        }
        retry_config = RetryConfig.from_config(config)
        assert retry_config is not None
        assert retry_config.ewidth_list == [2.0, 2.5, 3.0]
        assert retry_config.change_record_on_retry is False

    def test_from_config_no_retry_section(self):
        """Test RetryConfig.from_config returns None when no retry section."""
        from odatse_kkr import RetryConfig

        config = {"kkr": {"calculation": {"ewidth": 2.0}}}
        retry_config = RetryConfig.from_config(config)
        assert retry_config is None

    def test_from_config_empty_ewidth(self):
        """Test RetryConfig.from_config returns None when no ewidth_list."""
        from odatse_kkr import RetryConfig

        config = {"kkr": {"retry": {"change_record_on_retry": True}}}
        retry_config = RetryConfig.from_config(config)
        assert retry_config is None

    def test_get_ewidth(self):
        """Test RetryConfig.get_ewidth method."""
        from odatse_kkr import RetryConfig

        config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0])
        assert config.get_ewidth(0) == 2.0
        assert config.get_ewidth(1) == 2.5
        assert config.get_ewidth(2) == 3.0
        assert config.get_ewidth(3) is None


class TestInputFunctions:
    """Tests for input file handling functions."""

    def test_load_input_file(self):
        """Test load_input_file function."""
        from odatse_kkr import load_input_file

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".in", delete=False
        ) as f:
            f.write("brvtyp=fcc a=7.5\n")
            f.write("edelt=0.001 ewidth=2.0\n")
            f.write("go pot.dat\n")
            temp_path = f.name

        try:
            input_data = load_input_file(temp_path)
            assert input_data is not None
            assert input_data.content is not None
            assert "brvtyp=fcc" in input_data.content
        finally:
            os.unlink(temp_path)

    def test_load_input_file_not_found(self):
        """Test load_input_file raises FileNotFoundError."""
        from odatse_kkr import load_input_file

        with pytest.raises(FileNotFoundError):
            load_input_file("/nonexistent/path/input.in")

    def test_apply_kkr_parameters_from_config(self):
        """Test apply_kkr_parameters_from_config function."""
        from odatse_kkr import apply_kkr_parameters_from_config, load_input_file

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".in", delete=False
        ) as f:
            f.write("brvtyp=fcc a=7.5\n")
            f.write("edelt=0.001 ewidth=2.0\n")
            temp_path = f.name

        try:
            input_data = load_input_file(temp_path)
            config = {
                "kkr": {
                    "calculation": {"ewidth": 3.0, "edelt": 0.0005},
                    "output": {"bzqlty": 2},
                }
            }
            modified_data = apply_kkr_parameters_from_config(input_data, config)

            assert modified_data.get("ewidth") == 3.0
            assert modified_data.get("edelt") == 0.0005
            assert modified_data.get("bzqlty") == 2
        finally:
            os.unlink(temp_path)

    def test_apply_kkr_parameters_no_kkr_section(self):
        """Test apply_kkr_parameters_from_config with no kkr section."""
        from odatse_kkr import apply_kkr_parameters_from_config, load_input_file

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".in", delete=False
        ) as f:
            f.write("brvtyp=fcc a=7.5\n")
            temp_path = f.name

        try:
            input_data = load_input_file(temp_path)
            config = {"solver": {"name": "test"}}
            modified_data = apply_kkr_parameters_from_config(input_data, config)

            # Should return unchanged data
            assert modified_data.content == input_data.content
        finally:
            os.unlink(temp_path)


class TestCheckConvergence:
    """Tests for check_convergence function."""

    def test_converged_output(self):
        """Test check_convergence with converged output."""
        from odatse_kkr import check_convergence

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            f.write("Calculation completed successfully\n")
            f.write("Final energy: -1234.5678\n")
            temp_path = f.name

        try:
            assert check_convergence(temp_path) is True
        finally:
            os.unlink(temp_path)

    def test_not_converged_output(self):
        """Test check_convergence with 'no convergence' in output."""
        from odatse_kkr import check_convergence

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            f.write("Iteration 100\n")
            f.write("no convergence achieved\n")
            f.write("Calculation aborted\n")
            temp_path = f.name

        try:
            assert check_convergence(temp_path) is False
        finally:
            os.unlink(temp_path)

    def test_missing_output_file(self):
        """Test check_convergence with missing file."""
        from odatse_kkr import check_convergence

        assert check_convergence("/nonexistent/output.log") is False


class TestKKRInputData:
    """Tests for KKRInputData class."""

    def test_get_set(self):
        """Test get and set methods."""
        from odatse_kkr.input import KKRInputData

        data = KKRInputData(content="test", parameters={"a": 1, "b": 2})
        assert data.get("a") == 1
        assert data.get("c", "default") == "default"

        data.set("c", 3)
        assert data.get("c") == 3

    def test_copy(self):
        """Test copy method creates deep copy."""
        from odatse_kkr.input import KKRInputData

        data = KKRInputData(content="test", parameters={"a": [1, 2, 3]})
        copied = data.copy()

        # Modify original
        data.parameters["a"].append(4)

        # Copy should be unchanged
        assert copied.get("a") == [1, 2, 3]


class TestImports:
    """Tests for module imports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from odatse_kkr import (
            ConvergenceError,
            RetryConfig,
            apply_kkr_parameters_from_config,
            check_convergence,
            load_input_file,
            run_with_retry,
        )

        assert ConvergenceError is not None
        assert RetryConfig is not None
        assert apply_kkr_parameters_from_config is not None
        assert check_convergence is not None
        assert load_input_file is not None
        assert run_with_retry is not None

    def test_version_available(self):
        """Test that version is available."""
        import odatse_kkr

        assert hasattr(odatse_kkr, "__version__")
        assert odatse_kkr.__version__ == "0.1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

