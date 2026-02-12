"""
Unit tests for utility functions.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpk_lammps_ver4 import get_mixing_type, ValidationError


class TestGetMixingType:
    """Test the get_mixing_type function."""
    
    def test_homog_mixing_always_returns_zero(self):
        """Test that homog mixing always returns 0."""
        assert get_mixing_type("homog", 0, 0, 0) == 0
        assert get_mixing_type("homog", 1, 2, 3) == 0
        assert get_mixing_type("homog", 5, 5, 5) == 0
    
    def test_g_type_mixing_pattern(self):
        """Test G-type mixing follows correct pattern."""
        # When (i+j+k) is odd, pow(-1, i+j+k) = -1, should return 0
        assert get_mixing_type("G", 0, 0, 1) == 0  # sum=1 (odd)
        assert get_mixing_type("G", 1, 1, 1) == 0  # sum=3 (odd)
        
        # When (i+j+k) is even, pow(-1, i+j+k) = 1, should return 1
        assert get_mixing_type("G", 0, 0, 0) == 1  # sum=0 (even)
        assert get_mixing_type("G", 1, 1, 0) == 1  # sum=2 (even)
        assert get_mixing_type("G", 2, 2, 2) == 1  # sum=6 (even)
    
    def test_14_type_mixing_all_even(self):
        """Test 14-type mixing when all coordinates are even."""
        assert get_mixing_type("14", 0, 0, 0) == 0
        assert get_mixing_type("14", 2, 2, 2) == 0
        assert get_mixing_type("14", 4, 6, 8) == 0
    
    def test_14_type_mixing_with_odd(self):
        """Test 14-type mixing with at least one odd coordinate."""
        assert get_mixing_type("14", 1, 0, 0) == 1
        assert get_mixing_type("14", 0, 1, 0) == 1
        assert get_mixing_type("14", 0, 0, 1) == 1
        assert get_mixing_type("14", 1, 1, 1) == 1
        assert get_mixing_type("14", 2, 3, 4) == 1
    
    def test_invalid_mixing_type_raises_error(self):
        """Test that invalid mixing type raises ValidationError."""
        with pytest.raises(ValidationError, match="not defined"):
            get_mixing_type("invalid", 0, 0, 0)
        
        with pytest.raises(ValidationError, match="Valid options"):
            get_mixing_type("xyz", 1, 2, 3)
    
    def test_case_sensitivity(self):
        """Test that mixing type is case-sensitive."""
        # "G" should work
        result = get_mixing_type("G", 0, 0, 0)
        assert result in [0, 1]
        
        # "g" should raise error (case sensitive)
        with pytest.raises(ValidationError):
            get_mixing_type("g", 0, 0, 0)
