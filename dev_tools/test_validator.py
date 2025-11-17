#!/usr/bin/env python3
"""
Comprehensive tests for the Sudoku validator.

Tests all validation logic:
- Complete valid solutions
- Invalid solutions (row/column/box duplicates)
- Partial puzzles (valid and invalid)
- Grid comparison (overwrite detection)
"""

import numpy as np
from src.validator import SudokuValidator, print_validation_report


def test_valid_complete_solution():
    """Test a completely valid Sudoku solution."""
    print("\n" + "="*60)
    print("TEST 1: Valid Complete Solution")
    print("="*60)

    # Known valid Sudoku solution
    valid_solution = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])

    is_valid, errors = SudokuValidator.validate_solution(valid_solution, verbose=True)

    assert is_valid, f"Valid solution was marked invalid: {errors}"
    assert len(errors) == 0, f"Valid solution has errors: {errors}"

    print("✅ PASSED: Valid solution correctly validated")
    return True


def test_invalid_row_duplicate():
    """Test solution with duplicate in a row."""
    print("\n" + "="*60)
    print("TEST 2: Invalid - Row Duplicate")
    print("="*60)

    # Row 0 has two 5's
    invalid_row = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 5],  # Two 5's (missing 2)
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])

    is_valid, errors = SudokuValidator.validate_solution(invalid_row, verbose=True)

    assert not is_valid, "Invalid row was not detected"
    assert len(errors) > 0, "No errors reported for row duplicate"
    assert any("Row 0" in e and "Duplicate" in e for e in errors), \
        f"Row 0 duplicate not detected in errors: {errors}"

    print("✅ PASSED: Row duplicate correctly detected")
    return True


def test_invalid_column_duplicate():
    """Test solution with duplicate in a column."""
    print("\n" + "="*60)
    print("TEST 3: Invalid - Column Duplicate")
    print("="*60)

    # Column 0 has two 5's
    invalid_col = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [5, 2, 6, 8, 5, 3, 7, 9, 1],  # First cell is 5 (duplicate in col 0, missing 4)
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])

    is_valid, errors = SudokuValidator.validate_solution(invalid_col, verbose=True)

    assert not is_valid, "Invalid column was not detected"
    assert len(errors) > 0, "No errors reported for column duplicate"
    assert any("Column 0" in e and "Duplicate" in e for e in errors), \
        f"Column 0 duplicate not detected in errors: {errors}"

    print("✅ PASSED: Column duplicate correctly detected")
    return True


def test_invalid_box_duplicate():
    """Test solution with duplicate in a 3x3 box."""
    print("\n" + "="*60)
    print("TEST 4: Invalid - Box Duplicate")
    print("="*60)

    # Top-left box (0,0) has two 5's
    invalid_box = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [5, 9, 8, 3, 4, 2, 1, 6, 7],  # First cell is 5 (duplicate in box 0,0)
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])

    is_valid, errors = SudokuValidator.validate_solution(invalid_box, verbose=True)

    assert not is_valid, "Invalid box was not detected"
    assert len(errors) > 0, "No errors reported for box duplicate"
    assert any("Box (0,0)" in e and "Duplicate" in e for e in errors), \
        f"Box (0,0) duplicate not detected in errors: {errors}"

    print("✅ PASSED: Box duplicate correctly detected")
    return True


def test_incomplete_solution():
    """Test incomplete solution (has zeros)."""
    print("\n" + "="*60)
    print("TEST 5: Incomplete Solution")
    print("="*60)

    # Valid but incomplete (has empty cells)
    incomplete = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 0],  # Empty cell
        [2, 8, 7, 4, 1, 9, 6, 0, 5],  # Empty cell
        [3, 4, 5, 2, 8, 6, 1, 7, 0]   # Empty cell
    ])

    is_valid, errors = SudokuValidator.validate_solution(incomplete, verbose=True)

    assert not is_valid, "Incomplete solution was marked valid"
    assert len(errors) > 0, "No errors reported for incomplete solution"
    assert any("incomplete" in e.lower() or "empty" in e.lower() for e in errors), \
        f"Incomplete solution error not found: {errors}"

    print("✅ PASSED: Incomplete solution correctly detected")
    return True


def test_valid_partial_puzzle():
    """Test valid partial puzzle (no conflicts)."""
    print("\n" + "="*60)
    print("TEST 6: Valid Partial Puzzle")
    print("="*60)

    # Partial puzzle with no conflicts
    partial_valid = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])

    is_valid, errors = SudokuValidator.validate_puzzle(partial_valid, verbose=True)

    assert is_valid, f"Valid partial puzzle was marked invalid: {errors}"
    assert len(errors) == 0, f"Valid partial puzzle has errors: {errors}"

    print("✅ PASSED: Valid partial puzzle correctly validated")
    return True


def test_invalid_partial_puzzle():
    """Test invalid partial puzzle (has conflicts)."""
    print("\n" + "="*60)
    print("TEST 7: Invalid Partial Puzzle (Conflicts)")
    print("="*60)

    # Partial puzzle with row conflict (two 5's in row 0)
    partial_invalid = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 5],  # Two 5's in row 0
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])

    is_valid, errors = SudokuValidator.validate_puzzle(partial_invalid, verbose=True)

    assert not is_valid, "Invalid partial puzzle was not detected"
    assert len(errors) > 0, "No errors reported for partial puzzle conflict"
    assert any("Row 0" in e and "Duplicate" in e for e in errors), \
        f"Row 0 conflict not detected in errors: {errors}"

    print("✅ PASSED: Partial puzzle conflict correctly detected")
    return True


def test_grid_comparison_no_overwrites():
    """Test grid comparison with no overwrites."""
    print("\n" + "="*60)
    print("TEST 8: Grid Comparison - No Overwrites")
    print("="*60)

    # Original puzzle
    original = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])

    # Solved (all original values preserved)
    solved = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])

    comparison = SudokuValidator.compare_grids(original, solved, verbose=True)

    assert comparison['overwrite_count'] == 0, \
        f"No overwrites expected, found {comparison['overwrite_count']}"
    assert comparison['cells_filled_by_solver'] == 51, \
        f"Expected 51 cells filled by solver, got {comparison['cells_filled_by_solver']}"

    print("✅ PASSED: No overwrites correctly detected")
    return True


def test_grid_comparison_with_overwrites():
    """Test grid comparison with overwrites."""
    print("\n" + "="*60)
    print("TEST 9: Grid Comparison - With Overwrites")
    print("="*60)

    # Original puzzle
    original = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])

    # "Solved" with overwrites (changed cell (0,0) from 5 to 1, and (1,0) from 6 to 2)
    solved_with_overwrites = np.array([
        [1, 3, 4, 6, 7, 8, 9, 1, 2],  # (0,0): 5→1
        [2, 7, 2, 1, 9, 5, 3, 4, 8],  # (1,0): 6→2
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])

    comparison = SudokuValidator.compare_grids(original, solved_with_overwrites, verbose=True)

    assert comparison['overwrite_count'] == 2, \
        f"Expected 2 overwrites, found {comparison['overwrite_count']}"

    # Check specific overwrites
    overwrites = comparison['overwrites']
    positions = [ow['position'] for ow in overwrites]

    assert (0, 0) in positions, "Overwrite at (0,0) not detected"
    assert (1, 0) in positions, "Overwrite at (1,0) not detected"

    print("✅ PASSED: Overwrites correctly detected")
    return True


def test_validation_report():
    """Test validation report generation."""
    print("\n" + "="*60)
    print("TEST 10: Validation Report")
    print("="*60)

    # Valid complete solution
    valid_solution = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])

    report = SudokuValidator.get_validation_report(valid_solution)

    assert report['is_valid'] == True, "Valid solution report shows invalid"
    assert report['total_errors'] == 0, f"Valid solution has errors: {report['total_errors']}"
    assert report['filled_cells'] == 81, f"Expected 81 filled cells, got {report['filled_cells']}"
    assert report['empty_cells'] == 0, f"Expected 0 empty cells, got {report['empty_cells']}"
    assert report['completion'] == 100.0, f"Expected 100% completion, got {report['completion']}"

    print_validation_report(report)

    print("✅ PASSED: Validation report correctly generated")
    return True


def run_all_tests():
    """Run all validator tests."""
    print("\n" + "🔬" * 30)
    print("SUDOKU VALIDATOR - COMPREHENSIVE TEST SUITE")
    print("🔬" * 30)

    tests = [
        ("Valid Complete Solution", test_valid_complete_solution),
        ("Invalid Row Duplicate", test_invalid_row_duplicate),
        ("Invalid Column Duplicate", test_invalid_column_duplicate),
        ("Invalid Box Duplicate", test_invalid_box_duplicate),
        ("Incomplete Solution", test_incomplete_solution),
        ("Valid Partial Puzzle", test_valid_partial_puzzle),
        ("Invalid Partial Puzzle", test_invalid_partial_puzzle),
        ("Grid Comparison - No Overwrites", test_grid_comparison_no_overwrites),
        ("Grid Comparison - With Overwrites", test_grid_comparison_with_overwrites),
        ("Validation Report", test_validation_report),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            failed += 1

    # Final summary
    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe SudokuValidator is working correctly!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
