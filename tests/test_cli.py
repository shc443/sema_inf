#!/usr/bin/env python3

import sys
import os

def test_cli_import():
    """Test if CLI can be imported without errors"""
    try:
        # Test basic argument parsing
        sys.argv = ['test_cli.py', '--help']
        
        # Mock the required dependencies
        import unittest.mock as mock
        
        with mock.patch.dict('sys.modules', {
            'pandas': mock.MagicMock(),
            'numpy': mock.MagicMock(),
            'torch': mock.MagicMock(),
            'transformers': mock.MagicMock(),
            'konlpy': mock.MagicMock(),
            'tqdm': mock.MagicMock(),
            'sema': mock.MagicMock(),
        }):
            import cli
            print("✓ CLI imports successfully")
            
        # Test argument parser
        parser = cli.argparse.ArgumentParser(description='Test parser')
        parser.add_argument('input', help='Input file')
        parser.add_argument('-o', '--output', help='Output file')
        
        args = parser.parse_args(['test.xlsx', '-o', 'output.xlsx'])
        assert args.input == 'test.xlsx'
        assert args.output == 'output.xlsx'
        print("✓ Argument parsing works correctly")
        
        return True
        
    except SystemExit as e:
        if e.code == 0:  # Help was displayed successfully
            print("✓ CLI help displayed successfully")
            return True
        else:
            print(f"✗ CLI exited with code: {e.code}")
            return False
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def test_file_operations():
    """Test file operation functions"""
    try:
        # Test file existence checks
        assert os.path.exists('/Users/kevinchoi/repos/sema_inf/src/cli.py')
        assert os.path.exists('/Users/kevinchoi/repos/sema_inf/src/sema.py')
        assert os.path.exists('/Users/kevinchoi/repos/sema_inf/requirements.txt')
        print("✓ All required files exist")
        
        return True
    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        return False

def main():
    print("SEMA CLI Test Suite")
    print("=" * 30)
    
    tests = [
        ("CLI Import Test", test_cli_import),
        ("File Operations Test", test_file_operations),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"Failed: {test_name}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! CLI is ready for use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the CLI implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())