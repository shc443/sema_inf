#!/usr/bin/env python3
"""
Verification script to debug Java version issues
Run this BEFORE run_simple.py to verify setup
"""

import os
import subprocess
import sys

print("=" * 60)
print("SEMA SETUP VERIFICATION")
print("=" * 60)
print()

# 1. Check git commit
print("1. Git Status:")
print("-" * 60)
result = subprocess.run(['git', 'log', '-1', '--oneline'], capture_output=True, text=True)
print(f"Current commit: {result.stdout.strip()}")
expected_commits = ['8d543e1', '52ed430']
if any(commit in result.stdout for commit in expected_commits):
    print("✅ Git is up to date")
else:
    print("❌ WARNING: Old git commit! Run: git reset --hard origin/main")
print()

# 2. Check colab_cli.py content
print("2. Checking colab_cli.py for Java version:")
print("-" * 60)
try:
    with open('colab_cli.py', 'r') as f:
        content = f.read()
        if 'java-11-openjdk-amd64' in content:
            count = content.count('java-11-openjdk-amd64')
            print(f"✅ Found Java 11 in {count} locations")
        else:
            print("❌ ERROR: Java 11 NOT found in colab_cli.py!")

        if 'java-8-openjdk-amd64' in content:
            count = content.count('java-8-openjdk-amd64')
            print(f"❌ ERROR: Found Java 8 in {count} locations (OLD VERSION!)")
        else:
            print("✅ No Java 8 references found")
except FileNotFoundError:
    print("❌ ERROR: colab_cli.py not found! Wrong directory?")
print()

# 3. Check run_simple.py for reload fix
print("3. Checking run_simple.py for module reload fix:")
print("-" * 60)
try:
    with open('run_simple.py', 'r') as f:
        content = f.read()
        if 'importlib.reload' in content:
            print("✅ Module reload fix present")
        else:
            print("⚠️  Module reload fix NOT found (may need git pull)")
except FileNotFoundError:
    print("❌ ERROR: run_simple.py not found!")
print()

# 4. Check current JAVA_HOME
print("4. Current Environment:")
print("-" * 60)
java_home = os.environ.get('JAVA_HOME', 'NOT SET')
print(f"JAVA_HOME: {java_home}")
if 'java-11' in java_home:
    print("✅ JAVA_HOME points to Java 11")
elif 'java-8' in java_home:
    print("❌ ERROR: JAVA_HOME points to Java 8!")
else:
    print("⚠️  JAVA_HOME not set or unexpected value")
print()

# 5. Check Java installation
print("5. Installed Java Version:")
print("-" * 60)
try:
    result = subprocess.run(['java', '-version'], capture_output=True, text=True, stderr=subprocess.STDOUT)
    output = result.stdout
    print(output)
    if 'openjdk version "11' in output or 'openjdk 11' in output:
        print("✅ Java 11 is installed")
    elif 'openjdk version "8' in output or 'openjdk 1.8' in output:
        print("❌ ERROR: Java 8 is active (wrong version!)")
    else:
        print("⚠️  Java version unclear")
except FileNotFoundError:
    print("❌ Java not installed or not in PATH")
print()

# 6. Check Python module cache
print("6. Python Module Cache:")
print("-" * 60)
if 'colab_cli' in sys.modules:
    print("⚠️  WARNING: colab_cli is already loaded in memory!")
    print("   This means you imported it before running verification.")
    print("   SOLUTION: Restart runtime to clear cache")
    module = sys.modules['colab_cli']
    if hasattr(module, '__file__'):
        print(f"   Loaded from: {module.__file__}")
else:
    print("✅ colab_cli not loaded (clean state)")
print()

# Final verdict
print("=" * 60)
print("FINAL VERDICT:")
print("=" * 60)

issues = []
if 'java-8-openjdk-amd64' in open('colab_cli.py').read():
    issues.append("❌ colab_cli.py has OLD Java 8 code")
if 'java-8' in os.environ.get('JAVA_HOME', ''):
    issues.append("❌ JAVA_HOME points to Java 8")
if 'colab_cli' in sys.modules:
    issues.append("⚠️  Module cache needs clearing (restart runtime)")

if issues:
    print("Issues found:")
    for issue in issues:
        print(f"  {issue}")
    print()
    print("FIX:")
    print("  1. Run: git reset --hard origin/main")
    print("  2. Click: Runtime → Restart runtime")
    print("  3. Re-run from the beginning")
else:
    print("✅ All checks passed! Ready to run run_simple.py")
print()
