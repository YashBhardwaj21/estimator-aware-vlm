#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SRC="${SCRIPT_DIR}/tests/test_native_landmark_management.cpp"
BUILD_BIN="/tmp/test_native_landmark_management"

echo "=== Compiling Native ORB-SLAM3 Landmark Management Tests ==="
g++ -O3 -std=c++14 -I/usr/include/eigen3 "${TEST_SRC}" -o "${BUILD_BIN}"

echo "=== Executing Native Tests ==="
"${BUILD_BIN}"
echo "=== Native Test Execution Complete ==="

