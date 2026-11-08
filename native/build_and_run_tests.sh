#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SRC="${SCRIPT_DIR}/tests/test_native_landmark_management.cpp"
BUILD_BIN="/tmp/test_native_landmark_management"
ORB_ROOT="${ORB_SLAM3_ROOT:-/home/saturn/ORB_SLAM3}"

echo "=== Compiling Genuine Native ORB-SLAM3 Landmark Management Tests ==="
g++ -O3 -std=c++14 \
    -DCOMPILEDWITHC11 -DELPP_NO_DEFAULT_LOG_FILE -DHAVE_EIGEN -DHAVE_EPOXY -DHAVE_PYTHON -D_LINUX_ \
    -I"${ORB_ROOT}" \
    -I"${ORB_ROOT}/include" \
    -I"${ORB_ROOT}/include/CameraModels" \
    -I"${ORB_ROOT}/Thirdparty/Sophus" \
    -I/usr/include/eigen3 \
    -I/usr/include/opencv4 \
    "${TEST_SRC}" \
    -Wl,-rpath,"${ORB_ROOT}/lib:/usr/local/lib:${ORB_ROOT}/Thirdparty/DBoW2/lib:${ORB_ROOT}/Thirdparty/g2o/lib" \
    "${ORB_ROOT}/lib/libORB_SLAM3.so" \
    "${ORB_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so" \
    "${ORB_ROOT}/Thirdparty/g2o/lib/libg2o.so" \
    -lopencv_core -lopencv_features2d -lboost_serialization -lcrypto -lpthread -lepoxy \
    -L/usr/local/lib -lpango_core -lpango_display -lpango_opengl -lpango_windowing \
    -o "${BUILD_BIN}"

echo "=== Executing Genuine Native Tests ==="
"${BUILD_BIN}"
echo "=== Native Test Execution Complete ==="
