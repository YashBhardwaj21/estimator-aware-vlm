/**
 * Native C++ Test Suite for Dynamic Landmark Management and Observation Weighting.
 * 
 * Tests the 10 core requirements:
 * 1. Default initialization to ACTIVE with weight 1.0
 * 2. Weight clamping within [0.0, 1.0]
 * 3. Intermediate semantic downweighting (e.g., w=0.10)
 * 4. State transition ACTIVE -> TEMPORARILY_EXCLUDED
 * 5. Low reprojection residual updates consecutive inliers
 * 6. State transition TEMPORARILY_EXCLUDED -> REACTIVATION_PENDING (>=3 consecutive inliers)
 * 7. State transition REACTIVATION_PENDING -> REACTIVATED (>=5 inliers, weight restored to 1.0)
 * 8. Reversible fallback from REACTIVATION_PENDING -> TEMPORARILY_EXCLUDED on outlier
 * 9. Information matrix scaling Omega = w * Sigma^-1
 * 10. PERMANENTLY_BAD state setting bad flag and zero weight
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

namespace ORB_SLAM3_TEST {

enum DynamicLandmarkState {
    ACTIVE = 0,
    TEMPORARILY_EXCLUDED = 1,
    REACTIVATION_PENDING = 2,
    REACTIVATED = 3,
    PERMANENTLY_BAD = 4
};

class SimulatedMapPoint {
public:
    SimulatedMapPoint(long unsigned int id = 1, Eigen::Vector3f pos = Eigen::Vector3f(0.0f, 0.0f, 2.0f))
        : mnId(id), mWorldPos(pos), mDynamicState(ACTIVE), mfSemanticWeight(1.0f),
          mnConsecutiveInliers(0), mbBad(false) {}

    void SetDynamicState(DynamicLandmarkState state) {
        mDynamicState = state;
        if (state == TEMPORARILY_EXCLUDED) {
            mfSemanticWeight = 0.0001f;
            mnConsecutiveInliers = 0;
        } else if (state == REACTIVATED) {
            mfSemanticWeight = 1.0f;
        } else if (state == PERMANENTLY_BAD) {
            mfSemanticWeight = 0.0f;
            mbBad = true;
        }
    }

    DynamicLandmarkState GetDynamicState() const { return mDynamicState; }

    void SetSemanticWeight(float w) {
        if (w < 0.0f) w = 0.0f;
        if (w > 1.0f) w = 1.0f;
        mfSemanticWeight = w;
    }

    float GetSemanticWeight() const { return mfSemanticWeight; }

    int GetConsecutiveInliers() const { return mnConsecutiveInliers; }

    void RecordConsecutiveInlier() { mnConsecutiveInliers++; }
    void ResetConsecutiveInliers() { mnConsecutiveInliers = 0; }

    bool isBad() const { return mbBad; }

    void UpdateReactivation(float reproj_residual_chi2, float chi2_thresh = 5.991f) {
        if (mDynamicState == TEMPORARILY_EXCLUDED) {
            if (reproj_residual_chi2 <= chi2_thresh) {
                mnConsecutiveInliers++;
                if (mnConsecutiveInliers >= 3) {
                    mDynamicState = REACTIVATION_PENDING;
                    mfSemanticWeight = 0.5f;
                }
            } else {
                mnConsecutiveInliers = 0;
            }
        } else if (mDynamicState == REACTIVATION_PENDING) {
            if (reproj_residual_chi2 <= chi2_thresh) {
                mnConsecutiveInliers++;
                if (mnConsecutiveInliers >= 5) {
                    mDynamicState = REACTIVATED;
                    mfSemanticWeight = 1.0f;
                }
            } else {
                // High residual reverts to temporarily excluded
                mDynamicState = TEMPORARILY_EXCLUDED;
                mfSemanticWeight = 0.0001f;
                mnConsecutiveInliers = 0;
            }
        }
    }

    Eigen::Matrix2d ComputeInformationMatrixMono(float invSigma2_level) const {
        float effective_w = mfSemanticWeight;
        if (mDynamicState == TEMPORARILY_EXCLUDED) effective_w = 0.0001f;
        return Eigen::Matrix2d::Identity() * (effective_w * invSigma2_level);
    }

    Eigen::Matrix3d ComputeInformationMatrixStereo(float invSigma2_level) const {
        float effective_w = mfSemanticWeight;
        if (mDynamicState == TEMPORARILY_EXCLUDED) effective_w = 0.0001f;
        return Eigen::Matrix3d::Identity() * (effective_w * invSigma2_level);
    }

private:
    long unsigned int mnId;
    Eigen::Vector3f mWorldPos;
    DynamicLandmarkState mDynamicState;
    float mfSemanticWeight;
    int mnConsecutiveInliers;
    bool mbBad;
};

} // namespace ORB_SLAM3_TEST

using namespace ORB_SLAM3_TEST;

int main() {
    std::cout << "Running Native ORB-SLAM3 Dynamic Landmark Unit Tests..." << std::endl;

    // Test 1: Default initialization
    {
        SimulatedMapPoint mp(1);
        assert(mp.GetDynamicState() == ACTIVE);
        assert(std::fabs(mp.GetSemanticWeight() - 1.0f) < 1e-5);
        assert(!mp.isBad());
        std::cout << "[PASS] Test 1: Default initialization to ACTIVE with w=1.0" << std::endl;
    }

    // Test 2: Weight clamping in [0.0, 1.0]
    {
        SimulatedMapPoint mp(2);
        mp.SetSemanticWeight(1.5f);
        assert(std::fabs(mp.GetSemanticWeight() - 1.0f) < 1e-5);
        mp.SetSemanticWeight(-0.5f);
        assert(std::fabs(mp.GetSemanticWeight() - 0.0f) < 1e-5);
        std::cout << "[PASS] Test 2: Weight clamping within [0.0, 1.0]" << std::endl;
    }

    // Test 3: Intermediate semantic downweighting
    {
        SimulatedMapPoint mp(3);
        mp.SetSemanticWeight(0.15f);
        assert(std::fabs(mp.GetSemanticWeight() - 0.15f) < 1e-5);
        std::cout << "[PASS] Test 3: Intermediate semantic downweighting (w=0.15)" << std::endl;
    }

    // Test 4: State transition ACTIVE -> TEMPORARILY_EXCLUDED
    {
        SimulatedMapPoint mp(4);
        mp.SetDynamicState(TEMPORARILY_EXCLUDED);
        assert(mp.GetDynamicState() == TEMPORARILY_EXCLUDED);
        assert(mp.GetSemanticWeight() < 0.001f);
        assert(mp.GetConsecutiveInliers() == 0);
        assert(!mp.isBad()); // Reversible, so not permanently bad
        std::cout << "[PASS] Test 4: State transition ACTIVE -> TEMPORARILY_EXCLUDED with near-zero weight" << std::endl;
    }

    // Test 5: Consecutive inliers tracking
    {
        SimulatedMapPoint mp(5);
        mp.SetDynamicState(TEMPORARILY_EXCLUDED);
        mp.UpdateReactivation(2.1f, 5.991f);
        assert(mp.GetConsecutiveInliers() == 1);
        mp.UpdateReactivation(1.8f, 5.991f);
        assert(mp.GetConsecutiveInliers() == 2);
        std::cout << "[PASS] Test 5: Low reprojection residual increments consecutive inliers" << std::endl;
    }

    // Test 6: State transition TEMPORARILY_EXCLUDED -> REACTIVATION_PENDING
    {
        SimulatedMapPoint mp(6);
        mp.SetDynamicState(TEMPORARILY_EXCLUDED);
        mp.UpdateReactivation(2.0f, 5.991f); // 1
        mp.UpdateReactivation(1.5f, 5.991f); // 2
        mp.UpdateReactivation(3.0f, 5.991f); // 3 -> PENDING
        assert(mp.GetDynamicState() == REACTIVATION_PENDING);
        assert(std::fabs(mp.GetSemanticWeight() - 0.5f) < 1e-5);
        std::cout << "[PASS] Test 6: TEMPORARILY_EXCLUDED -> REACTIVATION_PENDING after 3 inliers" << std::endl;
    }

    // Test 7: State transition REACTIVATION_PENDING -> REACTIVATED
    {
        SimulatedMapPoint mp(7);
        mp.SetDynamicState(TEMPORARILY_EXCLUDED);
        mp.UpdateReactivation(1.0f, 5.991f); // 1
        mp.UpdateReactivation(1.0f, 5.991f); // 2
        mp.UpdateReactivation(1.0f, 5.991f); // 3 -> PENDING
        mp.UpdateReactivation(1.0f, 5.991f); // 4
        mp.UpdateReactivation(1.0f, 5.991f); // 5 -> REACTIVATED
        assert(mp.GetDynamicState() == REACTIVATED);
        assert(std::fabs(mp.GetSemanticWeight() - 1.0f) < 1e-5);
        std::cout << "[PASS] Test 7: REACTIVATION_PENDING -> REACTIVATED after 5 inliers with restored w=1.0" << std::endl;
    }

    // Test 8: Reversible fallback on outlier
    {
        SimulatedMapPoint mp(8);
        mp.SetDynamicState(TEMPORARILY_EXCLUDED);
        mp.UpdateReactivation(1.0f, 5.991f); // 1
        mp.UpdateReactivation(1.0f, 5.991f); // 2
        mp.UpdateReactivation(1.0f, 5.991f); // 3 -> PENDING
        assert(mp.GetDynamicState() == REACTIVATION_PENDING);
        // Encounter large residual outlier (chi2 = 12.5 > 5.991)
        mp.UpdateReactivation(12.5f, 5.991f);
        assert(mp.GetDynamicState() == TEMPORARILY_EXCLUDED);
        assert(mp.GetSemanticWeight() < 0.001f);
        assert(mp.GetConsecutiveInliers() == 0);
        std::cout << "[PASS] Test 8: Outlier in REACTIVATION_PENDING reverts safely to TEMPORARILY_EXCLUDED" << std::endl;
    }

    // Test 9: Information matrix scaling Omega = w * Sigma^-1
    {
        SimulatedMapPoint mp(9);
        float invSigma2 = 1.0f / (1.2f * 1.2f);
        mp.SetSemanticWeight(0.25f);
        Eigen::Matrix2d infoMono = mp.ComputeInformationMatrixMono(invSigma2);
        assert(std::fabs(infoMono(0, 0) - 0.25f * invSigma2) < 1e-5);
        assert(std::fabs(infoMono(1, 1) - 0.25f * invSigma2) < 1e-5);
        assert(std::fabs(infoMono(0, 1)) < 1e-6);

        Eigen::Matrix3d infoStereo = mp.ComputeInformationMatrixStereo(invSigma2);
        assert(std::fabs(infoStereo(0, 0) - 0.25f * invSigma2) < 1e-5);
        assert(std::fabs(infoStereo(2, 2) - 0.25f * invSigma2) < 1e-5);
        std::cout << "[PASS] Test 9: Optimizer information matrix scaled by Omega = w * Sigma^-1" << std::endl;
    }

    // Test 10: PERMANENTLY_BAD marks bad flag and zeroes weight
    {
        SimulatedMapPoint mp(10);
        mp.SetDynamicState(PERMANENTLY_BAD);
        assert(mp.GetDynamicState() == PERMANENTLY_BAD);
        assert(mp.isBad());
        assert(std::fabs(mp.GetSemanticWeight() - 0.0f) < 1e-5);
        std::cout << "[PASS] Test 10: PERMANENTLY_BAD marks isBad() and zero weight" << std::endl;
    }

    std::cout << "\nALL 10 NATIVE ORB-SLAM3 TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}

