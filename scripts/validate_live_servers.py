"""Validate live model server for Qwen2.5-VL-3B."""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from estimator_aware_vlm.multimodal_inference_engine import MultimodalInferenceEngine


def main():
    print("=" * 80)
    print("LIVE MODEL SERVER VALIDATION & MULTIMODAL PREFLIGHT CHECKS")
    print("=" * 80)

    for mkey in ["qwen2_5_vl_3b"]:
        print(f"\n--- Validating {mkey} ---")
        engine = MultimodalInferenceEngine(mkey)
        health = engine.check_health()
        online = (health.get("status") == "ONLINE")
        print(f"  Endpoint: {engine.server_url}")
        print(f"  Health Status: {'ONLINE' if online else 'OFFLINE'}")
        print(f"  Model Verified: {health.get('model_verified')}")
        if health.get("digest"):
            print(f"  Digest: {health.get('digest')}")

        if not online:
            print(f"  ERROR: {mkey} server is OFFLINE.")
            continue

        print(f"  Running preflight multimodal smoke test...")
        smoke = engine.run_preflight_smoke_test()
        print(f"  Smoke Test Result: {smoke.get('status')}")
        print(f"  Latency: {smoke.get('latency_ms')}ms")
        print(f"  Parse Valid: {smoke.get('parse_valid')}")
        print(f"  Schema Valid: {smoke.get('schema_valid')}")
        print(f"  Raw Preview: {smoke.get('raw_response', '')[:120]}...")

        if not smoke.get("parse_valid") or not smoke.get("schema_valid"):
            print(f"  ERROR: Smoke test failed canonical schema validation.")
        else:
            print(f"  SUCCESS: {mkey} passed live preflight checks.")


if __name__ == "__main__":
    main()
