"""Test live VLM inference with Qwen2.5-VL-3B-Instruct on port 8089 using a real crop."""

import base64
import json
import time
import urllib.request
from pathlib import Path
import numpy as np

from estimator_aware_vlm.common.paths import get_repo_root


def main() -> None:
    repo_root = get_repo_root()
    crop_path = repo_root / "outputs" / "perception_crops" / "chair_01" / "view_01.jpg"
    if not crop_path.exists():
        raise FileNotFoundError(f"Test crop not found at: {crop_path}")

    import cv2
    img = cv2.imread(str(crop_path))
    h, w = img.shape[:2]
    # In the composite, the tight crop is on the left before the white separator bar
    # Find separator or take left panel:
    # Notice composite was [resized_crop, separator (4px), resized_ctx]
    # Let's crop just the target object panel:
    white_col = np.where(np.all(img == 255, axis=(0, 2)))[0]
    sep_x = white_col[0] if len(white_col) > 0 else int(w * 0.3)
    target_crop = img[:, :sep_x]

    # Resize to max dim 256 for fast CPU inference
    tc_h, tc_w = target_crop.shape[:2]
    scale = 256 / max(tc_h, tc_w)
    resized = cv2.resize(target_crop, (int(tc_w * scale), int(tc_h * scale)))
    _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Identify this physical entity/object in the image. "
                            "Determine its semantic class and contextual movability (IMMOVABLE, MOVABLE, or MOBILE). "
                            "Return valid JSON strictly matching this schema: "
                            '{"semantic_class": "<name>", "contextual_movability": "<IMMOVABLE|MOVABLE|MOBILE>", "reasoning": "<brief>"}'
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 150,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8089/v1/chat/completions",
        data=req_data,
        headers={"Content-Type": "application/json"},
    )

    print("Sending live VLM inference request to http://127.0.0.1:8089/v1/chat/completions...")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=75) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    lat_ms = (time.time() - t0) * 1000

    content = result["choices"][0]["message"]["content"]
    print(f"Response received in {lat_ms:.1f} ms:")
    print(content)


if __name__ == "__main__":
    main()
