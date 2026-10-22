#!/usr/bin/env python3
"""Hardware validation script for Intel RealSense D455."""

import sys
import time

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 is not installed in this environment.")
    sys.exit(1)


def main():
    print("[1] Initializing RealSense Context...")
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("ERROR: No RealSense devices detected.")
        sys.exit(1)

    dev = devices[0]
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    firmware = dev.get_info(rs.camera_info.firmware_version)
    usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown"

    print(f"Device Name:     {name}")
    print(f"Serial Number:   {serial}")
    print(f"Firmware:        {firmware}")
    print(f"USB Speed:       {usb}")

    if not str(usb).startswith("3"):
        print(f"WARNING: Camera is negotiated on USB '{usb}', NOT USB 3.x!")
    else:
        print("SUCCESS: Negotiated on USB 3.x!")

    # Check depth sensor scale
    depth_sensor = dev.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale:     {depth_scale:.6f} m/unit ({(1.0/depth_scale):.1f} units/m)")

    # Configure pipeline
    print("\n[2] Testing RGB + Depth Pipeline (640x480 @ 30fps)...")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    # Intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()
    print(f"Color Intrinsics: {color_intr.width}x{color_intr.height}, fx={color_intr.fx:.2f}, fy={color_intr.fy:.2f}, ppx={color_intr.ppx:.2f}, ppy={color_intr.ppy:.2f}, model={color_intr.model}")

    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intr = depth_stream.get_intrinsics()
    print(f"Depth Intrinsics: {depth_intr.width}x{depth_intr.height}, fx={depth_intr.fx:.2f}, fy={depth_intr.fy:.2f}, ppx={depth_intr.ppx:.2f}, ppy={depth_intr.ppy:.2f}, model={depth_intr.model}")

    print("\n[3] Acquiring 30 test frames...")
    start_t = time.time()
    for i in range(30):
        frames = pipe.wait_for_frames(timeout_ms=5000)
        aligned = align.process(frames)
        c_frame = aligned.get_color_frame()
        d_frame = aligned.get_depth_frame()
        if not c_frame or not d_frame:
            print(f"Frame {i}: Missing color or depth frame!")
            sys.exit(1)
        c_ts = c_frame.get_timestamp()
        d_ts = d_frame.get_timestamp()
        dt_ms = abs(c_ts - d_ts)
        if i % 10 == 0:
            print(f"Frame {i:02d}: color_ts={c_ts:.1f}ms, depth_ts={d_ts:.1f}ms, dt={dt_ms:.2f}ms")

    elapsed = time.time() - start_t
    pipe.stop()
    fps = 30.0 / elapsed
    print(f"\n[4] Stream test completed successfully: 30 frames in {elapsed:.2f}s (~{fps:.1f} FPS)")


if __name__ == "__main__":
    main()
