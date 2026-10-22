#!/usr/bin/env python3
"""RealSense D455 Lightweight IMU Streamer & Heading Estimator (Zero Heavy Files).

Reads 200 Hz Gyroscope and 63/250 Hz Accelerometer from D455 connected to Raspberry Pi 5.
- NO RGB frames enabled.
- NO Depth maps enabled.
- 0 MB disk writes, 0 network bandwidth.
- Computes real-time integrated Yaw heading angle (θ) for closed-loop map navigation.
"""

from __future__ import annotations

import math
import sys
import time

try:
    import pyrealsense2 as rs
except ImportError:
    print("\n[ERROR] pyrealsense2 is not installed!")
    print("Run on your Raspberry Pi: pip install pyrealsense2")
    sys.exit(1)


def main():
    print("=" * 68)
    print("REALSENSE D455 LIGHTWEIGHT IMU DIAGNOSTIC & HEADING TRACKER")
    print("Mode: Pure Motion (Gyro + Accel) | RGB: DISABLED | Depth: DISABLED")
    print("=" * 68)

    # 1. Device Discovery
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[FAIL] No RealSense device detected on USB!")
        print("Please check that the D455 is plugged into the Pi's blue USB-3.0 port.")
        sys.exit(1)

    dev = devices[0]
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown"

    print(f"[FOUND] Device:        {name}")
    print(f"        Serial Number: {serial}")
    print(f"        USB Mode:      {usb}")
    print("-" * 68)

    # 2. Configure IMU-Only Pipeline
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)

    # Explicitly enable ONLY the motion sensors (No RGB / No Depth)
    cfg.enable_stream(rs.stream.gyro)
    cfg.enable_stream(rs.stream.accel)

    try:
        profile = pipe.start(cfg)
        print("[SUCCESS] IMU Pipeline started successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to start motion stream: {e}")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="RealSense D455 Lightweight IMU Diagnostics")
    parser.add_argument("--inverted", action="store_true", default=True, help="Camera is mounted upside-down/inverted (default: True)")
    args = parser.parse_args()


    inv_sign = -1.0 if args.inverted else 1.0
    print(f"Mounting Orientation: {'INVERTED (UPSIDE-DOWN)' if args.inverted else 'NORMAL (UPRIGHT)'}")
    print("=" * 68)
    print(">>> ROTATE THE CAR TO THE LEFT — YAW SHOULD BE POSITIVE (+) <<<")
    print("Press Ctrl+C to stop.\n")

    # 3. Real-Time Heading (Yaw) Integration
    yaw_deg = 0.0
    last_time = time.time()
    last_gyro = (0.0, 0.0, 0.0)
    last_accel = (0.0, 0.0, 0.0)
    sample_count = 0
    t_start = time.time()

    try:
        while True:
            frames = pipe.wait_for_frames()

            gyro_frame = frames.first_or_default(rs.stream.gyro)
            accel_frame = frames.first_or_default(rs.stream.accel)

            now = time.time()
            dt = now - last_time
            last_time = now

            if gyro_frame:
                g_data = gyro_frame.as_motion_frame().get_motion_data()
                # Apply inversion sign so Turning Left is ALWAYS positive (+), Right is negative (-)
                omega_z = inv_sign * g_data.y
                last_gyro = (g_data.x, g_data.y, g_data.z)

                # Deadband filter to prevent stationary drift
                if abs(omega_z) > 0.015:
                    yaw_deg += math.degrees(omega_z) * dt


                # Normalize yaw between -180 and +180
                while yaw_deg > 180.0:
                    yaw_deg -= 360.0
                while yaw_deg < -180.0:
                    yaw_deg += 360.0

            if accel_frame:
                a_data = accel_frame.as_motion_frame().get_motion_data()
                last_accel = (a_data.x, a_data.y, a_data.z)

            sample_count += 1

            # Print telemetry every ~100 ms (10 Hz update rate)
            if sample_count % 20 == 0:
                elapsed = now - t_start
                rate_hz = sample_count / elapsed
                gx, gy, gz = last_gyro
                ax, ay, az = last_accel

                # Direction visual indicator
                if abs(yaw_deg) < 3.0:
                    dir_tag = "CENTER (FORWARD)"
                elif yaw_deg > 0:
                    dir_tag = f"TURNED LEFT  (+{yaw_deg:.1f}°)"
                else:
                    dir_tag = f"TURNED RIGHT ({yaw_deg:.1f}°)"

                sys.stdout.write(
                    f"\r[HEADING] Yaw: {yaw_deg:+6.1f}° | {dir_tag:<24} | "
                    f"Gyro Z: {math.degrees(gy):+6.1f}°/s | "
                    f"Accel: ({ax:5.2f}, {ay:5.2f}, {az:5.2f}) m/s² | "
                    f"FPS: {rate_hz:4.0f} Hz"
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n" + "=" * 68)
        print(f"[STOPPED] IMU test complete.")
        print(f"Final Integrated Heading: {yaw_deg:+.1f}°")
        print("Total frames processed: ", sample_count)
        print("Heavy files generated:    0 MB (Confirmed pure telemetry)")
        print("=" * 68)
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
