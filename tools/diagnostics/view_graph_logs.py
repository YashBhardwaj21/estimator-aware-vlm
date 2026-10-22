#!/usr/bin/env python3
"""
Live terminal log viewer for the Dynamic 3D Scene Graph.
Subscribes to /scene_graph/state and renders a formatted real-time dashboard.
"""

import json
import sys
import os

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:
    print("[Error] rclpy is not installed or ROS 2 is not sourced.")
    print("Please run: source /opt/ros/humble/setup.bash")
    sys.exit(1)


class GraphLogViewer(Node):
    def __init__(self):
        super().__init__("graph_log_viewer")
        self.sub = self.create_subscription(
            String,
            "/scene_graph/state",
            self.on_state_received,
            10,
        )
        self.frame_count = 0
        print("[Log Viewer] Listening to /scene_graph/state... Waiting for first frame...")

    def on_state_received(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            return

        self.frame_count += 1
        frame = data.get("frame", self.frame_count)
        timestamp = data.get("timestamp", 0.0)
        telem = data.get("telemetry", {})
        objects = data.get("objects", [])
        relations = data.get("relations", [])
        summary = data.get("summary", {})

        fps = telem.get("processing_fps", 0.0)
        latency = telem.get("total_latency_ms", 0.0)
        qsize = telem.get("queue_size", 0)
        dropped = telem.get("dropped_frames", 0)

        # Clear screen and move cursor home for smooth terminal dashboard
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write("[DYNAMIC 3D SCENE GRAPH TELEMETRY]\n")
        sys.stdout.write(f" Frame: #{frame:05d} | Timestamp: {timestamp:12.4f}s | FPS: {fps:5.1f} | Latency: {latency:5.1f}ms | Queue: {qsize} | Dropped: {dropped}\n\n")

        sys.stdout.write(f" ACTIVE 3D OBJECTS ({len(objects)}):\n")
        if not objects:
            sys.stdout.write("   (No confirmed 3D objects currently tracked)\n")
        else:
            sys.stdout.write(f"   {'TRACK ID':<12} {'CLASS':<12} {'STATE':<10} {'CENTROID [X, Y, Z] (m)':<26} {'CONF':<6} {'OBS':<5}\n")
            for obj in objects:
                tid = obj.get("id", "N/A")
                cls = obj.get("class", "unknown")
                st = obj.get("state", "ACTIVE")
                conf = obj.get("confidence", 0.0)
                obs = obj.get("observations", 0)
                centroid = obj.get("centroid")
                pos_str = f"[{centroid[0]:6.2f}, {centroid[1]:6.2f}, {centroid[2]:6.2f}]" if centroid else "[  N/A,   N/A,   N/A]"
                sys.stdout.write(f"   {tid:<12} {cls:<12} {st:<10} {pos_str:<26} {conf:0.2f}   {obs:<5}\n")

        sys.stdout.write(f"\n CONFIRMED SPATIAL & TEMPORAL RELATIONS ({len(relations)}):\n")
        if not relations:
            sys.stdout.write("   (No confirmed relations in active belief model)\n")
        else:
            for rel in relations:
                sub = f"{rel.get('subject_class', 'obj')}:{rel.get('subject', '?')}"
                pred = rel.get("predicate", "REL")
                obj_str = f"{rel.get('object_class', 'obj')}:{rel.get('object', '?')}"
                conf = rel.get("confidence", 0.0)
                state = rel.get("state", "ACTIVE")
                sys.stdout.write(f"   * {sub:<22} -> [{pred:^13}] -> {obj_str:<22} (conf: {conf:0.2f}, {state})\n")

        sys.stdout.write("\n")
        sys.stdout.flush()


def main():
    rclpy.init()
    viewer = GraphLogViewer()
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
