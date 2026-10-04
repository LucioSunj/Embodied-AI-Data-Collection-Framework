
#!/usr/bin/env python3
"""Enumerate RealSense cameras and show each camera with its serial number.

Usage:
    python realsense_serial_preview.py

Press ``q`` or ``Esc`` to stop the preview.
"""

import math
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


WIDTH = 640
HEIGHT = 480
FPS = 30
WINDOW_NAME = "RealSense cameras - serial numbers"


def enumerate_devices():
    """Return the connected RealSense devices and print their identities."""
    context = rs.context()
    devices = context.query_devices()

    if devices.size() == 0:
        print("未检测到 RealSense 相机。")
        return []

    result = []
    print(f"检测到 {devices.size()} 台 RealSense 相机：")
    for index in range(devices.size()):
        device = devices[index]
        name = device.get_info(rs.camera_info.name)
        try:
            serial = device.get_info(rs.camera_info.serial_number)
        except RuntimeError:
            serial = "unknown"
        print(f"  [{index}] {name} | 序列号: {serial}")
        result.append((device, name, serial))
    return result


def make_pipeline(device):
    """Start a color pipeline for one device."""
    serial = device.get_info(rs.camera_info.serial_number)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    return pipeline


def make_placeholder(text, width=WIDTH, height=HEIGHT):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        text,
        (25, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def add_label(frame, index, name, serial):
    """Draw camera index, model, and serial number on a frame."""
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (WIDTH, 78), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"Camera {index}: {name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Serial: {serial}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def make_grid(frames):
    """Arrange frames in a compact grid for any number of cameras."""
    count = len(frames)
    columns = min(2, count)
    rows = math.ceil(count / columns)
    blank = np.zeros_like(frames[0])
    rows_of_frames = []

    for row in range(rows):
        row_frames = frames[row * columns : (row + 1) * columns]
        row_frames += [blank] * (columns - len(row_frames))
        rows_of_frames.append(np.hstack(row_frames))

    return np.vstack(rows_of_frames)


def main():
    try:
        devices = enumerate_devices()
    except Exception as exc:
        print(f"无法通过 RealSense SDK 枚举相机: {exc}", file=sys.stderr)
        print("请确认当前用户有 USB/udev 设备访问权限。", file=sys.stderr)
        return 1

    if not devices:
        return 1

    pipelines = []
    for index, (device, name, serial) in enumerate(devices):
        try:
            pipelines.append((index, name, serial, make_pipeline(device)))
        except Exception as exc:
            print(f"相机 {serial} 启动失败: {exc}", file=sys.stderr)
            pipelines.append((index, name, serial, None))

    print(f"实时预览已启动，按 q 或 Esc 退出。窗口: {WINDOW_NAME}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = []
            for index, name, serial, pipeline in pipelines:
                if pipeline is None:
                    frame = make_placeholder(f"Camera {index}: unavailable")
                else:
                    try:
                        color_frame = pipeline.wait_for_frames(1000).get_color_frame()
                        if color_frame:
                            frame = np.asanyarray(color_frame.get_data())
                        else:
                            frame = make_placeholder(f"Camera {index}: no frame")
                    except RuntimeError as exc:
                        frame = make_placeholder(f"Camera {index}: {exc}")
                frames.append(add_label(frame, index, name, serial))

            cv2.imshow(WINDOW_NAME, make_grid(frames))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for _, _, _, pipeline in pipelines:
            if pipeline is not None:
                pipeline.stop()
        cv2.destroyAllWindows()
        time.sleep(0.1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
