import cv2
import pytest


def test_rtsp():
    cap = cv2.VideoCapture("rtsp://localhost:8554/camera1", cv2.CAP_FFMPEG)
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            pytest.skip("RTSP stream rtsp://localhost:8554/camera1 is not available")
        cv2.imwrite(f"tests/frame_{i}.jpg", frame)
    cap.release()
