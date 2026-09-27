"""
tests/test_gesture.py
손하트 감지기(server/services/gesture_detector.py) 수동 실기 확인용 카메라 데모.
자동 테스트 함수는 없으며, `python -m tests.test_gesture`로 실행해 웹캠 앞에서 직접 확인한다.
"""

import cv2

from server.services.gesture_detector import HeartDetector


def main():
    cap = cv2.VideoCapture(0)
    detector = HeartDetector()
    win_name = "Project Mentio - Heart Detector"
    cv2.namedWindow(win_name)

    print("=== 하트 감지 엔진 가동 (q: 종료) ===")

    while cap.isOpened():
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        gesture, text, color, processed_frame = detector.process_frame(frame)

        cv2.putText(processed_frame, text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.imshow(win_name, processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
