"""
Project Mentio - Vision Gesture Detector Module
- 단일 타겟: 양손 정밀 손하트(BIG_HEART) 전용 감지
- 기하학적 필터:
  1. 검지 곡률(⌒ ⌒) 검증 (마름모 방지)
  2. 엄지 진직도(Straightness) 각도 검증 (구부러진 엄지 오탐 방지)
  3. 좌우 손 크기 비례 동적 정규화
"""

import cv2
import mediapipe as mp
import math


class HeartDetector:
    def __init__(self, min_detection_confidence=0.35, min_tracking_confidence=0.4):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    @staticmethod
    def _calculate_distance(p1, p2):
        """2차원 유클리드 거리 계산"""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    @staticmethod
    def _calculate_angle(a, b, c):
        """세 점(a-b-c) 사이의 사잇각(도, Degree) 계산 (b가 중심점)"""
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = abs(radians * 180.0 / math.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return angle

    def _check_big_heart(self, left, right):
        """양손 정밀 하트 판정"""
        # 1. 손 크기 동적 스케일 정규화 (손목 0 -> 중지시작 9)
        hand_scale = (self._calculate_distance(left[0], left[9]) + 
                      self._calculate_distance(right[0], right[9])) / 2.0
        if hand_scale < 0.01:
            hand_scale = 0.1

        # 엄지 끝(4) 및 검지 끝(8) 정규화 거리
        thumb_dist = self._calculate_distance(left[4], right[4]) / hand_scale
        index_dist = self._calculate_distance(left[8], right[8]) / hand_scale

        # 조건 1: 엄지 끝, 검지 끝 접촉
        is_contact = (thumb_dist < 0.42) and (index_dist < 0.42)

        # 조건 2: 검지가 엄지보다 위에 위치
        is_index_above = (left[8].y < left[4].y) and (right[8].y < right[4].y)

        # 조건 3 (마름모 방지): 검지 끝(8)이 관절(7)보다 수평이거나 아래로 곡선 형성
        is_index_curved = (left[8].y >= left[7].y - 0.02) and (right[8].y >= right[7].y - 0.02)

        # 조건 4 (핵심 추가): 엄지 진직도 검증 (말림 방지)
        # 엄지 관절 2-3-4가 이루는 각도가 펴진 상태(최소 150도 이상)여야 함
        left_thumb_angle = self._calculate_angle(left[2], left[3], left[4])
        right_thumb_angle = self._calculate_angle(right[2], right[3], right[4])
        is_thumbs_straight = (left_thumb_angle > 150.0) and (right_thumb_angle > 150.0)

        # 조건 5: 맞닿은 양 엄지가 수평에 가깝게 유지되는가
        is_thumbs_level = (abs(left[4].y - left[3].y) < 0.09) and (abs(right[4].y - right[3].y) < 0.09)

        # 판정 로직
        if is_contact and is_index_above:
            if not is_index_curved:
                return "DIAMOND_REJECTED"
            if not is_thumbs_straight:
                return "THUMB_BENT_REJECTED"
            if is_thumbs_level:
                return "BIG_HEART"

        return None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        detected_gesture = None
        status_msg = "Ready (Show Two Hands)"
        status_color = (180, 180, 180)

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            if num_hands == 2:
                h1 = results.multi_hand_landmarks[0].landmark
                h2 = results.multi_hand_landmarks[1].landmark

                # x좌표 기준 좌/우 손 정렬
                left = h1 if h1[0].x < h2[0].x else h2
                right = h2 if h1[0].x < h2[0].x else h1

                res = self._check_big_heart(left, right)
                if res == "BIG_HEART":
                    detected_gesture = "HEART_EYES"
                    status_msg = "HEART DETECTED!"
                    status_color = (147, 20, 255)  # 보라/핑크
                elif res == "DIAMOND_REJECTED":
                    status_msg = "Rejected: Diamond Shape"
                    status_color = (0, 165, 255)  # 주황
                elif res == "THUMB_BENT_REJECTED":
                    status_msg = "Rejected: Straighten Thumbs"
                    status_color = (0, 215, 255)  # 노랑

            # 손가락 스켈레톤 시각화
            for lm in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

        return detected_gesture, status_msg, status_color, frame


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