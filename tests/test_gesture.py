"""
Project Mentio - Vision Gesture Detector Module
- 목적: 웹캠 영상에서 실시간 손하트(양손 큰 하트 / 한 손 미니 하트)를 0.05초 내 초저지연 감지
- 기하학적 필터링: 단순 거리 판정 외에 마름모 형태 오인식 방어 알고리즘 탑재
"""

import cv2
import mediapipe as mp
import math


class HeartDetector:
    def __init__(self, min_detection_confidence=0.35, min_tracking_confidence=0.4):
        """
        MediaPipe Hands 파이프라인 초기화
        - detection confidence를 0.35로 설정하여 하트를 쥔 채로 카메라에 진입해도 즉각 검출
        """
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
        """2차원 좌표 간의 유클리드 거리 계산"""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _check_big_heart(self, left, right):
        """
        양손 큰 하트 판정 (마름모 오인식 방지 포함)
        - left, right: x좌표 기준으로 정렬된 좌/우 손 랜드마크
        """
        # 1. 사용자 손 크기(스케일) 정규화: 카메라 거리에 따른 오차 상쇄
        hand_scale = (self._calculate_distance(left[0], left[9]) + 
                      self._calculate_distance(right[0], right[9])) / 2.0
        if hand_scale < 0.01:
            hand_scale = 0.1

        # 정규화된 손가락 끝 접촉 거리
        thumb_dist = self._calculate_distance(left[4], right[4]) / hand_scale
        index_dist = self._calculate_distance(left[8], right[8]) / hand_scale

        # 조건 A: 엄지 끝(4번), 검지 끝(8번) 맞닿음
        is_contact = (thumb_dist < 0.42) and (index_dist < 0.42)

        # 조건 B: 검지 끝이 엄지 끝보다 화면상 위(y값이 더 작음)에 위치
        is_index_above = (left[8].y < left[4].y) and (right[8].y < right[4].y)

        # 조건 C (마름모 방어): 검지 끝(8)이 마디 관절(7)보다 살짝 내려오며 곡선(⌒)을 그리는가
        is_curved = (left[8].y >= left[7].y - 0.02) and (right[8].y >= right[7].y - 0.02)

        # 조건 D: 하단 엄지가 뾰족하지 않고 수평에 가깝게 맞닿아 있는가
        is_thumbs_flat = (abs(left[4].y - left[3].y) < 0.08) and (abs(right[4].y - right[3].y) < 0.08)

        if is_contact and is_index_above and is_curved and is_thumbs_flat:
            return "BIG_HEART"
        elif is_contact and is_index_above:
            return "DIAMOND_REJECTED"
        return None

    def _check_mini_heart(self, hand):
        """한 손 K-손하트(엄지-검지 교차) 판정"""
        palm_size = self._calculate_distance(hand[0], hand[9])
        if palm_size < 0.01:
            palm_size = 0.1

        # 엄지와 검지 끝 사이 거리 정규화
        thumb_index_dist = self._calculate_distance(hand[4], hand[8]) / palm_size

        # 나머지 세 손가락(중지 12, 약지 16, 소지 20)이 접혀 있는지 검사
        is_folded = (hand[12].y > hand[9].y) and (hand[16].y > hand[9].y) and (hand[20].y > hand[9].y)

        # 엄지와 검지가 적당히 교차하고 손목 위에 위치하는지 확인
        if 0.15 < thumb_index_dist < 0.55 and is_folded and (hand[4].y < hand[0].y):
            return "MINI_HEART"
        return None

    def process_frame(self, frame):
        """프레임 추론 후 감지 상태 및 렌더링된 프레임 반환"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        detected_gesture = None
        status_msg = "Ready"
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
                    status_msg = "BIG_HEART Detected!"
                    status_color = (147, 20, 255)
                elif res == "DIAMOND_REJECTED":
                    status_msg = "Diamond Shape Rejected"
                    status_color = (0, 165, 255)

            elif num_hands == 1:
                hand = results.multi_hand_landmarks[0].landmark
                if self._check_mini_heart(hand) == "MINI_HEART":
                    detected_gesture = "HEART_EYES"
                    status_msg = "MINI_HEART Detected!"
                    status_color = (255, 105, 180)

            # 손가락 랜드마크 스켈레톤 시각화
            for lm in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

        return detected_gesture, status_msg, status_color, frame


def main():
    cap = cv2.VideoCapture(0)
    detector = HeartDetector()
    win_name = "Project Mentio - Vision Engine"
    cv2.namedWindow(win_name)

    print("=== 비전 엔진 구동 중 (종료: q 또는 창 닫기 X 버튼) ===")

    while cap.isOpened():
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        gesture, text, color, processed_frame = detector.process_frame(frame)

        # 향후 ESP32 전송용 제어 이벤트 트리거 시점
        if gesture == "HEART_EYES":
            # 실제 배포 환경에서는 이 시점에 WebSocket으로 ESP32에 패킷 전송
            # payload: {"emotion": "HEART_EYES", "led_rgb": [255, 105, 180]}
            pass

        cv2.putText(processed_frame, text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow(win_name, processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()