import cv2
from PIL import Image
from config import settings
from tests.test_gesture import HeartDetector


class VisionService:
    def __init__(self):
        self.detector = HeartDetector()

    def process_gesture(self, frame):
        """실시간 프레임에서 손하트 제스처 감지 및 시각화"""
        if frame is None:
            return None, False

        # 거울 모드 좌우 반전 (좌우 손 landmark 정렬을 위해 필수)
        frame = cv2.flip(frame, 1)

        # HeartDetector의 실제 반환 시그니처: (gesture, msg, color, frame)
        detected_gesture, status_msg, status_color, processed_frame = self.detector.process_frame(frame)

        # 제스처 디텍터의 상태 안내 텍스트 렌더링
        cv2.putText(
            processed_frame,
            status_msg,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        is_heart = (detected_gesture == "HEART_EYES")
        return processed_frame, is_heart

    @staticmethod
    def prepare_snapshot_for_vlm(frame) -> Image.Image:
        """원본 프레임을 480p 비율로 리사이징하고 RGB 변환하여 PIL Image로 반환"""
        h, w, _ = frame.shape
        target_w = settings.TARGET_IMAGE_WIDTH
        target_h = int(h * (target_w / w))

        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)