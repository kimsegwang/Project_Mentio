"""
tests/test_gesture_detector.py
손하트 판정 로직(server/services/gesture_detector.HeartDetector._check_big_heart) 검증 pytest 단위 테스트.
MediaPipe 모델은 로딩하지 않고, 가짜 손 랜드마크 좌표로 기하학 판정만 검증한다.
"""
from types import SimpleNamespace

import pytest

from server.services.gesture_detector import HeartDetector


def _hand(points):
    """{랜드마크 인덱스: (x, y)} -> 21개 랜드마크 리스트 (미지정 인덱스는 원점)"""
    return [SimpleNamespace(x=points.get(i, (0.0, 0.0))[0], y=points.get(i, (0.0, 0.0))[1]) for i in range(21)]


def _heart_pose(left_overrides=None, right_overrides=None, right_shift=0.0):
    """정상 손하트 자세: 양 엄지가 수평으로 곧게 맞닿고, 검지 끝이 둥글게 말려 맞닿은 상태."""
    left = {0: (0.30, 0.80), 9: (0.30, 0.50), 2: (0.36, 0.60), 3: (0.42, 0.60), 4: (0.48, 0.60),
            7: (0.46, 0.44), 8: (0.49, 0.45)}
    right = {0: (0.70, 0.80), 9: (0.70, 0.50), 2: (0.64, 0.60), 3: (0.58, 0.60), 4: (0.52, 0.60),
             7: (0.54, 0.44), 8: (0.51, 0.45)}
    left.update(left_overrides or {})
    right.update(right_overrides or {})
    right = {i: (x + right_shift, y) for i, (x, y) in right.items()}
    return _hand(left), _hand(right)


@pytest.fixture()
def detector():
    # 판정 로직은 MediaPipe 인스턴스를 쓰지 않으므로 모델 로딩 없이 생성한다
    return HeartDetector.__new__(HeartDetector)


def test_detects_big_heart(detector):
    left, right = _heart_pose()
    assert detector._check_big_heart(left, right) == "BIG_HEART"


def test_rejects_diamond_when_index_tips_point_up(detector):
    left, right = _heart_pose(left_overrides={8: (0.49, 0.40)}, right_overrides={8: (0.51, 0.40)})
    assert detector._check_big_heart(left, right) == "DIAMOND_REJECTED"


def test_rejects_bent_thumb(detector):
    left, right = _heart_pose(left_overrides={2: (0.42, 0.66)})  # 엄지 2-3-4 각도 90도
    assert detector._check_big_heart(left, right) == "THUMB_BENT_REJECTED"


def test_returns_none_when_hands_not_touching(detector):
    left, right = _heart_pose(right_shift=0.3)
    assert detector._check_big_heart(left, right) is None
