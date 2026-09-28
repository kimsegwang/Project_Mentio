"""
tests/conftest.py
pytest 공용 설정.

[하드웨어 테스트 격리] 실제 마이크/카메라 입력을 무기한 대기하는 대화형 테스트는 `@pytest.mark.hardware`로
표시하고, 기본 실행(`pytest tests/`)에서는 수집 단계에서 스킵한다. 무인 실행 중 웹캠을 점유한 채 마이크
입력을 기다리며 멈추면 `python -m server.main`이 카메라 프레임을 얻지 못하므로, 명시적으로 옵트인할
때만 실행한다:

    pytest tests/ --run-hardware          # 또는 환경변수 MENTIO_RUN_HARDWARE_TESTS=1
"""
import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="실제 마이크/카메라가 필요한 대화형 하드웨어 테스트(@pytest.mark.hardware)를 실행한다.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "hardware: 실제 마이크/카메라 입력이 필요한 대화형 테스트 (기본 스킵, --run-hardware로 실행)"
    )


def _hardware_tests_enabled(config) -> bool:
    return config.getoption("--run-hardware") or os.getenv("MENTIO_RUN_HARDWARE_TESTS") == "1"


def pytest_collection_modifyitems(config, items):
    if _hardware_tests_enabled(config):
        return
    skip_hardware = pytest.mark.skip(
        reason="실제 마이크/카메라가 필요한 대화형 테스트 (--run-hardware 또는 MENTIO_RUN_HARDWARE_TESTS=1로 실행)"
    )
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)
