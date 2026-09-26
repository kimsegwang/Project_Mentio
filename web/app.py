"""
web/app.py
멘티오 가족 프로필 & RAG 기억 관리 대시보드 (Streamlit 단일 진입점).

실행: streamlit run web/app.py   (프로젝트 루트에서)

- DB 접근은 전부 server/repositories 계층을 통해서만 수행한다 (이 파일에 SQL 없음).
- 대시보드는 로봇 엔진(server.main)과 별도 프로세스이므로 자체 커넥션 풀을 가진다.
  두 프로세스는 PostgreSQL 커넥션을 공유하지 않아 병렬 실행 시에도 충돌하지 않는다.
"""
import os
import sys

# `streamlit run web/app.py`는 web/만 sys.path에 넣으므로 프로젝트 루트를 추가해 server/config 패키지를 import한다.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st  # noqa: E402

from server.repositories import connection, memory_repository, speaker_repository  # noqa: E402
from web.dashboard_logic import (  # noqa: E402
    DISPLAY_NAME_MAX_LENGTH,
    build_overview,
    build_user_options,
    format_timestamp,
    validate_display_name,
)
from web.shutdown import install_shutdown_handlers  # noqa: E402

MEMORY_LIST_LIMIT = 200

st.set_page_config(page_title="멘티오 가족 대시보드", page_icon="🤖", layout="wide")

# 우측 상단 Deploy 버튼 / 툴바 메뉴(⋮) 숨김. 1차는 .streamlit/config.toml(toolbarMode="minimal")이고,
# 프로젝트 루트 밖에서 실행되어 설정 파일이 로드되지 않는 경우를 위한 CSS 폴백이다.
st.markdown(
    """
    <style>
    [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _init_db_pool():
    """
    Streamlit은 상호작용마다 스크립트를 재실행하고 세션별 스레드에서 돌리므로,
    커넥션 풀은 프로세스당 한 번만 생성한다(cache_resource). 종료 시 정리는 web/shutdown.py가 담당한다.
    개별 쿼리는 레포지토리의 get_db_connection() 컨텍스트 매니저가 대여/롤백/반납을 보장하므로
    세션(브라우저 탭)이 커넥션을 붙잡고 있지 않다 → 세션 종료 시 별도 반납 작업이 필요 없다.
    """
    return connection.init_db_pool()


def _flash(message: str, icon: str = "✅") -> None:
    """st.rerun() 이후에도 결과 알림이 보이도록 세션 상태에 보관한다."""
    st.session_state["_flash"] = (message, icon)


def _show_flash() -> None:
    flash = st.session_state.pop("_flash", None)
    if flash:
        st.toast(flash[0], icon=flash[1])


# ---------------------------------------------------------------------------
# 초기화 & 데이터 로드 (매 rerun마다 최신 DB 상태를 조회)
# ---------------------------------------------------------------------------
# Ctrl+C/창 닫기 시 즉시 종료 + 풀 closeall (DB 초기화 성공 여부와 무관하게 먼저 설치)
install_shutdown_handlers(cleanup=connection.close_db_pool)

try:
    _init_db_pool()
except Exception as e:
    st.error(f"DB 연결에 실패했습니다. PostgreSQL(docker-compose)이 실행 중인지 확인해 주세요.\n\n`{e}`")
    st.stop()

_show_flash()

speakers = speaker_repository.list_speaker_profiles(include_inactive=True)
memory_counts = memory_repository.count_active_memories_by_user()
overview = build_overview(speakers, memory_counts)

# ---------------------------------------------------------------------------
# 헤더 & 상태 개요
# ---------------------------------------------------------------------------
title_col, refresh_col = st.columns([5, 1], vertical_alignment="bottom")
title_col.title("🤖 멘티오 가족 대시보드")
title_col.caption("등록된 가족 구성원(화자)과 사용자별 장기 기억(RAG)·프로필 요약을 관리합니다.")
if refresh_col.button("🔄 새로고침", use_container_width=True):
    st.rerun()

m1, m2, m3 = st.columns(3)
m1.metric("활성 가족 구성원", f"{overview.active_speaker_count}명")
m2.metric("전체 등록 화자", f"{overview.total_speaker_count}명")
m3.metric("활성 장기 기억", f"{overview.total_memory_count}개")

tab_speakers, tab_memories = st.tabs(["👨‍👩‍👧 가족/화자 프로필", "🧠 RAG 기억 & 프로필 요약"])

# ---------------------------------------------------------------------------
# [탭 1] 가족/화자 프로필 관리
# ---------------------------------------------------------------------------
with tab_speakers:
    st.info(
        "변경 사항은 DB에 즉시 저장됩니다. 실행 중인 로봇 엔진은 화자 목록을 메모리에 캐싱하므로, "
        "이름/활성 상태 변경은 엔진 재시작 후(또는 다음 음성 온보딩 완료 시) 음성 식별에 반영됩니다.",
        icon="ℹ️",
    )
    show_inactive = st.toggle("비활성 화자 포함", value=True)
    visible_speakers = [s for s in speakers if show_inactive or s.is_active]

    if not visible_speakers:
        st.caption("등록된 화자가 없습니다. 로봇에게 음성으로 등록을 요청하거나 `scripts/enroll_speaker.py`를 사용하세요.")

    for speaker in visible_speakers:
        uid = speaker.user_id
        with st.container(border=True):
            info_col, action_col = st.columns([3, 2])

            with info_col:
                status = ":green[● 활성]" if speaker.is_active else ":gray[○ 비활성]"
                st.markdown(f"#### {speaker.display_name}  \n`{uid}` · {status}")
                st.caption(
                    f"등록 {format_timestamp(speaker.created_at)} · 수정 {format_timestamp(speaker.updated_at)} · "
                    f"활성 기억 {memory_counts.get(uid, 0)}개"
                )
                with st.form(key=f"rename_form_{uid}", border=False):
                    new_name = st.text_input(
                        "표시 이름(호칭)", value=speaker.display_name, max_chars=DISPLAY_NAME_MAX_LENGTH, key=f"name_{uid}"
                    )
                    if st.form_submit_button("이름 저장"):
                        try:
                            name = validate_display_name(new_name)
                        except ValueError as ve:
                            st.warning(str(ve))
                        else:
                            if name == speaker.display_name:
                                st.caption("변경된 내용이 없습니다.")
                            elif speaker_repository.update_display_name(uid, name):
                                _flash(f"'{speaker.display_name}' → '{name}' 이름을 변경했습니다.")
                                st.rerun()
                            else:
                                st.error("이름 변경에 실패했습니다. 서버 로그를 확인해 주세요.")

            with action_col:
                toggle_label = "⏸️ 비활성화" if speaker.is_active else "▶️ 다시 활성화"
                if st.button(toggle_label, key=f"toggle_{uid}", use_container_width=True):
                    if speaker_repository.set_speaker_active(uid, not speaker.is_active):
                        state = "활성화" if not speaker.is_active else "비활성화"
                        _flash(f"'{speaker.display_name}' 화자를 {state}했습니다.")
                        st.rerun()
                    else:
                        st.error("상태 변경에 실패했습니다.")

                confirm = st.checkbox("영구 삭제 확인", key=f"confirm_del_{uid}")
                if st.button(
                    "🗑️ 화자 프로필 삭제",
                    key=f"delete_{uid}",
                    type="primary",
                    disabled=not confirm,
                    use_container_width=True,
                    help="목소리 등록 정보만 삭제되며, 이 사용자의 장기 기억과 프로필 요약은 보존됩니다.",
                ):
                    if speaker_repository.delete_speaker_profile(uid):
                        for key in (f"name_{uid}", f"confirm_del_{uid}"):
                            st.session_state.pop(key, None)
                        _flash(f"'{speaker.display_name}' 화자 프로필을 삭제했습니다.", icon="🗑️")
                        st.rerun()
                    else:
                        st.error("삭제에 실패했습니다.")

# ---------------------------------------------------------------------------
# [탭 2] 화자별 RAG 기억 & 프로필 요약
# ---------------------------------------------------------------------------
with tab_memories:
    user_options = build_user_options(speakers, memory_counts)
    if not user_options:
        st.caption("조회할 사용자가 없습니다. 화자를 등록하거나 로봇과 대화해 기억을 쌓아 보세요.")
    else:
        labels = {opt.user_id: opt.label for opt in user_options}
        selected_uid = st.selectbox(
            "사용자 선택", options=list(labels.keys()), format_func=lambda uid: labels.get(uid, uid), key="selected_user"
        )

        summary = memory_repository.get_profile_summary(selected_uid)
        with st.container(border=True):
            st.markdown("##### 🧾 프로필 요약")
            if summary:
                st.write(summary.summary_text)
                st.caption(
                    f"활성 기억 {summary.source_memory_count}개 기반 · 갱신 {format_timestamp(summary.updated_at)} "
                    "(기억을 삭제해도 요약은 다음 자동 요약 주기에 갱신됩니다)"
                )
            else:
                st.caption("아직 생성된 프로필 요약이 없습니다.")

        memories = memory_repository.get_active_memories(selected_uid, limit=MEMORY_LIST_LIMIT)
        total = memory_counts.get(selected_uid, len(memories))
        st.markdown(f"##### 💭 장기 기억 (최신순, {total}개)")
        if total > len(memories):
            st.caption(f"최근 {len(memories)}개만 표시합니다.")
        if not memories:
            st.caption("저장된 활성 기억이 없습니다.")

        for memory in memories:
            with st.container(border=True):
                text_col, btn_col = st.columns([9, 1], vertical_alignment="center")
                text_col.write(memory.fact_text)
                text_col.caption(f"#{memory.id} · {format_timestamp(memory.created_at)}")
                if btn_col.button(
                    "삭제",
                    key=f"del_mem_{memory.id}",
                    use_container_width=True,
                    help="RAG 검색/요약 대상에서 제외합니다 (소프트 삭제, DB 이력은 보존).",
                ):
                    if memory_repository.soft_delete_memory(memory.id):
                        _flash(f"기억 #{memory.id}을(를) 삭제했습니다.", icon="🗑️")
                        st.rerun()
                    else:
                        st.error("삭제에 실패했습니다. 이미 삭제되었을 수 있으니 새로고침해 주세요.")
