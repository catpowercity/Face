import base64
import hashlib
import html
import io
import os
import re

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps

# .env 파일 로드 (로컬 개발용)
load_dotenv()

# ---------------------------------------------------------------- 설정 상수
DEFAULT_MODEL = "gpt-4o-mini"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5MB
MAX_SIDE = 1024  # 전송 전 긴 변 최대 픽셀
NO_FACE = "NO_FACE"

STYLES = {
    "전통": "점잖고 고풍스러운 전통 관상가의 말투(~하오, ~이로다 같은 사극 톤은 과하지 않게)로 풀이하세요.",
    "현대적": "친근하고 세련된 현대적인 말투(존댓말)로 풀이하세요.",
    "유머": "재치 있고 유머러스한 말투(존댓말)로, 가볍게 웃을 수 있게 풀이하세요.",
}

SYSTEM_PROMPT = """당신은 오락용 전통 관상 풀이 전문가입니다. 사용자가 올린 얼굴 사진을 보고, 재미로 보는 동양 관상학 관점에서 풀이합니다.

규칙:
- 이 풀이는 과학적 근거가 없는 오락 콘텐츠입니다. 긍정적이고 격려하는 톤을 유지하고, 부정적으로 단정하는 표현은 쓰지 마세요.
- 건강, 질병, 인종, 종교, 범죄 성향, 신원(누구인지)은 절대 추정하거나 언급하지 마세요.
- 사진에 사람 얼굴이 없거나 얼굴을 식별하기 어려우면 풀이하지 말고 정확히 "NO_FACE" 라고만 답하세요.
- 반드시 한국어 마크다운으로, 아래 형식과 '## ' 제목을 그대로 사용하세요.

## 한줄 총평
(한 문장)

## 부위별 풀이
### 이마 (초년운)
### 눈 (성격·지혜)
### 눈썹 (대인관계)
### 코 (재물운)
### 입 (말·식복)
### 귀
### 턱/얼굴형 (말년운)
(각 부위마다 1~2문장)

## 종합 운세
- **재물운**:
- **직업운**:
- **연애·인간관계운**:

## 조언
(실천 가능한 따뜻한 한마디)"""

USER_PROMPT_TEMPLATE = "이 사진으로 관상을 풀이해 주세요. {style_instruction}"

# ---------------------------------------------------------------- 디자인 상수
COLOR_BG = "#0f1b2d"
COLOR_CARD = "#16263d"
COLOR_GOLD = "#d4af37"
COLOR_TEXT = "#f3ead7"
COLOR_MUTED = "#a9b4c6"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@600;700&display=swap');
:root {{
  --bg: {COLOR_BG}; --card: {COLOR_CARD}; --gold: {COLOR_GOLD};
  --text: {COLOR_TEXT}; --muted: {COLOR_MUTED};
}}
.block-container {{ padding-top: 2rem; max-width: 720px; }}
.gx-title {{
  font-family: 'Noto Serif KR', serif; font-weight: 700; font-size: 2.2rem;
  color: var(--gold); text-align: center; margin: 0; letter-spacing: 0.08em;
}}
.gx-sub {{ text-align: center; color: var(--muted); margin: 0.3rem 0 1rem; }}
.gx-notice {{
  border: 1px solid var(--gold); border-radius: 10px; padding: 0.6rem 0.9rem;
  color: var(--text); background: rgba(212,175,55,0.08);
  font-size: 0.9rem; text-align: center; margin-bottom: 1.2rem;
}}
.gx-summary {{
  background: var(--card); border-left: 5px solid var(--gold); border-radius: 10px;
  padding: 1rem 1.2rem; margin: 1rem 0; color: var(--text);
  font-family: 'Noto Serif KR', serif; font-size: 1.15rem; line-height: 1.6;
}}
.gx-footer {{ color: var(--muted); font-size: 0.8rem; text-align: center; margin-top: 2rem; }}
div.stButton > button, div.stDownloadButton > button {{ border-radius: 10px; font-weight: 600; }}
img {{ border-radius: 12px; }}
</style>
"""


# ---------------------------------------------------------------- 키/모델
def get_api_key() -> str | None:
    """st.secrets 우선, 없으면 환경변수(.env)에서 API 키를 가져온다."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass  # 로컬에 secrets 파일이 없는 경우
    return os.getenv("OPENAI_API_KEY")


def get_model() -> str:
    return os.getenv("OPENAI_MODEL") or DEFAULT_MODEL


# ---------------------------------------------------------------- 이미지 처리
def preprocess_image(image_bytes: bytes) -> bytes:
    """EXIF 보정 → RGB 변환 → 긴 변 1024px 이하 리사이즈 → JPEG 바이트."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        # 투명 배경은 흰색으로 합성
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def to_data_url(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------- 프롬프트/API
def build_messages(jpeg_bytes: bytes, style: str) -> list[dict]:
    user_text = USER_PROMPT_TEMPLATE.format(style_instruction=STYLES[style])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": to_data_url(jpeg_bytes)}},
            ],
        },
    ]


def analyze_face(jpeg_bytes: bytes, style: str, api_key: str) -> str:
    """OpenAI Vision 호출. 예외는 호출부에서 처리한다."""
    client = OpenAI(api_key=api_key, timeout=60.0, max_retries=1)
    response = client.chat.completions.create(
        model=get_model(),
        messages=build_messages(jpeg_bytes, style),
    )
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        return "REFUSED"
    return (message.content or "").strip()


def friendly_error(exc: Exception) -> str:
    """API 예외를 사용자용 한국어 메시지로 변환한다 (키/스택트레이스 비노출)."""
    if isinstance(exc, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다. 키 설정을 확인해 주세요."
    if isinstance(exc, openai.RateLimitError):
        return "요청이 너무 많거나 사용 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, openai.APITimeoutError):
        return "응답 시간이 너무 오래 걸렸습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, openai.APIConnectionError):
        return "OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인해 주세요."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI 서버에서 오류가 발생했습니다. (상태 코드 {exc.status_code})"
    return "알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


# ---------------------------------------------------------------- 결과 렌더링
def split_sections(text: str) -> list[tuple[str, str]]:
    """'## ' 제목 기준으로 (제목, 본문) 목록을 만든다. 실패하면 빈 리스트."""
    parts = re.split(r"^## +", text, flags=re.MULTILINE)
    sections = []
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        sections.append((title.strip(), body.strip()))
    return sections


def render_result(sections: list[tuple[str, str]]):
    for title, body in sections:
        if "총평" in title:
            # 모델 출력은 HTML 이스케이프 후 삽입 (마크다운 강조 기호는 제거)
            safe = html.escape(body.replace("**", "").replace("__", ""))
            st.markdown(f'<div class="gx-summary">{safe}</div>', unsafe_allow_html=True)
        elif "부위별" in title:
            with st.expander(title, expanded=True):
                st.markdown(body)
        else:
            st.subheader(title)
            st.markdown(body)


# ---------------------------------------------------------------- 입력 UI
def pick_image():
    """업로드/카메라 탭에서 이미지를 받아 (바이트, 오류메시지)를 돌려준다."""
    # 선택한 방식의 위젯만 렌더링 (카메라 권한을 불필요하게 요청하지 않도록)
    mode = st.radio("입력 방식", ["📁 사진 업로드", "📷 카메라 촬영"], horizontal=True)
    if mode.endswith("업로드"):
        file = st.file_uploader("얼굴 사진 (jpg, jpeg, png / 5MB 이하)", type=["jpg", "jpeg", "png"])
    else:
        file = st.camera_input("정면을 바라보고 촬영해 주세요")
    if file is None:
        return None, None
    data = file.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        return None, "파일이 5MB를 넘습니다. 더 작은 사진을 올려 주세요."
    return data, None


# ---------------------------------------------------------------- 메인
def main():
    st.set_page_config(page_title="AI 관상 분석기", page_icon="🔮", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown('<p class="gx-title">🔮 AI 관상 분석기</p>', unsafe_allow_html=True)
    st.markdown('<p class="gx-sub">얼굴 사진 한 장으로 보는 재미있는 관상 풀이</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gx-notice">이 서비스는 오락 목적이며 과학적 근거가 없습니다.</div>',
        unsafe_allow_html=True,
    )

    api_key = get_api_key()
    if not api_key:
        st.error(
            "OpenAI API 키가 없습니다. `.env`에 OPENAI_API_KEY를 설정하거나 "
            "Streamlit Secrets에 등록해 주세요."
        )
        st.stop()

    run_analysis(api_key)
    _footer()  # 어떤 경로로 끝나도 개인정보 고지는 항상 표시


def run_analysis(api_key: str):
    style = st.selectbox("풀이 스타일", list(STYLES.keys()), index=1)
    data, error = pick_image()
    if error:
        st.error(error)
        return
    if data is None:
        st.session_state.pop("result", None)  # 이미지가 없으면 이전 결과 제거
        st.session_state.pop("result_key", None)
        return

    # 이미지 검증 + 전처리
    try:
        jpeg = preprocess_image(data)
    except Exception:
        st.error("이미지를 열 수 없습니다. 다른 사진으로 다시 시도해 주세요.")
        return

    st.image(jpeg, caption="분석할 사진", width=320)

    key = hashlib.sha256(jpeg).hexdigest() + "|" + style
    if st.session_state.get("result_key") != key:
        # 사진이나 스타일이 바뀌면 이전 결과를 지운다
        st.session_state.pop("result", None)

    if st.button("관상 보기", type="primary", width="stretch"):
        if st.session_state.get("result_key") != key or "result" not in st.session_state:
            with st.spinner("관상을 살피는 중입니다..."):
                try:
                    st.session_state["result"] = analyze_face(jpeg, style, api_key)
                    st.session_state["result_key"] = key
                except Exception as exc:
                    st.error(friendly_error(exc))
                    return

    result = st.session_state.get("result")
    if result and st.session_state.get("result_key") == key:
        sections = split_sections(result)
        if result.startswith(NO_FACE):
            st.warning("얼굴이 잘 보이는 정면 사진으로 다시 올려주세요.")
        elif result == "REFUSED" or not sections:
            # 거절 응답이거나 형식에 맞지 않는 응답: 결과로 취급하지 않는다
            st.warning("이 사진은 분석할 수 없었어요. 다른 사진으로 다시 시도해 주세요.")
        else:
            render_result(sections)
            st.download_button(
                "결과 텍스트로 저장",
                data=result,
                file_name="gwansang_result.txt",
                mime="text/plain",
                width="stretch",
            )


def _footer():
    st.markdown(
        '<div class="gx-footer">사진은 OpenAI API로 전송되며 서버에 저장되지 않습니다.<br>'
        "본인 또는 동의받은 사진만 사용하세요.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
