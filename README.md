# AI 관상 분석기

얼굴 사진을 업로드하면 OpenAI Vision 모델이 관상을 재미로 풀이해 주는 Streamlit 앱입니다. (오락 목적)

## 로컬 실행
1. `.env.example`을 복사해 `.env`를 만들고 `OPENAI_API_KEY`를 입력
2. `python -m venv .venv` → `.venv\Scripts\activate` → `pip install -r requirements.txt`
3. `streamlit run app.py`

## Streamlit Cloud 배포
1. GitHub에 푸시 (`.env`가 올라가지 않았는지 확인)
2. [share.streamlit.io](https://share.streamlit.io)에서 저장소 연결, Main file path: `app.py`
3. Advanced settings → Secrets에 아래 입력 후 배포
   ```toml
   OPENAI_API_KEY = "sk-..."
   # OPENAI_MODEL = "gpt-4o-mini"  # 선택
   ```
4. Python 버전은 3.11 이상으로 선택 (streamlit 1.67 요구사항)
