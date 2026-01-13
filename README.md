# 인터넷뱅킹 FAQ RAG 시스템

Gemini LLM과 FAISS 벡터 DB를 사용한 인터넷뱅킹 FAQ 기반 RAG 시스템입니다.

## 기능

- FAQ 기반 정확한 답변 생성
- 출처(citations) 필수 포함
- Agent 스타일 도구 호출
- PII 보안 감지 및 마스킹
- 평가 시스템 (Recall@5, Faithfulness, Hallucination rate)

## 설치

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 GEMINI_API_KEY 입력
```

## 사용 방법

### 1. FAQ 데이터 준비

```bash
python scripts/generate_sample_faq.py
```

### 2. 전처리 및 청킹

```bash
python scripts/preprocess.py
```

### 3. 벡터 인덱스 생성

```bash
python scripts/build_index.py
```

### 4. 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. API 사용

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "이체가 안돼요",
    "channel": "web",
    "user_context": "공동인증서 오류"
  }'
```

### 6. 평가 실행

```bash
python scripts/evaluate.py
```

## 프로젝트 구조

- `app/`: 메인 애플리케이션 코드
- `data/`: FAQ 데이터 및 인덱스
- `scripts/`: 유틸리티 스크립트
- `tests/`: 테스트 코드

## 환경 변수

- `GEMINI_API_KEY`: Gemini API 키 (필수)
- `LOG_LEVEL`: 로그 레벨 (기본값: INFO)
- `FAISS_INDEX_PATH`: FAISS 인덱스 저장 경로
- `PORT`: 서버 포트 (기본값: 8000)
