"""
FastAPI 메인 애플리케이션
"""
import os
import uuid
import time
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.models.schemas import (
    AskRequest,
    AskResponse,
    IndexRequest,
    IndexResponse,
    Citation,
    Confidence
)
from app.retriever.vector_store import FAISSVectorStore
from app.retriever.hybrid_search import HybridRetriever
from app.llm.gemini_client import GeminiClient
from app.llm.embeddings import GeminiEmbeddings
from app.agent.orchestrator import FAQAgent

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="인터넷뱅킹 FAQ RAG API",
    description="FAQ 기반 질의응답 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수 (초기화는 lazy loading)
_agent: Optional[FAQAgent] = None
_vector_store: Optional[FAISSVectorStore] = None
_retriever: Optional[HybridRetriever] = None


def get_agent() -> FAQAgent:
    """Agent 인스턴스 가져오기 (lazy loading)"""
    global _agent, _vector_store, _retriever
    
    if _agent is None:
        logger.info("Agent 초기화 중...")
        
        # 벡터 스토어 초기화
        embedding_model = GeminiEmbeddings()
        _vector_store = FAISSVectorStore(embedding_model=embedding_model)
        
        # 하이브리드 검색기 초기화
        base_dir = Path(__file__).parent.parent
        chunks_file = base_dir / "data" / "processed_chunks.jsonl"
        _retriever = HybridRetriever(
            _vector_store,
            chunks_file if chunks_file.exists() else None
        )
        
        # LLM 클라이언트 초기화
        llm_client = GeminiClient()
        
        # Agent 초기화
        _agent = FAQAgent(_retriever, llm_client)
        
        logger.info("Agent 초기화 완료")
    
    return _agent


@app.get("/", response_class=HTMLResponse)
async def root():
    """간단한 웹 UI"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>인터넷뱅킹 FAQ RAG 시스템</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #555;
            }
            input, select, textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                box-sizing: border-box;
            }
            textarea {
                height: 100px;
                resize: vertical;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
            }
            button:hover {
                background-color: #45a049;
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                display: none;
            }
            .result.show {
                display: block;
            }
            .answer {
                font-size: 16px;
                line-height: 1.6;
                margin-bottom: 20px;
            }
            .steps {
                margin: 20px 0;
            }
            .steps ol {
                margin-left: 20px;
            }
            .citations {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }
            .citation {
                margin: 10px 0;
                padding: 10px;
                background: white;
                border-left: 3px solid #4CAF50;
            }
            .citation a {
                color: #4CAF50;
                text-decoration: none;
            }
            .confidence {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
            }
            .confidence.high {
                background-color: #4CAF50;
                color: white;
            }
            .confidence.medium {
                background-color: #ff9800;
                color: white;
            }
            .confidence.low {
                background-color: #f44336;
                color: white;
            }
            .safety {
                margin-top: 15px;
                padding: 10px;
                background-color: #fff3cd;
                border-left: 3px solid #ffc107;
                color: #856404;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>인터넷뱅킹 FAQ 질의응답</h1>
            <form id="questionForm">
                <div class="form-group">
                    <label for="question">질문 *</label>
                    <textarea id="question" name="question" required placeholder="예: 이체가 안돼요"></textarea>
                </div>
                <div class="form-group">
                    <label for="channel">채널</label>
                    <select id="channel" name="channel">
                        <option value="">선택 안함</option>
                        <option value="web">웹</option>
                        <option value="mobile">모바일</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="context">상황/컨텍스트</label>
                    <textarea id="context" name="context" placeholder="예: 공동인증서 오류"></textarea>
                </div>
                <button type="submit">질문하기</button>
            </form>
            <div id="result" class="result">
                <h2>답변</h2>
                <div id="answer" class="answer"></div>
                <div id="steps" class="steps"></div>
                <div id="citations" class="citations"></div>
                <div id="safety" class="safety" style="display: none;"></div>
            </div>
        </div>
        <script>
            document.getElementById('questionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const question = document.getElementById('question').value;
                const channel = document.getElementById('channel').value;
                const context = document.getElementById('context').value;
                
                const resultDiv = document.getElementById('result');
                resultDiv.classList.add('show');
                resultDiv.innerHTML = '<p>처리 중...</p>';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: question,
                            channel: channel || null,
                            user_context: context || null
                        })
                    });
                    
                    const data = await response.json();
                    
                    let html = '<h2>답변</h2>';
                    html += '<div class="answer">' + data.answer + '</div>';
                    
                    if (data.steps && data.steps.length > 0) {
                        html += '<div class="steps"><h3>해결 절차</h3><ol>';
                        data.steps.forEach(step => {
                            html += '<li>' + step + '</li>';
                        });
                        html += '</ol></div>';
                    }
                    
                    if (data.citations && data.citations.length > 0) {
                        html += '<div class="citations"><h3>참고 FAQ</h3>';
                        data.citations.forEach(citation => {
                            html += '<div class="citation">';
                            html += '<strong><a href="' + citation.url + '" target="_blank">' + citation.title + '</a></strong><br>';
                            html += '<small>' + citation.snippet + '</small>';
                            html += '</div>';
                        });
                        html += '</div>';
                    }
                    
                    html += '<p>신뢰도: <span class="confidence ' + data.confidence + '">' + data.confidence.toUpperCase() + '</span></p>';
                    
                    if (data.safety) {
                        html += '<div class="safety">' + data.safety + '</div>';
                    }
                    
                    resultDiv.innerHTML = html;
                } catch (error) {
                    resultDiv.innerHTML = '<p style="color: red;">오류가 발생했습니다: ' + error.message + '</p>';
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    질문 처리 엔드포인트
    
    Args:
        request: 질문 요청
    
    Returns:
        답변 응답
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"[{request_id}] 질문 수신: {request.question[:50]}...")
    
    try:
        # Agent 가져오기
        agent = get_agent()
        
        # 질문 처리
        result = agent.process_question(
            question=request.question,
            channel=request.channel.value if request.channel else None,
            user_context=request.user_context
        )
        
        # 응답 포맷팅
        response = AskResponse(
            answer=result.get("answer", ""),
            steps=result.get("steps", []),
            citations=[
                Citation(**citation) for citation in result.get("citations", [])
            ],
            followups=result.get("followups", []),
            confidence=Confidence(result.get("confidence", "low")),
            safety=result.get("safety")
        )
        
        latency = time.time() - start_time
        logger.info(
            f"[{request_id}] 처리 완료 - "
            f"latency: {latency:.2f}s, "
            f"confidence: {response.confidence}, "
            f"citations: {len(response.citations)}"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"[{request_id}] 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {str(e)}")


@app.post("/index", response_model=IndexResponse)
async def rebuild_index(request: IndexRequest = IndexRequest()):
    """
    벡터 인덱스 재구축 엔드포인트
    
    Args:
        request: 재인덱싱 요청
    
    Returns:
        재인덱싱 응답
    """
    try:
        logger.info("인덱스 재구축 시작...")
        
        # 인덱스 재구축 스크립트 실행
        import subprocess
        result = subprocess.run(
            ["python", "scripts/build_index.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            # Agent 재초기화
            global _agent, _vector_store, _retriever
            _agent = None
            _vector_store = None
            _retriever = None
            
            logger.info("인덱스 재구축 완료")
            return IndexResponse(
                success=True,
                message="인덱스 재구축이 완료되었습니다.",
                total_vectors=None  # 필요시 벡터 스토어에서 가져올 수 있음
            )
        else:
            logger.error(f"인덱스 재구축 실패: {result.stderr}")
            return IndexResponse(
                success=False,
                message=f"인덱스 재구축 실패: {result.stderr}",
                total_vectors=None
            )
    
    except Exception as e:
        logger.error(f"인덱스 재구축 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"인덱스 재구축 중 오류가 발생했습니다: {str(e)}")


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
