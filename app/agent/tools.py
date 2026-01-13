"""
Agent 도구 함수들
"""
from typing import List, Dict, Optional
from app.retriever.hybrid_search import HybridRetriever
from app.llm.gemini_client import GeminiClient


def retrieve_faq(
    retriever: HybridRetriever,
    query: str,
    top_k: int = 5,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    FAQ 검색 도구
    
    Args:
        retriever: 하이브리드 검색기
        query: 검색 쿼리
        top_k: 상위 k개 결과
        filters: 필터 조건
    
    Returns:
        검색 결과 리스트
    """
    results = retriever.search(query, top_k=top_k, filters=filters)
    
    # 재정렬 (선택적)
    if len(results) > top_k:
        results = retriever.rerank(results, query, top_k=top_k)
    
    return results


def classify_intent(llm_client: GeminiClient, question: str) -> str:
    """
    의도 분류 도구
    
    Args:
        llm_client: Gemini 클라이언트
        question: 사용자 질문
    
    Returns:
        의도 카테고리
    """
    return llm_client.classify_intent(question)


def rewrite_query(llm_client: GeminiClient, question: str, intent: str) -> str:
    """
    질의 재작성 도구
    
    Args:
        llm_client: Gemini 클라이언트
        question: 원본 질문
        intent: 의도 카테고리
    
    Returns:
        재작성된 질문
    """
    return llm_client.rewrite_query(question, intent)


def format_answer(
    answer_data: Dict,
    retrieved_docs: List[Dict]
) -> Dict:
    """
    답변 포맷팅 도구
    
    Args:
        answer_data: LLM이 생성한 답변 데이터
        retrieved_docs: 검색된 문서들
    
    Returns:
        포맷팅된 답변
    """
    # citations 포맷팅
    citations = answer_data.get("citations", [])
    
    # citations가 비어있으면 검색된 문서에서 생성
    if not citations and retrieved_docs:
        citations = []
        for doc in retrieved_docs[:5]:  # 상위 5개만
            citation = {
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "snippet": doc.get("snippet", doc.get("text", "")[:200]),
                "faq_id": doc.get("faq_id", "")
            }
            citations.append(citation)
    
    # confidence 조정 (citations가 없으면 low)
    confidence = answer_data.get("confidence", "medium")
    if not citations:
        confidence = "low"
        if not answer_data.get("answer", "").startswith("관련 FAQ를 찾지 못했습니다"):
            answer_data["answer"] = "관련 FAQ를 찾지 못했습니다. 고객센터(1588-0000)로 문의하시거나 인터넷뱅킹 도움말을 참고하세요."
    
    # steps 포맷팅 (리스트가 아니면 변환)
    steps = answer_data.get("steps", [])
    if isinstance(steps, str):
        # 문자열을 리스트로 변환 (줄바꿈 기준)
        steps = [s.strip() for s in steps.split("\n") if s.strip()]
    
    # followups 포맷팅
    followups = answer_data.get("followups", [])
    if isinstance(followups, str):
        followups = [f.strip() for f in followups.split("\n") if f.strip()]
    
    return {
        "answer": answer_data.get("answer", ""),
        "steps": steps[:7],  # 최대 7단계
        "citations": citations,
        "followups": followups[:2],  # 최대 2개
        "confidence": confidence,
        "safety": answer_data.get("safety", "")
    }
