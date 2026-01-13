"""
Agent 오케스트레이션
플로우 제어, 재검색, 상충 처리
"""
from typing import Dict, List, Optional
from datetime import datetime
from app.retriever.hybrid_search import HybridRetriever
from app.llm.gemini_client import GeminiClient
from app.agent.tools import (
    retrieve_faq,
    classify_intent,
    rewrite_query,
    format_answer
)
from app.security.pii_detector import detect_and_mask_pii


class FAQAgent:
    """FAQ 기반 RAG Agent"""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: GeminiClient
    ):
        """
        초기화
        
        Args:
            retriever: 하이브리드 검색기
            llm_client: Gemini 클라이언트
        """
        self.retriever = retriever
        self.llm_client = llm_client
    
    def process_question(
        self,
        question: str,
        channel: Optional[str] = None,
        user_context: Optional[str] = None
    ) -> Dict:
        """
        질문 처리 메인 함수
        
        Args:
            question: 사용자 질문
            channel: 채널 (web/mobile)
            user_context: 사용자 컨텍스트
        
        Returns:
            답변 딕셔너리
        """
        # PII 감지 및 마스킹
        question_clean, pii_warnings = detect_and_mask_pii(question)
        if user_context:
            user_context_clean, context_warnings = detect_and_mask_pii(user_context)
            pii_warnings.extend(context_warnings)
        else:
            user_context_clean = None
        
        # Step 1: Intent 분류
        intent = classify_intent(self.llm_client, question_clean)
        
        # Step 2: Query 재작성
        rewritten_query = rewrite_query(self.llm_client, question_clean, intent)
        
        # Step 3: FAQ 검색
        filters = {}
        if channel:
            filters["channel"] = channel
        if intent and intent != "기타":
            filters["category"] = intent
        
        retrieved_docs = retrieve_faq(
            self.retriever,
            rewritten_query,
            top_k=10,
            filters=filters
        )
        
        # Step 4: 검색 결과가 부족하면 재검색 (조건 완화)
        if len(retrieved_docs) < 3:
            # 필터 완화하여 재검색
            relaxed_filters = {}
            if channel:
                relaxed_filters["channel"] = channel
            
            relaxed_docs = retrieve_faq(
                self.retriever,
                rewritten_query,
                top_k=10,
                filters=relaxed_filters
            )
            
            # 중복 제거하면서 병합
            existing_ids = {doc.get("chunk_id") for doc in retrieved_docs}
            for doc in relaxed_docs:
                if doc.get("chunk_id") not in existing_ids:
                    retrieved_docs.append(doc)
        
        # Step 5: 상충 FAQ 처리 (최신 우선)
        retrieved_docs = self._resolve_conflicts(retrieved_docs)
        
        # Step 6: LLM 답변 생성
        answer_data = self.llm_client.generate_answer(
            question_clean,
            retrieved_docs[:5],  # 상위 5개만 사용
            user_context_clean
        )
        
        # Step 7: 답변 포맷팅
        formatted_answer = format_answer(answer_data, retrieved_docs)
        
        # Step 8: PII 경고 추가
        if pii_warnings:
            safety_message = "주의: " + " ".join(pii_warnings)
            formatted_answer["safety"] = safety_message
        
        # Step 9: Confidence 조정
        if not formatted_answer.get("citations"):
            formatted_answer["confidence"] = "low"
            if not formatted_answer["answer"].startswith("관련 FAQ를 찾지 못했습니다"):
                formatted_answer["answer"] = (
                    "관련 FAQ를 찾지 못했습니다. "
                    "고객센터(1588-0000)로 문의하시거나 인터넷뱅킹 도움말을 참고하세요."
                )
        
        return formatted_answer
    
    def _resolve_conflicts(self, docs: List[Dict]) -> List[Dict]:
        """
        상충 FAQ 처리 (최신 우선)
        
        Args:
            docs: 검색 결과 리스트
        
        Returns:
            정렬된 리스트 (최신순)
        """
        # updated_at 기준 정렬 (최신순)
        def get_date(doc):
            date_str = doc.get("updated_at", "")
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except:
                return datetime.min
        
        # 날짜 기준 정렬
        sorted_docs = sorted(docs, key=get_date, reverse=True)
        
        # 같은 FAQ의 다른 청크는 하나만 유지 (최신 것)
        seen_faq_ids = set()
        unique_docs = []
        
        for doc in sorted_docs:
            faq_id = doc.get("faq_id")
            if faq_id and faq_id not in seen_faq_ids:
                unique_docs.append(doc)
                seen_faq_ids.add(faq_id)
            elif not faq_id:
                # faq_id가 없으면 그냥 추가
                unique_docs.append(doc)
        
        return unique_docs
