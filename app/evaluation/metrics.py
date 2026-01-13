"""
평가 지표 계산
"""
from typing import List, Dict, Optional
import time


def calculate_recall_at_k(
    retrieved_faq_ids: List[str],
    ground_truth_faq_ids: List[str],
    k: int = 5
) -> float:
    """
    Recall@K 계산
    
    Args:
        retrieved_faq_ids: 검색된 FAQ ID 리스트
        ground_truth_faq_ids: 정답 FAQ ID 리스트
        k: 상위 k개
    
    Returns:
        Recall@K 점수 (0.0 ~ 1.0)
    """
    if not ground_truth_faq_ids:
        return 0.0
    
    top_k_ids = set(retrieved_faq_ids[:k])
    ground_truth_set = set(ground_truth_faq_ids)
    
    # 교집합 크기 / 정답 집합 크기
    intersection = len(top_k_ids & ground_truth_set)
    recall = intersection / len(ground_truth_set) if ground_truth_set else 0.0
    
    return recall


def calculate_faithfulness(
    answer: str,
    retrieved_snippets: List[str],
    ground_truth_snippet: Optional[str] = None
) -> float:
    """
    Faithfulness (근거성) 계산
    답변이 검색된 snippet에 근거하는지 확인
    
    Args:
        answer: 생성된 답변
        retrieved_snippets: 검색된 snippet 리스트
        ground_truth_snippet: 정답 snippet (선택적)
    
    Returns:
        Faithfulness 점수 (0.0 ~ 1.0)
    """
    if not answer or not retrieved_snippets:
        return 0.0
    
    # 간단한 키워드 기반 근거성 계산
    # 실제로는 더 정교한 방법 (예: NLI 모델) 사용 가능
    
    answer_lower = answer.lower()
    answer_words = set(answer_lower.split())
    
    # 검색된 snippet에서 키워드 매칭
    matched_keywords = 0
    total_keywords = len(answer_words)
    
    if total_keywords == 0:
        return 0.0
    
    all_snippet_text = " ".join(retrieved_snippets).lower()
    all_snippet_words = set(all_snippet_text.split())
    
    # 답변의 키워드 중 snippet에 있는 비율
    matched_keywords = len(answer_words & all_snippet_words)
    faithfulness = matched_keywords / total_keywords if total_keywords > 0 else 0.0
    
    return faithfulness


def detect_hallucination(
    answer: str,
    retrieved_snippets: List[str],
    citations: List[Dict]
) -> bool:
    """
    Hallucination (환각) 감지
    근거 없는 단정이 있는지 확인
    
    Args:
        answer: 생성된 답변
        retrieved_snippets: 검색된 snippet 리스트
        citations: 출처 리스트
    
    Returns:
        Hallucination 여부 (True: 환각 있음, False: 없음)
    """
    if not answer:
        return False
    
    # citations가 없으면 환각 가능성 높음
    if not citations:
        # "관련 FAQ를 찾지 못했습니다" 같은 문구가 있으면 환각 아님
        if "찾지 못했습니다" in answer or "찾지 못했" in answer:
            return False
        return True
    
    # 답변이 너무 구체적이지만 snippet에 근거가 없는 경우
    # 간단한 휴리스틱: 답변 길이와 snippet 매칭 비율
    answer_lower = answer.lower()
    answer_words = set(answer_lower.split())
    
    all_snippet_text = " ".join(retrieved_snippets).lower()
    all_snippet_words = set(all_snippet_text.split())
    
    # 답변의 주요 키워드가 snippet에 없는 비율
    unmatched_keywords = answer_words - all_snippet_words
    match_ratio = 1.0 - (len(unmatched_keywords) / len(answer_words) if answer_words else 0.0)
    
    # 매칭 비율이 너무 낮으면 환각 가능성
    if match_ratio < 0.3:  # 30% 미만 매칭
        return True
    
    return False


def calculate_metrics(
    question: str,
    answer: str,
    retrieved_docs: List[Dict],
    citations: List[Dict],
    ground_truth: Dict,
    latency: float,
    tokens: Optional[int] = None
) -> Dict:
    """
    전체 평가 지표 계산
    
    Args:
        question: 질문
        answer: 생성된 답변
        retrieved_docs: 검색된 문서들
        citations: 출처 리스트
        ground_truth: 정답 데이터
        latency: 응답 시간 (초)
        tokens: 사용된 토큰 수 (선택적)
    
    Returns:
        평가 지표 딕셔너리
    """
    # 검색된 FAQ ID 추출
    retrieved_faq_ids = [doc.get("faq_id", "") for doc in retrieved_docs if doc.get("faq_id")]
    
    # 정답 FAQ ID
    ground_truth_faq_ids = ground_truth.get("faq_ids", [])
    
    # 검색된 snippet 추출
    retrieved_snippets = [doc.get("text", "") for doc in retrieved_docs]
    
    # Recall@5
    recall_at_5 = calculate_recall_at_k(retrieved_faq_ids, ground_truth_faq_ids, k=5)
    
    # Faithfulness
    ground_truth_snippet = ground_truth.get("snippet", "")
    faithfulness = calculate_faithfulness(answer, retrieved_snippets, ground_truth_snippet)
    
    # Hallucination
    hallucination = detect_hallucination(answer, retrieved_snippets, citations)
    
    metrics = {
        "recall_at_5": recall_at_5,
        "faithfulness": faithfulness,
        "hallucination": hallucination,
        "latency": latency,
        "num_retrieved": len(retrieved_docs),
        "num_citations": len(citations),
        "has_citations": len(citations) > 0
    }
    
    if tokens is not None:
        metrics["tokens"] = tokens
    
    return metrics


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    """
    여러 평가 결과 집계
    
    Args:
        all_metrics: 평가 지표 리스트
    
    Returns:
        집계된 지표
    """
    if not all_metrics:
        return {}
    
    n = len(all_metrics)
    
    avg_recall_at_5 = sum(m.get("recall_at_5", 0) for m in all_metrics) / n
    avg_faithfulness = sum(m.get("faithfulness", 0) for m in all_metrics) / n
    hallucination_rate = sum(1 for m in all_metrics if m.get("hallucination", False)) / n
    avg_latency = sum(m.get("latency", 0) for m in all_metrics) / n
    avg_citations = sum(m.get("num_citations", 0) for m in all_metrics) / n
    citation_rate = sum(1 for m in all_metrics if m.get("has_citations", False)) / n
    
    total_tokens = sum(m.get("tokens", 0) for m in all_metrics if "tokens" in m)
    
    return {
        "total_questions": n,
        "avg_recall_at_5": avg_recall_at_5,
        "avg_faithfulness": avg_faithfulness,
        "hallucination_rate": hallucination_rate,
        "avg_latency": avg_latency,
        "avg_citations": avg_citations,
        "citation_rate": citation_rate,
        "total_tokens": total_tokens if total_tokens > 0 else None
    }
