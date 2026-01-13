"""
평가 스크립트
평가셋을 사용하여 시스템 성능 평가
"""
import json
import time
from pathlib import Path
from typing import List, Dict

from app.retriever.vector_store import FAISSVectorStore
from app.retriever.hybrid_search import HybridRetriever
from app.llm.gemini_client import GeminiClient
from app.llm.embeddings import GeminiEmbeddings
from app.agent.orchestrator import FAQAgent
from app.evaluation.metrics import calculate_metrics, aggregate_metrics


def load_test_set(test_set_file: Path) -> List[Dict]:
    """평가셋 로드"""
    with open(test_set_file, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_question(
    agent: FAQAgent,
    question_data: Dict,
    retriever: HybridRetriever
) -> Dict:
    """단일 질문 평가"""
    question = question_data["question"]
    ground_truth = {
        "faq_ids": question_data.get("faq_ids", []),
        "snippet": question_data.get("snippet", "")
    }
    
    start_time = time.time()
    
    # 질문 처리
    result = agent.process_question(question=question)
    
    latency = time.time() - start_time
    
    # 검색된 문서 가져오기 (평가용)
    # 실제로는 agent 내부에서 검색된 문서를 반환하도록 수정 필요
    # 여기서는 간단히 재검색
    retrieved_docs = retriever.search(question, top_k=10)
    
    # 지표 계산
    metrics = calculate_metrics(
        question=question,
        answer=result.get("answer", ""),
        retrieved_docs=retrieved_docs,
        citations=result.get("citations", []),
        ground_truth=ground_truth,
        latency=latency
    )
    
    return {
        "question": question,
        "answer": result.get("answer", ""),
        "ground_truth_summary": question_data.get("ground_truth_summary", ""),
        "metrics": metrics
    }


def main():
    """메인 함수"""
    base_dir = Path(__file__).parent.parent
    
    # 평가셋 로드
    test_set_file = base_dir / "app" / "evaluation" / "test_set.json"
    if not test_set_file.exists():
        print(f"오류: {test_set_file} 파일이 없습니다.")
        return
    
    print("평가셋 로드 중...")
    test_set = load_test_set(test_set_file)
    print(f"{len(test_set)}개 질문 로드 완료")
    
    # Agent 초기화
    print("Agent 초기화 중...")
    try:
        embedding_model = GeminiEmbeddings()
        vector_store = FAISSVectorStore(embedding_model=embedding_model)
        
        chunks_file = base_dir / "data" / "processed_chunks.jsonl"
        retriever = HybridRetriever(
            vector_store,
            chunks_file if chunks_file.exists() else None
        )
        
        llm_client = GeminiClient()
        agent = FAQAgent(retriever, llm_client)
        print("Agent 초기화 완료")
    except Exception as e:
        print(f"Agent 초기화 실패: {e}")
        return
    
    # 평가 실행
    print("\n평가 시작...")
    all_results = []
    
    for i, question_data in enumerate(test_set, 1):
        print(f"\n[{i}/{len(test_set)}] 평가 중: {question_data['question'][:50]}...")
        
        try:
            result = evaluate_question(agent, question_data, retriever)
            all_results.append(result)
            
            metrics = result["metrics"]
            print(f"  Recall@5: {metrics['recall_at_5']:.3f}")
            print(f"  Faithfulness: {metrics['faithfulness']:.3f}")
            print(f"  Hallucination: {metrics['hallucination']}")
            print(f"  Latency: {metrics['latency']:.2f}s")
        except Exception as e:
            print(f"  오류 발생: {e}")
            continue
    
    # 결과 집계
    print("\n" + "="*50)
    print("평가 결과 집계")
    print("="*50)
    
    all_metrics = [r["metrics"] for r in all_results]
    aggregated = aggregate_metrics(all_metrics)
    
    print(f"\n총 질문 수: {aggregated['total_questions']}")
    print(f"평균 Recall@5: {aggregated['avg_recall_at_5']:.3f}")
    print(f"평균 Faithfulness: {aggregated['avg_faithfulness']:.3f}")
    print(f"Hallucination Rate: {aggregated['hallucination_rate']:.3f}")
    print(f"평균 Latency: {aggregated['avg_latency']:.2f}초")
    print(f"평균 Citations: {aggregated['avg_citations']:.2f}")
    print(f"Citation Rate: {aggregated['citation_rate']:.3f}")
    
    if aggregated.get("total_tokens"):
        print(f"총 Tokens: {aggregated['total_tokens']}")
    
    # 결과 저장
    output_file = base_dir / "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "aggregated": aggregated,
            "detailed_results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n상세 결과 저장: {output_file}")


if __name__ == "__main__":
    main()
