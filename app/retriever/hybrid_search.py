"""
하이브리드 검색기 (벡터 + BM25)
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from app.retriever.vector_store import FAISSVectorStore


class HybridRetriever:
    """하이브리드 검색기 (벡터 검색 + BM25)"""
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        chunks_file: Optional[Path] = None
    ):
        """
        초기화
        
        Args:
            vector_store: FAISS 벡터 스토어
            chunks_file: 청크 파일 경로 (BM25용)
        """
        self.vector_store = vector_store
        self.bm25_index = None
        self.chunks = []
        
        if chunks_file:
            self._build_bm25_index(chunks_file)
    
    def _build_bm25_index(self, chunks_file: Path):
        """BM25 인덱스 구축"""
        print("BM25 인덱스 구축 중...")
        
        # 청크 로드
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))
        
        # 토큰화된 텍스트 리스트 생성
        tokenized_corpus = []
        for chunk in self.chunks:
            # 간단한 토큰화 (한글/영문 단어 분리)
            text = chunk.get("text", "")
            tokens = self._tokenize(text)
            tokenized_corpus.append(tokens)
        
        # BM25 인덱스 생성
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"BM25 인덱스 구축 완료: {len(self.chunks)}개 문서")
    
    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화 (한글/영문 단어 분리)"""
        import re
        # 한글, 영문, 숫자를 단어로 분리
        tokens = re.findall(r'[\w가-힣]+', text.lower())
        return tokens
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Dict]:
        """
        하이브리드 검색
        
        Args:
            query: 검색 쿼리
            top_k: 상위 k개 결과
            filters: 필터 조건
            use_hybrid: 하이브리드 검색 사용 여부
            vector_weight: 벡터 검색 가중치
            bm25_weight: BM25 검색 가중치
        
        Returns:
            검색 결과 리스트 (메타데이터 포함)
        """
        if use_hybrid and self.bm25_index:
            return self._hybrid_search(query, top_k, filters, vector_weight, bm25_weight)
        else:
            return self._vector_search(query, top_k, filters)
    
    def _vector_search(self, query: str, top_k: int, filters: Optional[Dict]) -> List[Dict]:
        """벡터 검색만 사용"""
        results = self.vector_store.search(query, top_k=top_k, filters=filters)
        return [self._format_result(metadata, score) for metadata, score in results]
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict],
        vector_weight: float,
        bm25_weight: float
    ) -> List[Dict]:
        """하이브리드 검색 (벡터 + BM25)"""
        # 벡터 검색 (더 많이 가져옴)
        vector_results = self.vector_store.search(query, top_k=top_k * 3, filters=filters)
        
        # BM25 검색
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # 점수 정규화 및 결합
        chunk_scores = {}
        
        # 벡터 검색 결과 점수 (거리를 점수로 변환)
        for metadata, distance in vector_results:
            chunk_id = metadata.get("chunk_id")
            if chunk_id:
                # 거리를 점수로 변환 (거리가 작을수록 높은 점수)
                vector_score = 1.0 / (1.0 + distance)
                chunk_scores[chunk_id] = {
                    "metadata": metadata,
                    "vector_score": vector_score,
                    "bm25_score": 0.0
                }
        
        # BM25 점수 추가
        for i, chunk in enumerate(self.chunks):
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                bm25_score = float(bm25_scores[i])
                
                # BM25 점수 정규화 (0-1 범위)
                if bm25_score > 0:
                    # 최대값으로 정규화 (간단한 방법)
                    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
                    normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
                else:
                    normalized_bm25 = 0.0
                
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id]["bm25_score"] = normalized_bm25
                else:
                    # 벡터 검색에 없었지만 BM25에 있는 경우
                    # 필터 확인
                    if filters:
                        if "category" in filters and chunk.get("category") != filters["category"]:
                            continue
                        if "channel" in filters:
                            filter_channel = filters["channel"]
                            chunk_channel = chunk.get("channel", "both")
                            if chunk_channel != "both" and chunk_channel != filter_channel:
                                continue
                    
                    chunk_scores[chunk_id] = {
                        "metadata": {
                            "chunk_id": chunk.get("chunk_id"),
                            "faq_id": chunk.get("faq_id"),
                            "title": chunk.get("title"),
                            "category": chunk.get("category"),
                            "url": chunk.get("url"),
                            "updated_at": chunk.get("updated_at"),
                            "channel": chunk.get("channel"),
                            "text": chunk.get("text")
                        },
                        "vector_score": 0.0,
                        "bm25_score": normalized_bm25
                    }
        
        # 최종 점수 계산 및 정렬
        final_results = []
        for chunk_id, scores in chunk_scores.items():
            final_score = (
                scores["vector_score"] * vector_weight +
                scores["bm25_score"] * bm25_weight
            )
            final_results.append((scores["metadata"], final_score))
        
        # 점수 기준 정렬 (내림차순)
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 반환
        return [self._format_result(metadata, score) for metadata, score in final_results[:top_k]]
    
    def _format_result(self, metadata: Dict, score: float) -> Dict:
        """검색 결과 포맷팅"""
        return {
            "chunk_id": metadata.get("chunk_id"),
            "faq_id": metadata.get("faq_id"),
            "title": metadata.get("title"),
            "text": metadata.get("text", ""),
            "category": metadata.get("category"),
            "url": metadata.get("url"),
            "updated_at": metadata.get("updated_at"),
            "channel": metadata.get("channel"),
            "score": score,
            "snippet": self._extract_snippet(metadata.get("text", ""), 200)
        }
    
    def _extract_snippet(self, text: str, max_length: int = 200) -> str:
        """텍스트에서 스니펫 추출"""
        if len(text) <= max_length:
            return text
        
        # 첫 부분 반환
        return text[:max_length] + "..."
    
    def rerank(
        self,
        results: List[Dict],
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        재정렬 (간단한 키워드 매칭 기반)
        
        Args:
            results: 검색 결과 리스트
            query: 원본 쿼리
            top_k: 상위 k개 반환
        
        Returns:
            재정렬된 결과
        """
        query_lower = query.lower()
        query_tokens = set(self._tokenize(query))
        
        # 각 결과에 재정렬 점수 부여
        reranked = []
        for result in results:
            text = result.get("text", "").lower()
            title = result.get("title", "").lower()
            
            # 제목에 쿼리 토큰이 포함되면 가산점
            title_score = sum(1 for token in query_tokens if token in title) * 2
            
            # 본문에 쿼리 토큰이 포함되면 가산점
            text_score = sum(1 for token in query_tokens if token in text)
            
            rerank_score = title_score + text_score
            result["rerank_score"] = rerank_score
            reranked.append(result)
        
        # 재정렬 점수 기준 정렬
        reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return reranked[:top_k]
