"""
FAISS 벡터 스토어 관리
"""
import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from app.llm.embeddings import GeminiEmbeddings


class FAISSVectorStore:
    """FAISS 벡터 스토어"""
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_model: Optional[GeminiEmbeddings] = None,
        dimension: Optional[int] = None
    ):
        """
        초기화
        
        Args:
            index_path: FAISS 인덱스 저장 경로
            embedding_model: 임베딩 모델
            dimension: 임베딩 차원 (None이면 모델에서 자동 설정)
        """
        self.index_path = index_path or os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
        self.embedding_model = embedding_model or GeminiEmbeddings()
        # 임베딩 모델에서 차원 가져오기 (없으면 기본값 384 사용)
        self.dimension = dimension or getattr(self.embedding_model, 'dimension', 384)
        self.index = None
        self.metadata = []  # 각 벡터의 메타데이터 저장
        
        # 인덱스 로드 시도
        self._load_index()
    
    def _load_index(self):
        """기존 인덱스 로드"""
        index_file = Path(self.index_path) / "index.faiss"
        metadata_file = Path(self.index_path) / "metadata.json"
        
        if index_file.exists() and metadata_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                print(f"기존 인덱스 로드 완료: {len(self.metadata)}개 벡터")
            except Exception as e:
                print(f"인덱스 로드 실패: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """새 인덱스 생성"""
        # L2 거리 기반 인덱스 (Inner Product도 가능)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        print("새 인덱스 생성 완료")
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict],
        batch_size: int = 100
    ):
        """
        문서 추가
        
        Args:
            texts: 문서 텍스트 리스트
            metadatas: 각 문서의 메타데이터 리스트
            batch_size: 배치 크기
        """
        if not texts:
            return
        
        print(f"{len(texts)}개 문서 임베딩 중...")
        
        # 배치로 임베딩 생성
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
            print(f"  {min(i + batch_size, len(texts))}/{len(texts)} 완료")
        
        # numpy 배열로 변환
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # FAISS 인덱스에 추가
        self.index.add(embeddings_array)
        
        # 메타데이터 저장
        self.metadata.extend(metadatas)
        
        print(f"{len(texts)}개 문서 추가 완료 (총 {self.index.ntotal}개 벡터)")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, float]]:
        """
        유사도 검색
        
        Args:
            query: 검색 쿼리
            top_k: 상위 k개 결과
            filters: 필터 조건 (category, channel, updated_at 등)
        
        Returns:
            (메타데이터, 거리) 튜플 리스트
        """
        if self.index.ntotal == 0:
            return []
        
        # 쿼리 임베딩
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # 검색 (거리 계산)
        # 더 많은 결과를 가져온 후 필터링
        search_k = min(top_k * 10, self.index.ntotal)  # 필터링을 위해 더 많이 가져옴
        distances, indices = self.index.search(query_vector, search_k)
        
        # 결과 필터링 및 정렬
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # FAISS가 반환하는 무효 인덱스
                continue
            
            metadata = self.metadata[idx]
            
            # 필터 적용
            if filters:
                if "category" in filters and metadata.get("category") != filters["category"]:
                    continue
                if "channel" in filters:
                    filter_channel = filters["channel"]
                    metadata_channel = metadata.get("channel", "both")
                    if metadata_channel != "both" and metadata_channel != filter_channel:
                        continue
                if "updated_at" in filters:
                    # 날짜 필터링 (간단한 예시)
                    pass  # 필요시 구현
            
            results.append((metadata, float(dist)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self):
        """인덱스 저장"""
        index_dir = Path(self.index_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        index_file = index_dir / "index.faiss"
        metadata_file = index_dir / "metadata.json"
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, str(index_file))
        
        # 메타데이터 저장
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        print(f"인덱스 저장 완료: {index_file}")
        print(f"메타데이터 저장 완료: {metadata_file}")
    
    def get_stats(self) -> Dict:
        """인덱스 통계"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "metadata_count": len(self.metadata)
        }
