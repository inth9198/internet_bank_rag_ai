"""
임베딩 모듈 - sentence-transformers 로컬 임베딩 사용
"""
import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class LocalEmbeddings:
    """sentence-transformers 기반 로컬 임베딩 클라이언트"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        초기화
        
        Args:
            model_name: sentence-transformers 모델명
        """
        from sentence_transformers import SentenceTransformer
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"임베딩 모델 로드 완료: {model_name} (차원: {self.dimension})")
    
    def embed_query(self, text: str) -> List[float]:
        """
        단일 텍스트 임베딩
        
        Args:
            text: 임베딩할 텍스트
        
        Returns:
            임베딩 벡터
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트 임베딩
        
        Args:
            texts: 임베딩할 텍스트 리스트
        
        Returns:
            임베딩 벡터 리스트
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()


# 기본 임베딩 클래스로 LocalEmbeddings 사용
# Gemini 임베딩은 할당량 문제로 비활성화
GeminiEmbeddings = LocalEmbeddings


class GeminiEmbeddingsAPI:
    """Gemini 임베딩 클라이언트 (백업용 - 할당량 문제 발생 가능)"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            api_key: Gemini API 키 (없으면 환경변수에서 로드)
        """
        import google.generativeai as genai
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        genai.configure(api_key=self.api_key)
        self.genai = genai
        self.dimension = 768  # embedding-001의 차원
    
    def embed_query(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        return self.embed_documents([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩"""
        if not texts:
            return []
        
        try:
            embeddings = []
            for text in texts:
                result = self.genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            return embeddings
        except Exception as e:
            raise ValueError(
                f"Gemini 임베딩 API 호출 실패: {e}\n"
                "sentence-transformers를 사용하거나 할당량을 확인하세요."
            )
