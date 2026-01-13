"""
벡터 인덱스 생성 스크립트
processed_chunks.jsonl을 읽어서 FAISS 인덱스 생성
"""
import json
from pathlib import Path
from app.retriever.vector_store import FAISSVectorStore
from app.llm.embeddings import GeminiEmbeddings


def load_chunks(input_file: Path) -> list:
    """청크 파일 로드"""
    chunks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def main():
    """메인 함수"""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    input_file = data_dir / "processed_chunks.jsonl"
    
    if not input_file.exists():
        print(f"오류: {input_file} 파일이 없습니다.")
        print("먼저 python scripts/preprocess.py를 실행하세요.")
        return
    
    print(f"청크 로드 중: {input_file}")
    chunks = load_chunks(input_file)
    print(f"{len(chunks)}개의 청크 로드 완료")
    
    # 임베딩 모델 초기화
    print("임베딩 모델 초기화 중...")
    try:
        embedding_model = GeminiEmbeddings()
        print("임베딩 모델 초기화 완료")
    except Exception as e:
        print(f"임베딩 모델 초기화 실패: {e}")
        print("GEMINI_API_KEY가 .env 파일에 설정되어 있는지 확인하세요.")
        return
    
    # 벡터 스토어 초기화
    print("벡터 스토어 초기화 중...")
    vector_store = FAISSVectorStore(embedding_model=embedding_model)
    
    # 텍스트와 메타데이터 준비
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "faq_id": chunk["faq_id"],
            "title": chunk["title"],
            "category": chunk["category"],
            "url": chunk["url"],
            "updated_at": chunk["updated_at"],
            "channel": chunk["channel"],
            "text": chunk["text"]  # 검색 결과에 포함
        }
        for chunk in chunks
    ]
    
    # 문서 추가
    print("벡터 인덱스 생성 중...")
    vector_store.add_documents(texts, metadatas, batch_size=50)
    
    # 인덱스 저장
    print("인덱스 저장 중...")
    vector_store.save()
    
    # 통계 출력
    stats = vector_store.get_stats()
    print(f"\n인덱스 생성 완료!")
    print(f"  총 벡터 수: {stats['total_vectors']}")
    print(f"  차원: {stats['dimension']}")
    print(f"  메타데이터 수: {stats['metadata_count']}")


if __name__ == "__main__":
    main()
