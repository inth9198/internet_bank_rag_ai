"""
FAQ 전처리 및 청킹 스크립트
raw_faq.jsonl을 읽어서 processed_chunks.jsonl로 변환
"""
import json
from pathlib import Path
from app.preprocessing.cleaning import clean_faq_item
from app.preprocessing.chunking import chunk_faq_items


def load_faqs(input_file: Path) -> list:
    """FAQ 파일 로드"""
    faqs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                faqs.append(json.loads(line))
    return faqs


def save_chunks(chunks: list, output_file: Path):
    """청크 저장"""
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main():
    """메인 함수"""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    input_file = data_dir / "raw_faq.jsonl"
    output_file = data_dir / "processed_chunks.jsonl"
    
    if not input_file.exists():
        print(f"오류: {input_file} 파일이 없습니다.")
        print("먼저 python scripts/generate_sample_faq.py를 실행하세요.")
        return
    
    print(f"FAQ 로드 중: {input_file}")
    faqs = load_faqs(input_file)
    print(f"{len(faqs)}개의 FAQ 로드 완료")
    
    print("FAQ 정제 중...")
    cleaned_faqs = [clean_faq_item(faq) for faq in faqs]
    
    print("청킹 중...")
    chunks = chunk_faq_items(cleaned_faqs, max_chunk_tokens=500)
    print(f"{len(chunks)}개의 청크 생성 완료")
    
    print(f"저장 중: {output_file}")
    save_chunks(chunks, output_file)
    
    print(f"전처리 완료: {len(chunks)}개의 청크가 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()
