"""
FAQ 청킹 모듈
제목+본문을 기본 단위로 하되, 긴 본문은 문단 단위로 분할
"""
import re
import tiktoken
from typing import List, Dict, Optional
from app.preprocessing.cleaning import clean_text


# 토큰 인코더 (한국어 지원을 위해 cl100k_base 사용)
try:
    encoding = tiktoken.get_encoding("cl100k_base")
except:
    encoding = None


def count_tokens(text: str) -> int:
    """텍스트의 토큰 수 계산"""
    if not encoding:
        # tiktoken이 없으면 대략적인 계산 (한글 1자 = 1토큰, 영문 1단어 = 1토큰)
        return len(text.split())
    
    try:
        return len(encoding.encode(text))
    except:
        return len(text.split())


def split_by_paragraphs(text: str) -> List[str]:
    """문단 단위로 텍스트 분할"""
    if not text:
        return []
    
    # 문단 구분자로 분할 (빈 줄 또는 줄바꿈)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # 각 문단 정제
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def create_chunks(
    faq_id: str,
    title: str,
    body: str,
    category: str,
    url: str,
    updated_at: str,
    channel: str,
    max_chunk_tokens: int = 500,
    overlap_tokens: int = 50
) -> List[Dict]:
    """
    FAQ를 청크로 분할
    
    Args:
        faq_id: FAQ ID
        title: 제목
        body: 본문
        category: 카테고리
        url: URL
        updated_at: 업데이트 날짜
        channel: 채널
        max_chunk_tokens: 최대 청크 토큰 수
        overlap_tokens: 오버랩 토큰 수
    
    Returns:
        청크 리스트
    """
    # 텍스트 정제
    title = clean_text(title)
    body = clean_text(body)
    
    chunks = []
    
    # 제목 + 본문 전체가 max_chunk_tokens 이하면 하나의 청크로
    full_text = f"{title}\n\n{body}"
    full_tokens = count_tokens(full_text)
    
    if full_tokens <= max_chunk_tokens:
        chunk = {
            "chunk_id": f"{faq_id}_chunk_0",
            "faq_id": faq_id,
            "title": title,
            "text": full_text,
            "category": category,
            "url": url,
            "updated_at": updated_at,
            "channel": channel
        }
        chunks.append(chunk)
        return chunks
    
    # 본문이 길면 문단 단위로 분할
    paragraphs = split_by_paragraphs(body)
    
    if not paragraphs:
        # 문단이 없으면 그냥 하나의 청크로
        chunk = {
            "chunk_id": f"{faq_id}_chunk_0",
            "faq_id": faq_id,
            "title": title,
            "text": full_text,
            "category": category,
            "url": url,
            "updated_at": updated_at,
            "channel": channel
        }
        chunks.append(chunk)
        return chunks
    
    # 제목 포함 첫 청크
    current_chunk = f"{title}\n\n"
    current_tokens = count_tokens(current_chunk)
    chunk_idx = 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        # 현재 청크에 추가했을 때 토큰 수 확인
        if current_tokens + para_tokens <= max_chunk_tokens:
            # 추가 가능
            current_chunk += para + "\n\n"
            current_tokens += para_tokens
        else:
            # 현재 청크 저장
            if current_chunk.strip():
                chunk = {
                    "chunk_id": f"{faq_id}_chunk_{chunk_idx}",
                    "faq_id": faq_id,
                    "title": title,
                    "text": current_chunk.strip(),
                    "category": category,
                    "url": url,
                    "updated_at": updated_at,
                    "channel": channel
                }
                chunks.append(chunk)
                chunk_idx += 1
            
            # 오버랩 처리: 이전 청크의 마지막 부분 포함
            if chunks and overlap_tokens > 0:
                prev_text = chunks[-1]["text"]
                prev_tokens = count_tokens(prev_text)
                
                # 이전 청크의 마지막 부분 추출 (overlap_tokens만큼)
                words = prev_text.split()
                overlap_words = []
                overlap_count = 0
                
                for word in reversed(words):
                    word_tokens = count_tokens(word)
                    if overlap_count + word_tokens <= overlap_tokens:
                        overlap_words.insert(0, word)
                        overlap_count += word_tokens
                    else:
                        break
                
                if overlap_words:
                    current_chunk = " ".join(overlap_words) + "\n\n" + para + "\n\n"
                else:
                    current_chunk = f"{title}\n\n{para}\n\n"
            else:
                # 오버랩 없이 새 청크 시작
                current_chunk = f"{title}\n\n{para}\n\n"
            
            current_tokens = count_tokens(current_chunk)
    
    # 마지막 청크 저장
    if current_chunk.strip():
        chunk = {
            "chunk_id": f"{faq_id}_chunk_{chunk_idx}",
            "faq_id": faq_id,
            "title": title,
            "text": current_chunk.strip(),
            "category": category,
            "url": url,
            "updated_at": updated_at,
            "channel": channel
        }
        chunks.append(chunk)
    
    return chunks


def chunk_faq_items(faq_items: List[Dict], max_chunk_tokens: int = 500) -> List[Dict]:
    """
    FAQ 항목 리스트를 청크로 변환
    
    Args:
        faq_items: FAQ 딕셔너리 리스트
        max_chunk_tokens: 최대 청크 토큰 수
    
    Returns:
        청크 리스트
    """
    all_chunks = []
    
    for faq in faq_items:
        chunks = create_chunks(
            faq_id=faq.get("faq_id", ""),
            title=faq.get("title", ""),
            body=faq.get("body", ""),
            category=faq.get("category", ""),
            url=faq.get("url", ""),
            updated_at=faq.get("updated_at", ""),
            channel=faq.get("channel", "both"),
            max_chunk_tokens=max_chunk_tokens
        )
        all_chunks.extend(chunks)
    
    return all_chunks
