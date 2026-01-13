"""
FAQ 텍스트 정제 모듈
HTML 제거, 공백 정리, 특수문자 최소화
"""
import re
from bs4 import BeautifulSoup
from typing import Optional


def remove_html(text: str) -> str:
    """HTML 태그 제거"""
    if not text:
        return ""
    
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def normalize_whitespace(text: str) -> str:
    """공백 정규화"""
    if not text:
        return ""
    
    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text


def remove_special_chars(text: str, keep_newlines: bool = False) -> str:
    """특수문자 최소화 (필요한 것만 유지)"""
    if not text:
        return ""
    
    # 기본적으로 유지할 문자: 한글, 영문, 숫자, 공백, 기본 구두점
    if keep_newlines:
        pattern = r'[^\w\s가-힣.,!?;:()\[\]{}"\'-]'
    else:
        pattern = r'[^\w\s가-힣.,!?;:()\[\]{}"\'-]'
    
    text = re.sub(pattern, '', text)
    return text


def clean_text(text: str, remove_html_tags: bool = True) -> str:
    """
    텍스트 정제 통합 함수
    
    Args:
        text: 정제할 텍스트
        remove_html_tags: HTML 태그 제거 여부
    
    Returns:
        정제된 텍스트
    """
    if not text:
        return ""
    
    # HTML 제거
    if remove_html_tags:
        text = remove_html(text)
    
    # 공백 정규화
    text = normalize_whitespace(text)
    
    # 특수문자 최소화 (기본 구두점은 유지)
    text = remove_special_chars(text, keep_newlines=False)
    
    # 최종 공백 정규화
    text = normalize_whitespace(text)
    
    return text


def clean_faq_item(faq_item: dict) -> dict:
    """
    FAQ 항목 전체 정제
    
    Args:
        faq_item: FAQ 딕셔너리 (title, body 포함)
    
    Returns:
        정제된 FAQ 딕셔너리
    """
    cleaned = faq_item.copy()
    
    if "title" in cleaned:
        cleaned["title"] = clean_text(cleaned["title"])
    
    if "body" in cleaned:
        cleaned["body"] = clean_text(cleaned["body"])
    
    return cleaned
