"""
Pydantic 스키마 정의
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Channel(str, Enum):
    """채널 타입"""
    WEB = "web"
    MOBILE = "mobile"
    BOTH = "both"


class Confidence(str, Enum):
    """신뢰도 레벨"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Citation(BaseModel):
    """FAQ 출처"""
    title: str = Field(..., description="FAQ 제목")
    url: str = Field(..., description="FAQ URL")
    snippet: str = Field(..., description="발췌 문장")
    faq_id: Optional[str] = Field(None, description="FAQ ID")


class Step(BaseModel):
    """해결 절차 단계"""
    step_number: int = Field(..., description="단계 번호")
    description: str = Field(..., description="단계 설명")


class AskRequest(BaseModel):
    """질문 요청"""
    question: str = Field(..., description="사용자 질문", min_length=1)
    channel: Optional[Channel] = Field(None, description="채널 (web/mobile)")
    user_context: Optional[str] = Field(None, description="사용자 컨텍스트/상황")


class AskResponse(BaseModel):
    """답변 응답"""
    answer: str = Field(..., description="최종 답변")
    steps: List[str] = Field(default_factory=list, description="해결 절차 (최대 7단계)")
    citations: List[Citation] = Field(default_factory=list, description="FAQ 출처 리스트")
    followups: List[str] = Field(default_factory=list, description="추가 질문 (최대 2개)")
    confidence: Confidence = Field(..., description="신뢰도")
    safety: Optional[str] = Field(None, description="PII 안내/주의 문구")


class IndexRequest(BaseModel):
    """재인덱싱 요청"""
    force: bool = Field(False, description="강제 재인덱싱 여부")


class IndexResponse(BaseModel):
    """재인덱싱 응답"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="결과 메시지")
    total_vectors: Optional[int] = Field(None, description="총 벡터 수")
