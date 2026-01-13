"""
PII (개인식별정보) 감지 및 마스킹
"""
import re
from typing import Tuple, List, Optional


# PII 패턴 정의
PII_PATTERNS = {
    "password": [
        r"비밀번호\s*[:=]\s*[\w!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{6,}",  # 비밀번호: xxx
        r"패스워드\s*[:=]\s*[\w!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{6,}",
    ],
    "security_card_full": [
        r"보안카드\s*(전체|번호|전체번호)\s*[:=]\s*[\d\s\-]{12,}",  # 보안카드 전체번호
        r"보안카드\s*[:=]\s*[\d\s\-]{12,}",
    ],
    "otp_full": [
        r"OTP\s*(전체|번호|전체번호)\s*[:=]\s*[\d]{6,}",  # OTP 전체번호
        r"일회용\s*비밀번호\s*(전체|번호)\s*[:=]\s*[\d]{6,}",
    ],
    "account_full": [
        r"계좌번호\s*(전체|번호)\s*[:=]\s*[\d\s\-]{10,}",  # 계좌번호 전체
        r"계좌\s*[:=]\s*[\d\s\-]{10,}",
    ],
    "card_number": [
        r"카드번호\s*(전체|번호)\s*[:=]\s*[\d\s\-]{13,}",  # 카드번호 전체
        r"신용카드\s*(전체|번호)\s*[:=]\s*[\d\s\-]{13,}",
    ],
    "resident_number": [
        r"주민등록번호\s*[:=]\s*[\d]{6}\s*[-]?\s*[\d]{7}",  # 주민등록번호
        r"주민번호\s*[:=]\s*[\d]{6}\s*[-]?\s*[\d]{7}",
    ],
    "phone_full": [
        r"전화번호\s*(전체|번호)\s*[:=]\s*[\d\s\-]{10,}",  # 전화번호 전체
    ]
}


def mask_text(text: str, pattern: str, mask_char: str = "*") -> Tuple[str, bool]:
    """
    텍스트 마스킹
    
    Args:
        text: 원본 텍스트
        pattern: 마스킹할 패턴 (정규식)
        mask_char: 마스킹 문자
    
    Returns:
        (마스킹된 텍스트, 마스킹 여부)
    """
    matches = re.finditer(pattern, text, re.IGNORECASE)
    masked_text = text
    was_masked = False
    
    for match in reversed(list(matches)):  # 뒤에서부터 처리 (인덱스 유지)
        start, end = match.span()
        # 패턴의 값 부분만 마스킹 (예: "비밀번호: abc123" -> "비밀번호: ***")
        match_text = match.group()
        
        # ":" 또는 "=" 이후 부분만 마스킹
        if ":" in match_text or "=" in match_text:
            parts = re.split(r"[:=]", match_text, 1)
            if len(parts) == 2:
                prefix = parts[0] + (":" if ":" in match_text else "=")
                value = parts[1].strip()
                masked_value = mask_char * len(value)
                masked_match = prefix + " " + masked_value
            else:
                masked_match = mask_char * len(match_text)
        else:
            masked_match = mask_char * len(match_text)
        
        masked_text = masked_text[:start] + masked_match + masked_text[end:]
        was_masked = True
    
    return masked_text, was_masked


def detect_and_mask_pii(text: str) -> Tuple[str, List[str]]:
    """
    PII 감지 및 마스킹
    
    Args:
        text: 검사할 텍스트
    
    Returns:
        (마스킹된 텍스트, 경고 메시지 리스트)
    """
    if not text:
        return text, []
    
    warnings = []
    masked_text = text
    
    # 각 PII 타입 검사
    for pii_type, patterns in PII_PATTERNS.items():
        for pattern in patterns:
            masked_text, was_masked = mask_text(masked_text, pattern)
            
            if was_masked:
                warning_msg = _get_warning_message(pii_type)
                if warning_msg not in warnings:
                    warnings.append(warning_msg)
    
    return masked_text, warnings


def _get_warning_message(pii_type: str) -> str:
    """PII 타입별 경고 메시지"""
    messages = {
        "password": "비밀번호는 입력하지 마세요. 비밀번호 찾기 기능을 사용하세요.",
        "security_card_full": "보안카드 전체 번호는 입력하지 마세요. 화면에 표시된 좌표에 해당하는 번호만 입력하세요.",
        "otp_full": "OTP 전체 번호는 입력하지 마세요. OTP 앱에서 생성된 번호만 입력하세요.",
        "account_full": "계좌번호 전체는 입력하지 마세요. 필요한 경우 마지막 4자리만 확인하세요.",
        "card_number": "카드번호 전체는 입력하지 마세요.",
        "resident_number": "주민등록번호는 입력하지 마세요.",
        "phone_full": "전화번호 전체는 입력하지 마세요."
    }
    
    return messages.get(pii_type, "민감한 정보는 입력하지 마세요.")


def check_pii_in_input(question: str, user_context: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    입력에 PII가 포함되어 있는지 확인
    
    Args:
        question: 질문 텍스트
        user_context: 사용자 컨텍스트
    
    Returns:
        (PII 포함 여부, 경고 메시지 리스트)
    """
    _, warnings_q = detect_and_mask_pii(question)
    
    warnings = warnings_q.copy()
    if user_context:
        _, warnings_c = detect_and_mask_pii(user_context)
        warnings.extend(warnings_c)
    
    return len(warnings) > 0, warnings
