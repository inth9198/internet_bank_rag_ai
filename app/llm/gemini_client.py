"""
Gemini API 클라이언트
"""
import os
import logging
from typing import Optional, List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiClient:
    """Gemini API 클라이언트"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-flash-latest"):
        """
        초기화
        
        Args:
            api_key: Gemini API 키
            model_name: 사용할 모델명 (gemini-1.5-flash, gemini-1.5-pro 등)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Gemini 모델 초기화 완료: {model_name}")
    
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 사용자 프롬프트
            system_instruction: 시스템 지시사항
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
        
        Returns:
            생성된 텍스트
        """
        generation_config = {
            "temperature": temperature,
        }
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        try:
            # 시스템 지시사항이 있으면 프롬프트에 포함
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n---\n\n{prompt}"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini API 호출 실패: {e}")
            raise ValueError(f"Gemini API 호출 실패: {e}")
    
    def classify_intent(self, question: str) -> str:
        """
        의도 분류
        
        Args:
            question: 사용자 질문
        
        Returns:
            의도 카테고리
        """
        system_prompt = """당신은 인터넷뱅킹 FAQ 시스템의 의도 분류기입니다.
사용자의 질문을 다음 카테고리 중 하나로 분류하세요:
- 로그인: 로그인, 비밀번호, 계정 관련
- 이체: 이체, 송금, 계좌이체 관련
- 인증서: 공동인증서, 공인인증서, 인증서 발급/갱신
- 오류코드: 오류 메시지, 에러 코드 해석
- 보안: 보안카드, OTP, 보안 설정
- 수수료: 이체 수수료, 거래 수수료
- 한도: 이체 한도, 거래 한도
- 계좌등록: 계좌 등록, 자주쓰는계좌
- 기타: 위에 해당하지 않는 경우

카테고리명만 반환하세요. 예: 이체"""
        
        prompt = f"질문: {question}\n\n카테고리:"
        
        try:
            result = self.generate(prompt, system_instruction=system_prompt, temperature=0.3)
            # 결과 정제 (앞뒤 공백 제거, 첫 단어만 추출)
            category = result.strip().split()[0] if result.strip() else "기타"
            logger.info(f"의도 분류 결과: {category}")
            return category
        except Exception as e:
            logger.warning(f"의도 분류 실패: {e}, 기본값 '기타' 사용")
            return "기타"
    
    def rewrite_query(self, question: str, intent: str) -> str:
        """
        질의 재작성 (동의어/약어 확장)
        
        Args:
            question: 원본 질문
            intent: 의도 카테고리
        
        Returns:
            재작성된 질문
        """
        system_prompt = """당신은 검색 쿼리 재작성 전문가입니다.
사용자의 질문을 FAQ 검색에 최적화된 형태로 재작성하세요.

주의사항:
- 동의어/유사어 추가 (예: "공인인증서" → "공동인증서")
- 약어 확장 (예: "OTP" → "일회용 비밀번호")
- 오타 수정
- 검색에 유용한 키워드 추가

원본 질문의 의미는 유지하되, 검색 성능을 높이도록 재작성하세요.
재작성된 질문만 반환하세요."""
        
        prompt = f"의도: {intent}\n원본 질문: {question}\n\n재작성된 질문:"
        
        try:
            result = self.generate(prompt, system_instruction=system_prompt, temperature=0.5)
            rewritten = result.strip() if result.strip() else question
            logger.info(f"질의 재작성: '{question}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"질의 재작성 실패: {e}, 원본 사용")
            return question
    
    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Dict],
        user_context: Optional[str] = None
    ) -> Dict:
        """
        FAQ 기반 답변 생성
        
        Args:
            question: 사용자 질문
            retrieved_docs: 검색된 FAQ 문서들
            user_context: 사용자 컨텍스트
        
        Returns:
            답변 딕셔너리
        """
        system_prompt = """당신은 인터넷뱅킹 FAQ 전문 상담원입니다.

중요 규칙:
1. 반드시 제공된 FAQ 문서만을 근거로 답변하세요. 추측하지 마세요.
2. 답변은 짧고 명확하게, 단계형으로 작성하세요.
3. 반드시 출처(FAQ 제목, URL)를 명시하세요.
4. FAQ에 없는 내용은 "관련 FAQ를 찾지 못했습니다"라고 답변하세요.
5. 근거가 부족하면 confidence를 low로 설정하세요.

출력 형식 (JSON):
{
  "answer": "최종 답변 (3-5줄)",
  "steps": ["1단계", "2단계", ...],
  "citations": [{"title": "FAQ 제목", "url": "URL", "snippet": "발췌"}],
  "confidence": "high/medium/low",
  "followups": ["추가 질문1"]
}

반드시 유효한 JSON 형식으로만 응답하세요."""
        
        # 검색된 문서 포맷팅
        docs_text = ""
        for i, doc in enumerate(retrieved_docs, 1):
            docs_text += f"\n[FAQ {i}]\n"
            docs_text += f"제목: {doc.get('title', '')}\n"
            docs_text += f"내용: {doc.get('text', '')}\n"
            docs_text += f"URL: {doc.get('url', '')}\n"
        
        user_prompt = f"사용자 질문: {question}"
        
        if user_context:
            user_prompt += f"\n상황: {user_context}"
        
        user_prompt += f"\n\n관련 FAQ:\n{docs_text}\n\n위 FAQ를 근거로 JSON 형식 답변을 생성하세요."
        
        try:
            response = self.generate(
                user_prompt,
                system_instruction=system_prompt,
                temperature=0.7
            )
            
            logger.debug(f"Gemini 응답: {response[:200]}...")
            
            # JSON 파싱 시도
            import json
            try:
                # JSON 코드 블록 제거
                response_clean = response.strip()
                if response_clean.startswith("```json"):
                    response_clean = response_clean[7:]
                if response_clean.startswith("```"):
                    response_clean = response_clean[3:]
                if response_clean.endswith("```"):
                    response_clean = response_clean[:-3]
                response_clean = response_clean.strip()
                
                result = json.loads(response_clean)
                logger.info(f"답변 생성 성공: confidence={result.get('confidence', 'unknown')}")
                return result
            except json.JSONDecodeError as je:
                logger.warning(f"JSON 파싱 실패: {je}, 원본 응답 사용")
                # JSON 파싱 실패 시 기본 형식으로 반환
                return {
                    "answer": response,
                    "steps": [],
                    "citations": [],
                    "confidence": "medium",
                    "followups": []
                }
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "steps": [],
                "citations": [],
                "confidence": "low",
                "followups": []
            }
