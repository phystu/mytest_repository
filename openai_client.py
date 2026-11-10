"""OpenAI API 호출 전용 모듈 (기본 모델: gpt-5.5)"""
import os

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 논문 평가·집필용 모델 (.env의 OPENAI_MODEL로 바꿀 수 있음)
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5").strip() or "gpt-5.5"

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
        _client = OpenAI(api_key=api_key, timeout=180.0, max_retries=1)
    return _client


def chat(messages: list[dict], max_tokens: int = 16000) -> dict:
    """OpenAI Chat Completions 호출. {'ok': bool, 'content': str, 'error': str} 반환

    GPT-5 계열은 추론(reasoning) 모델이라 temperature를 지원하지 않고,
    출력 한도는 max_completion_tokens(추론 토큰 포함)로 지정합니다.
    """
    try:
        res = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return {"ok": True, "content": res.choices[0].message.content or "", "error": ""}
    except RuntimeError as e:
        return {"ok": False, "content": "", "error": str(e)}
    except AuthenticationError:
        return {"ok": False, "content": "", "error": "OpenAI API 키가 올바르지 않습니다. .env를 확인해 주세요."}
    except RateLimitError:
        return {"ok": False, "content": "", "error": "요청 한도 또는 크레딧을 초과했습니다. 잠시 후 다시 시도해 주세요."}
    except APIConnectionError:
        return {"ok": False, "content": "", "error": "OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인해 주세요."}
    except APIStatusError as e:
        return {"ok": False, "content": "", "error": f"OpenAI 오류 (HTTP {e.status_code}): {e.message}"}
