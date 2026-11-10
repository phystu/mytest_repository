"""Dr. 리뷰어 - 논문 평가·고도화 챗봇 (Flask)"""
import os

from flask import Flask, jsonify, request, send_from_directory

import openai_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 논문 본문 최대 글자 수 (초과 시 앞부분만 사용) / 대화 기록 최대 유지 개수
MAX_PAPER_CHARS = 80000
MAX_HISTORY = 20
MAX_MEMORY_CHARS = 4000   # 대화 기억(요약) 최대 길이
MAX_DRAFT_CHARS = 40000   # 빌드업 초안 최대 길이


def _clean_history(raw: list) -> list[dict]:
    out = []
    for m in raw[-MAX_HISTORY:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and "content" in m:
            out.append({"role": m["role"], "content": str(m["content"])})
    return out

app = Flask(__name__, static_folder="assets", static_url_path="/assets")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 요청 본문 최대 10MB

SYSTEM_PROMPT = """당신은 "Dr. 리뷰어"입니다. 20년 경력의 저명한 연구자이자 국제 학술지(SCI/SSCI/KCI) 논문 심사위원·편집위원입니다.
수백 편의 논문을 심사하고 지도한 경험이 있으며, 날카롭지만 건설적이고 정중한 태도로 피드백합니다.

[역할]
1. 사용자가 업로드한 논문을 엄밀하게 읽고 평가·분석합니다.
2. 논문의 완성도를 한 단계 더 높일 수 있도록 구체적인 수정안과 다시 쓴 문장/단락을 제시합니다.

[논문 평가 시 기본 형식] (사용자가 "평가/리뷰/심사"를 요청하면 아래 구조를 따르세요)
## 1. 논문 요약 (3~5문장)
## 2. 종합 평가 점수 (각 항목 10점 만점, 한 줄 근거 포함)
- 독창성/참신성, 연구 질문의 명확성, 이론적 배경·선행연구, 연구방법론의 타당성, 데이터·결과 분석, 논의·시사점, 논리 전개·구성, 문장·표현·형식
## 3. 강점
## 4. 주요 문제점 (Major Concerns) - 심각도 순, 논문 내 해당 위치(섹션/문장 인용)를 명시
## 5. 사소한 문제점 (Minor Concerns)
## 6. 개선 로드맵 - 우선순위별 구체적 실행 방안
## 7. 심사 결과 권고 (Accept / Minor Revision / Major Revision / Reject) 및 그 이유

[논문을 더 높은 수준으로 다시 써 줄 때]
- 원문의 연구 내용과 사실을 바꾸지 말고, 논리 구조·학술적 어조·명확성·간결성·설득력을 높입니다.
- 가능하면 "원문 → 개선안" 형태로 보여주고, 무엇을 왜 바꿨는지 짧게 설명합니다.
- 초록, 서론(연구 공백·기여점), 방법, 결과, 논의, 결론 등 요청된 부분만 정확히 다룹니다.

[원칙]
- 논문 본문에 없는 데이터, 수치, 참고문헌을 절대 지어내지 마세요. 필요한 근거/자료가 부족하면 "[저자가 보완할 내용: ...]"처럼 표시하세요.
- 특정 참고문헌을 추천할 때는 실제로 존재함이 확실한 경우에만 언급하고, 불확실하면 검색 키워드를 제안하세요.
- 칭찬은 구체적으로, 비판은 근거와 대안과 함께 제시하세요.
- 사용자가 사용하는 언어로 답하세요(기본은 한국어). 전문 용어는 필요하면 영문을 병기하세요.
- 마크다운으로 읽기 쉽게 작성하세요.
- 논문이 아직 업로드되지 않았다면 논문 파일(PDF/DOCX/TXT)을 먼저 올려 달라고 안내하고, 일반적인 논문 작성 질문에는 답변하세요.

[논문 빌드업(누적 개선) 원칙]
- "[대화 기억]"이 제공되면 이전에 합의한 연구 목표, 투고 대상 저널, 용어·문체 규칙, 확정한 수정 방향을 계속 유지하세요.
- "[빌드업 중인 논문 초안]"이 제공되면 그것이 현재까지 확정된 최신 원고입니다. 새 답변은 이 초안과 모순되지 않게 이어 쓰고, 용어와 문체를 일관되게 맞추세요.
- 다시 쓴 본문은 사용자가 초안에 바로 붙일 수 있도록 섹션 단위(예: '## 초록', '## 1. 서론')로 깔끔하게 제시하고, 설명·코멘트는 본문과 구분해서 쓰세요."""

MEMORY_PROMPT = """당신은 논문 공동 작업 기록 담당자입니다. 아래의 '기존 기억'과 '새 대화'를 합쳐, 이후 논문 빌드업 작업에 꼭 필요한 정보만 남긴 새 '대화 기억'을 한국어로 작성하세요.

반드시 포함할 항목(해당 없으면 생략):
- 연구 주제·목표, 투고 대상 저널/학회, 연구 질문·가설·방법·핵심 결과
- 심사 평가에서 지적된 핵심 문제와 현재 해결 여부(해결/미해결)
- 사용자가 확정했거나 요청한 수정 방향, 문체·용어 규칙, 하지 말라고 한 것
- 이미 다시 쓴 섹션과 앞으로 할 일(TODO)

규칙: 새로운 사실을 지어내지 말 것. 불필요한 인사·반복은 제외. 항목별 글머리표로 간결하게(최대 1,500자)."""


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list) or not data["messages"]:
        return jsonify(ok=False, error="잘못된 요청입니다."), 400

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 논문 본문 삽입
    truncated = False
    paper = data.get("paper")
    if isinstance(paper, dict) and isinstance(paper.get("text"), str) and paper["text"].strip():
        text = paper["text"]
        title = paper.get("title") if isinstance(paper.get("title"), str) else "업로드된 논문"
        if len(text) > MAX_PAPER_CHARS:
            text = text[:MAX_PAPER_CHARS]
            truncated = True
        note = (
            f"(주의: 분량이 길어 앞부분 {MAX_PAPER_CHARS}자만 제공되었습니다. 평가 시 이 점을 언급하세요.)\n"
            if truncated
            else ""
        )
        messages.append(
            {
                "role": "system",
                "content": f"[사용자가 업로드한 논문: {title}]\n{note}"
                f"----- 논문 본문 시작 -----\n{text}\n----- 논문 본문 끝 -----",
            }
        )

    # 대화 기억(요약) 삽입
    memory = data.get("memory")
    if isinstance(memory, str) and memory.strip():
        messages.append({"role": "system", "content": f"[대화 기억]\n{memory.strip()[:MAX_MEMORY_CHARS]}"})

    # 빌드업 중인 초안 삽입
    draft = data.get("draft")
    if isinstance(draft, str) and draft.strip():
        draft = draft.strip()
        if len(draft) > MAX_DRAFT_CHARS:
            draft = draft[-MAX_DRAFT_CHARS:]  # 가장 최근 부분을 우선 유지
        messages.append(
            {
                "role": "system",
                "content": f"[빌드업 중인 논문 초안]\n----- 초안 시작 -----\n{draft}\n----- 초안 끝 -----",
            }
        )

    # 대화 기록 (최근 N개만)
    messages.extend(_clean_history(data["messages"]))

    result = openai_client.chat(messages)
    if not result["ok"]:
        return jsonify(ok=False, error=result["error"]), 502

    return jsonify(ok=True, reply=result["content"], truncated=truncated)


@app.post("/api/summarize")
def summarize():
    """오래된 대화를 '대화 기억'으로 요약해 컨텍스트를 압축한다."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return jsonify(ok=False, error="잘못된 요청입니다."), 400

    old_memory = data.get("memory") if isinstance(data.get("memory"), str) else ""
    lines = []
    for m in data["messages"]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and "content" in m:
            who = "사용자" if m["role"] == "user" else "Dr. 리뷰어"
            lines.append(f"{who}: {str(m['content'])[:6000]}")
    if not lines:
        return jsonify(ok=True, memory=old_memory)

    result = openai_client.chat(
        [
            {"role": "system", "content": MEMORY_PROMPT},
            {
                "role": "user",
                "content": f"[기존 기억]\n{old_memory or '(없음)'}\n\n[새 대화]\n" + "\n\n".join(lines),
            },
        ],
        max_tokens=8000,
    )
    if not result["ok"]:
        return jsonify(ok=False, error=result["error"]), 502
    return jsonify(ok=True, memory=result["content"].strip()[:MAX_MEMORY_CHARS])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
