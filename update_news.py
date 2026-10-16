import os
import json
import re
import time
from datetime import datetime
import zoneinfo
import urllib.request
import xml.etree.ElementTree as ET
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

# 1. 환경 변수 및 날짜 설정 (KST)
KST = zoneinfo.ZoneInfo("Asia/Seoul")
today_str = datetime.now(KST).strftime("%Y-%m-%d")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")

client = genai.Client(api_key=GEMINI_API_KEY)

# 2. 경제 뉴스 RSS 수집
RSS_URL = "https://www.yna.co.kr/rss/economy.xml"
req = urllib.request.Request(
    RSS_URL,
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
)

articles = []
try:
    with urllib.request.urlopen(req, timeout=10) as response:
        xml_data = response.read()
    root = ET.fromstring(xml_data)
    for item in root.findall(".//item")[:15]:
        title = item.findtext("title", "")
        desc = item.findtext("description", "")
        clean_desc = re.sub('<[^<]+?>', '', desc)
        articles.append(f"제목: {title}\n내용: {clean_desc}")
    print(f"RSS 피드 수집 성공: {RSS_URL} ({len(articles)}건)")
except Exception as e:
    print(f"RSS 수집 오류: {e}")
    articles.append("주요 경제 뉴스 수집 중 통신 오류 발생")

news_context = "\n---\n".join(articles)

# 3. 프롬프트 지시문
system_prompt = f"""당신은 최고 수준의 거시경제 데이터 애널리스트이자 구조화 데이터 처리 엔진입니다.
제공된 당일 경제 기사 요약 텍스트를 분석하여 웹 프론트엔드 연동용 JSON을 생성하십시오.

[엄격한 제약조건]
1. 순수 객관성 및 팩트 기반 작성 (주관적 단정 금지).
2. 마크다운 기호 없이 오직 유효한 JSON 형식으로만 응답할 것.
3. 기각 조건(falsification_condition)에는 구체적인 데이터 수치와 선행 지표를 명시할 것.
4. 오늘 날짜는 {today_str} 입니다.
5. top_news는 반드시 5개를 작성할 것.

[출력 JSON 구조]
{{
  "report_date": "{today_str}",
  "macro_core_takeaways": [
    "핵심 요약 1",
    "핵심 요약 2",
    "핵심 요약 3"
  ],
  "market_indicators": [
    {{ "name": "코스피", "value": "0,000.00", "change": "+0.00", "change_rate": "+0.00%" }},
    {{ "name": "코스닥", "value": "000.00", "change": "-0.00", "change_rate": "-0.00%" }},
    {{ "name": "원/달러 환율", "value": "0,000.00", "change": "-0.00", "change_rate": "-0.00%" }},
    {{ "name": "WTI 원유", "value": "00.00", "change": "-0.00", "change_rate": "-0.00%" }}
  ],
  "top_news": [
    {{
      "id": 1,
      "title": "기사 헤드라인 요약",
      "sector": "관련 산업군",
      "observed_fact": "확인된 공식 데이터, 수치, 통계 요약",
      "transmission_pathway": "실물 경제 및 밸류체인 전반에 미치는 인과적 파급 경로",
      "falsification_condition": "이 분석을 무효화할 수 있는 구체적인 선행 지표 또는 정책 변화 수치",
      "analyst_tip": "모니터링 시 주목할 1줄 핵심 체크포인트"
    }}
  ]
}}
"""

user_prompt = f"다음 기사 목록을 분석하여 위 스키마에 맞는 유효한 JSON을 생성하세요:\n\n{news_context}"

# 4. 503 과부하 방지: 모델 다중화(Fallback) 및 재시도 로직
target_models = ['gemini-3.8-flash', 'gemini-2.5-flash', 'gemini-2.5-pro']
response = None

for model_name in target_models:
    for attempt in range(1, 4):
        try:
            print(f"[{model_name}] 호출 시도 중... (시도 {attempt}/3)")
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            if response and response.text:
                break
        except (ServerError, APIError) as e:
            print(f"[{model_name}] 서버 응답 오류({e.code if hasattr(e, 'code') else 'Error'}): {e.message if hasattr(e, 'message') else e}")
            if attempt < 3:
                wait_time = attempt * 5  # 5초, 10초 대기 후 재시도
                print(f"{wait_time}초 후 재시도합니다...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"[{model_name}] 일반 예외 발생: {e}")
            break
            
    if response and response.text:
        print(f"[{model_name}] 모델을 통해 성공적으로 데이터를 받았습니다.")
        break

if not response or not response.text:
    raise RuntimeError("모든 대체 모델에서 응답을 받지 못했습니다.")

result_text = response.text.strip()
if result_text.startswith("```json"):
    result_text = result_text[7:]
if result_text.endswith("```"):
    result_text = result_text[:-3]
result_text = result_text.strip()

parsed_json = json.loads(result_text)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(parsed_json, f, ensure_ascii=False, indent=2)

print(f"[{today_str}] data.json 파일이 정상적으로 갱신되었습니다.")
