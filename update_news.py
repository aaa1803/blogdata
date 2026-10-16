import os
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
import zoneinfo
import google.generativeai as genai

# 1. 환경 변수 및 날짜 설정 (KST 기준)
KST = zoneinfo.ZoneInfo("Asia/Seoul")
today_str = datetime.now(KST).strftime("%Y-%m-%d")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is missing. Please set it in GitHub Secrets or environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# 2. 경제 뉴스 RSS 수집 (연합뉴스 공식 경제 RSS 및 Google 뉴스 백업)
RSS_URLS = [
    "https://www.yna.co.kr/rss/economy.xml",
    "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=ko&gl=KR&ceid=KR:ko"
]

articles = []
for rss_url in RSS_URLS:
    try:
        req = urllib.request.Request(
            rss_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        items = root.findall(".//item")
        for item in items[:15]:
            title = item.findtext("title", "")
            desc = item.findtext("description", "")
            clean_desc = re.sub('<[^<]+?>', '', desc)
            if title:
                articles.append(f"제목: {title}\n내용: {clean_desc.strip()}")
        if articles:
            print(f"RSS 피드 수집 성공: {rss_url} ({len(articles)}건)")
            break
    except Exception as e:
        print(f"RSS 피드 수집 오류 ({rss_url}): {e}")

if not articles:
    articles.append("주요 국내외 거시경제 및 반도체 밸류체인 뉴스 업데이트 진행 중")

news_context = "\n---\n".join(articles)

# 3. Gemini 시스템 프롬프트 및 JSON 스키마
system_instruction = f"""
당신은 최고 수준의 거시경제 데이터 애널리스트이자 구조화 데이터 처리 엔진입니다.
제공된 당일 경제 기사 요약 텍스트를 정밀 분석하여, 웹 프론트엔드 연동에 최적화된 단일 유효 JSON(Valid JSON) 객체로 변환하여 출력하십시오.

[엄격한 제약조건]
1. 순수 객관성 및 팩트 기반 (주관적인 가격/지수 상승·하락 단정 절대 금지).
2. 마크다운 태그(```json 등)를 붙이지 말고 순수 JSON 문자열만 출력하십시오.
3. 기각 조건(falsification_condition)에는 분석을 뒤집을 구체적 데이터 수치/선행 지표/임계값을 명시하십시오.
4. 오늘 날짜는 {today_str} 입니다.
5. 시장 지표의 경우 기사 및 최신 데이터 맥락에 맞춰 작성하되 수치 형식(코스피, 코스닥, 원/달러 환율, WTI 원유)을 준수하십시오.

[출력 스키마]
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
      "observed_fact": "확인된 공식 데이터, 수치, 발표 내용 요약",
      "transmission_pathway": "실물 경제 및 밸류체인 전반에 미치는 인과적 파급 경로",
      "falsification_condition": "이 가설 및 분석을 무효화할 수 있는 구체적인 선행 지표 또는 정책 변화 수치",
      "analyst_tip": "모니터링 시 주목할 1줄 핵심 팁"
    }}
  ]
}}
(top_news 배열에는 정확히 5개의 항목이 포함되어야 합니다.)
"""

# 4. 모델 호출 및 JSON 추출 (모델 가용성 대비 폴백 지원)
prompt = f"다음 기사 목록을 분석하여 위 스키마의 유효한 JSON을 생성하세요:\n\n{news_context}"

candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
response = None
for model_name in candidate_models:
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        response = model.generate_content(
            contents=[prompt],
            generation_config={
                "response_mime_type": "application/json"
            }
        )
        print(f"Gemini 모델 호출 성공: {model_name}")
        break
    except Exception as e:
        print(f"모델 {model_name} 호출 실패, 다음 모델로 시도합니다: {e}")

if not response or not response.text:
    raise RuntimeError("Gemini API로부터 유효한 응답을 받지 못했습니다.")

result_text = response.text.strip()
if result_text.startswith("```json"):
    result_text = result_text[7:]
elif result_text.startswith("```"):
    result_text = result_text[3:]
if result_text.endswith("```"):
    result_text = result_text[:-3]
result_text = result_text.strip()

# JSON 검증 후 저장
parsed_json = json.loads(result_text)

# 필수 필드 보장
if "report_date" not in parsed_json:
    parsed_json["report_date"] = today_str

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(parsed_json, f, ensure_ascii=False, indent=2)

print(f"[{today_str}] data.json 파일이 정상적으로 갱신되었습니다.")
