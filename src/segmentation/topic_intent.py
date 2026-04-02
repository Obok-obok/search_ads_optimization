from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

_GENERIC_REMOVE_TERMS = [
    '보험료비교', '보험비교', '보험료', '보험', '가입조건', '가입', '보장', '문의', '상담',
    '추천', '비교', '순위', '가격', '비용', '청구', '지급', '서류', '가능', '조건',
]

_INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ('price', ('보험료', '비용', '가격', '얼마', '견적')),
    ('compare', ('비교', '추천', '순위', 'best', 'top')),
    ('eligibility', ('조건', '가입', '가능', '연령', '나이', '대상')),
    ('claim', ('청구', '지급', '보상', '서류')),
]

_KOREAN_RE = re.compile(r'[가-힣]+')
_NON_ALNUM_RE = re.compile(r'[^0-9A-Za-z가-힣]+')


def _normalize_keyword(keyword: str) -> str:
    text = _NON_ALNUM_RE.sub('', str(keyword or '').strip())
    return text


def infer_intent(keyword: str) -> str:
    normalized = _normalize_keyword(keyword)
    if not normalized:
        return 'general'
    for intent, patterns in _INTENT_RULES:
        if any(p in normalized for p in patterns):
            return intent
    return 'general'


def infer_topic(keyword: str) -> str:
    normalized = _normalize_keyword(keyword)
    if not normalized:
        return 'unknown'

    cleaned = normalized
    for term in sorted(_GENERIC_REMOVE_TERMS, key=len, reverse=True):
        cleaned = cleaned.replace(term, '')

    cleaned = _normalize_keyword(cleaned)
    if not cleaned:
        return 'unknown'

    korean_chunks = _KOREAN_RE.findall(cleaned)
    if korean_chunks:
        primary = max(korean_chunks, key=len)
        return primary[:2] if len(primary) >= 2 else primary

    return cleaned[:4].lower() if len(cleaned) >= 4 else cleaned.lower()


def build_topic_intent_frame(keywords: Iterable[str], *, segment: str | None = None) -> pd.DataFrame:
    series = pd.Series(list(keywords), dtype='object').dropna().astype(str).str.strip()
    if series.empty:
        return pd.DataFrame(columns=['keyword', 'topic', 'intent', 'routing_key'])
    series = series.loc[series.ne('')].drop_duplicates().reset_index(drop=True)
    out = pd.DataFrame({'keyword': series})
    out['topic'] = out['keyword'].map(infer_topic)
    out['intent'] = out['keyword'].map(infer_intent)
    prefix = f"{segment}__" if segment else ''
    out['routing_key'] = prefix + out['topic'].astype(str) + '__' + out['intent'].astype(str)
    return out
