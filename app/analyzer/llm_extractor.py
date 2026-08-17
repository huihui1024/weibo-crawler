"""
LLM实体提取器 - 用DeepSeek/OpenAI替代正则匹配提取股票、板块、情感
批量处理一个博主的所有微博，一次API调用提取所有实体
"""
import os
import json
import re
import hashlib
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import yaml
from openai import OpenAI


@dataclass
class ExtractedStock:
    name: str
    code: Optional[str] = None
    sentiment: str = "neutral"  # bullish/bearish/neutral
    signal: Optional[str] = None  # 买入/卖出/加仓/减仓/建仓/清仓
    confidence: float = 0.5
    chain_position: str = ""  # 上游/中游/下游/瓶颈节点
    chain_path: str = ""      # 产业链路径


@dataclass
class ExtractedSector:
    name: str
    sentiment: str = "neutral"  # bullish/bearish/neutral
    confidence: float = 0.5


@dataclass
class ExtractionResult:
    stocks: List[ExtractedStock] = field(default_factory=list)
    sectors: List[ExtractedSector] = field(default_factory=list)
    method: str = "regex"  # llm / regex
    error: str = ""


EXTRACTION_PROMPT = """你是一个专业的A股/港股/美股金融文本分析专家。

请分析以下微博文本，提取其中提到的股票、行业板块，以及博主对它们的情感倾向和交易信号。

【微博文本】
{posts_text}

请严格按照以下JSON格式输出（不要输出其他内容）：
{{
  "stocks": [
    {{"name": "股票简称", "code": "6位代码或null", "sentiment": "bullish或bearish或neutral", "signal": "买入/卖出/加仓/减仓/建仓/清仓/null", "confidence": 0.0到1.0, "chain_position": "上游/中游/下游/瓶颈节点/null", "chain_path": "产业链路径如GPU→互联→CPO/null"}}
  ],
  "sectors": [
    {{"name": "板块名称", "sentiment": "bullish或bearish或neutral", "confidence": 0.0到1.0}}
  ]
}}

提取规则：
1. 股票：提取博主明确提到的股票名称。如果提到简称（如"茅台"→"贵州茅台"），请还原为标准全称。如果能推断出6位代码则填code，否则填null。
2. 板块：提取博主讨论的行业/概念板块（如"光模块"、"半导体"、"煤炭"、"商业航天"、"白酒"等）。
3. sentiment判断：
   - bullish：博主看好、认为会涨、已买入/持有
   - bearish：博主看空、认为会跌、已卖出/清仓
   - neutral：仅讨论/提及，无明确看好看空
4. signal：只有博主明确表达了交易动作时才填，如"买了"、"加仓了"、"清仓了"。仅讨论不填。
5. confidence：0-1，表示对该判断的置信度。明确信号0.8-1.0，较明确0.6-0.8，模糊0.4-0.6。
6. 不要提取大盘指数（如"上证指数"、"沪深300"）作为股票。
7. chain_position：判断该股票在产业链中的位置。如果该股票处于产业链瓶颈节点（具有定价权、卡住下游脖子），标记为"瓶颈节点"。
8. chain_path：如果文本中隐含了产业链上下游关系，请还原完整路径，如"GPU→互联带宽→CPO→激光器散热"。"""


class LLMExtractor:
    """LLM实体提取器"""

    def __init__(self):
        self._client = None
        self._model = ""
        self._cache: Dict[str, ExtractionResult] = {}
        self._lock = threading.Lock()
        self._init_client()

    def _init_client(self):
        """初始化OpenAI兼容客户端（DeepSeek等）"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                llm_cfg = cfg.get('llm', {})
            else:
                llm_cfg = {}

            api_key = llm_cfg.get('api_key', '') or os.environ.get('DEEPSEEK_API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')
            base_url = llm_cfg.get('base_url', '') or os.environ.get('DEEPSEEK_BASE_URL', '') or os.environ.get('OPENAI_BASE_URL', '')
            model = llm_cfg.get('model', '') or os.environ.get('LLM_MODEL', '')

            if not api_key:
                print("LLM提取器: 未配置API Key，将使用正则回退")
                return

            if not base_url:
                base_url = "https://api.deepseek.com/v1"
            if not model:
                model = "deepseek-chat"

            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self._model = model
            print(f"LLM提取器已初始化: model={model}, base_url={base_url}")

        except Exception as e:
            print(f"LLM提取器初始化失败: {e}")

    @property
    def available(self) -> bool:
        return self._client is not None

    def _make_cache_key(self, posts: List[Dict]) -> str:
        """根据微博内容生成缓存key"""
        text = "".join(p.get('content', '')[:200] for p in posts)
        return hashlib.md5(text.encode()).hexdigest()

    def extract(self, posts: List[Dict], max_posts: int = 30) -> ExtractionResult:
        """
        从微博列表中提取股票和板块实体

        Args:
            posts: 微博列表，每条需有 'content' 和可选 'publish_time'
            max_posts: 最多处理的微博条数

        Returns:
            ExtractionResult
        """
        if not posts:
            return ExtractionResult(method="regex")

        # 过滤有效微博
        valid_posts = [p for p in posts if p.get('content', '').strip()]
        if not valid_posts:
            return ExtractionResult(method="regex")

        # 按时间排序，优先最近的
        valid_posts.sort(key=lambda p: p.get('publish_time', ''), reverse=True)
        valid_posts = valid_posts[:max_posts]

        # 检查缓存
        cache_key = self._make_cache_key(valid_posts)
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # 尝试LLM提取
        if self.available:
            result = self._extract_with_llm(valid_posts)
            if result.stocks or result.sectors:
                with self._lock:
                    self._cache[cache_key] = result
                return result
            # LLM返回空结果，可能是格式解析失败，用regex
            print("LLM提取返回空结果，回退到正则")

        # 正则回退
        result = self._extract_with_regex(valid_posts)
        with self._lock:
            self._cache[cache_key] = result
        return result

    def _extract_with_llm(self, posts: List[Dict]) -> ExtractionResult:
        """使用LLM提取实体"""
        # 构建微博文本
        posts_text = ""
        for i, post in enumerate(posts, 1):
            content = post.get('content', '').strip()
            time_str = post.get('publish_time', '')
            posts_text += f"\n--- 微博{i} [{time_str}] ---\n{content}\n"

        prompt = EXTRACTION_PROMPT.format(posts_text=posts_text)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "你是专业的金融文本分析专家，擅长从微博文本中提取股票、板块和交易信号。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                timeout=60.0,
            )

            raw = response.choices[0].message.content.strip()
            return self._parse_llm_response(raw)

        except Exception as e:
            print(f"LLM提取失败: {e}")
            return ExtractionResult(method="llm", error=str(e))

    def _parse_llm_response(self, raw: str) -> ExtractionResult:
        """解析LLM返回的JSON"""
        try:
            # 提取JSON块
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                return ExtractionResult(method="llm", error="no JSON found")

            data = json.loads(json_match.group())

            stocks = []
            for s in data.get('stocks', []):
                name = s.get('name', '').strip()
                if not name or len(name) < 2:
                    continue
                sentiment = s.get('sentiment', 'neutral')
                if sentiment not in ('bullish', 'bearish', 'neutral'):
                    sentiment = 'neutral'
                signal = s.get('signal')
                if signal and str(signal).lower() == 'null':
                    signal = None
                confidence = float(s.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))
                code = s.get('code')
                if code and str(code).lower() == 'null':
                    code = None

                stocks.append(ExtractedStock(
                    name=name,
                    code=code,
                    sentiment=sentiment,
                    signal=signal,
                    confidence=confidence,
                    chain_position=s.get('chain_position', ''),
                    chain_path=s.get('chain_path', ''),
                ))

            sectors = []
            for s in data.get('sectors', []):
                name = s.get('name', '').strip()
                if not name:
                    continue
                sentiment = s.get('sentiment', 'neutral')
                if sentiment not in ('bullish', 'bearish', 'neutral'):
                    sentiment = 'neutral'
                confidence = float(s.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))

                sectors.append(ExtractedSector(
                    name=name,
                    sentiment=sentiment,
                    confidence=confidence,
                ))

            return ExtractionResult(stocks=stocks, sectors=sectors, method="llm")

        except Exception as e:
            print(f"LLM响应解析失败: {e}")
            return ExtractionResult(method="llm", error=str(e))

    def _extract_with_regex(self, posts: List[Dict]) -> ExtractionResult:
        """正则回退提取（复用StockAnalyzer的逻辑）"""
        try:
            from .stock_analyzer import StockAnalyzer
            analyzer = StockAnalyzer(enable_llm=False)

            all_stocks = []
            stock_counter = {}
            sector_counter = {}

            for post in posts:
                content = post.get('content', '')
                if not content:
                    continue
                stocks = analyzer._extract_stocks(content)
                for s in stocks:
                    if s.startswith('[板块]'):
                        sector_name = s.replace('[板块]', '')
                        sector_counter[sector_name] = sector_counter.get(sector_name, 0) + 1
                    else:
                        all_stocks.append(s)
                        stock_counter[s] = stock_counter.get(s, 0) + 1

            # 转换为ExtractedStock
            extracted_stocks = []
            seen = set()
            for name in all_stocks:
                if name in seen:
                    continue
                seen.add(name)
                extracted_stocks.append(ExtractedStock(
                    name=name,
                    confidence=0.3,
                    sentiment="neutral",
                ))

            # 转换为ExtractedSector
            extracted_sectors = []
            for name, count in sorted(sector_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
                extracted_sectors.append(ExtractedSector(
                    name=name,
                    confidence=0.3,
                    sentiment="neutral",
                ))

            # 同时用INDUSTRY_KEYWORDS补充板块
            for post in posts:
                content = post.get('content', '')
                if not content:
                    continue
                for industry, keywords in analyzer.INDUSTRY_KEYWORDS.items():
                    if any(kw in content for kw in keywords):
                        if industry not in {s.name for s in extracted_sectors}:
                            extracted_sectors.append(ExtractedSector(
                                name=industry,
                                confidence=0.3,
                                sentiment="neutral",
                            ))

            return ExtractionResult(stocks=extracted_stocks, sectors=extracted_sectors, method="regex")

        except Exception as e:
            print(f"正则回退提取失败: {e}")
            return ExtractionResult(method="regex", error=str(e))
