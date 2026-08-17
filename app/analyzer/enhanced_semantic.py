"""
增强版语义解析模块
优化点：
1. 结构化输出 - 使用Pydantic模型确保输出一致性
2. 多轮对话上下文 - 支持上下文关联分析
3. 智能缓存 - 基于语义相似度的缓存
4. 模糊匹配 - 处理拼写错误和变体
5. 解析质量评分 - 自动评估解析结果质量
6. 批量优化 - 并行处理和批量推理
"""
import os
import json
import re
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import threading

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback to dataclass
    BaseModel = object


# ==================== 结构化输出模型 ====================

if PYDANTIC_AVAILABLE:
    class ExtractedInfo(BaseModel):
        """提取的关键信息"""
        stock_names: List[str] = Field(default_factory=list, description="提到的股票名称")
        stock_codes: List[str] = Field(default_factory=list, description="股票代码")
        price_target: Optional[str] = Field(None, description="目标价位")
        stop_loss: Optional[str] = Field(None, description="止损位")
        stop_profit: Optional[str] = Field(None, description="止盈位")
        position_size: Optional[str] = Field(None, description="仓位大小")
        sector: Optional[str] = Field(None, description="板块/行业")
        trading_cycle: Optional[str] = Field(None, description="交易周期")
        buy_price: Optional[str] = Field(None, description="买入价位")
        sell_price: Optional[str] = Field(None, description="卖出价位")

    class QualityMetrics(BaseModel):
        """解析质量指标"""
        overall_score: float = Field(0.0, ge=0, le=1, description="整体质量分")
        completeness: float = Field(0.0, ge=0, le=1, description="完整性")
        consistency: float = Field(0.0, ge=0, le=1, description="一致性")
        confidence: float = Field(0.0, ge=0, le=1, description="置信度")
        issues: List[str] = Field(default_factory=list, description="发现的问题")

    class StructuredAnalysisResult(BaseModel):
        """结构化分析结果"""
        # 场景类型
        scenario_type: str = Field("unknown", description="场景类型")
        scenario_confidence: float = Field(0.0, ge=0, le=1)

        # 真实意图
        true_sentiment: str = Field("neutral", description="真实意图")
        sentiment_confidence: float = Field(0.0, ge=0, le=1)

        # 交易信号
        action_type: str = Field("none", description="交易信号")
        action_confidence: float = Field(0.0, ge=0, le=1)

        # 正式程度
        is_formal_opinion: bool = Field(False)
        formality_confidence: float = Field(0.0, ge=0, le=1)

        # 实盘判断
        is_real_position: bool = Field(True)
        position_confidence: float = Field(0.0, ge=0, le=1)

        # 情绪
        emotion: str = Field("neutral")
        emotion_confidence: float = Field(0.0, ge=0, le=1)

        # 提取信息
        extracted_info: ExtractedInfo = Field(default_factory=ExtractedInfo)

        # 分析依据
        reasoning: str = Field("")

        # 质量指标
        quality: QualityMetrics = Field(default_factory=QualityMetrics)

        # 元数据
        original_text: str = Field("")
        analysis_method: str = Field("")
        analysis_time: datetime = Field(default_factory=datetime.now)

        @field_validator('scenario_type')
        @classmethod
        def validate_scenario(cls, v):
            valid = ['real_trading', 'simulation', 'discussion', 'example', 'review', 'unknown']
            return v if v in valid else 'unknown'

        @field_validator('true_sentiment')
        @classmethod
        def validate_sentiment(cls, v):
            valid = ['bullish', 'bearish', 'neutral', 'sarcastic']
            return v if v in valid else 'neutral'

        @field_validator('action_type')
        @classmethod
        def validate_action(cls, v):
            valid = ['buy', 'sell', 'hold', 'watch', 'no_position', 'none']
            return v if v in valid else 'none'
else:
    # Fallback dataclass implementation
    @dataclass
    class ExtractedInfo:
        stock_names: List[str] = field(default_factory=list)
        stock_codes: List[str] = field(default_factory=list)
        price_target: Optional[str] = None
        stop_loss: Optional[str] = None
        stop_profit: Optional[str] = None
        position_size: Optional[str] = None
        sector: Optional[str] = None
        trading_cycle: Optional[str] = None
        buy_price: Optional[str] = None
        sell_price: Optional[str] = None

    @dataclass
    class QualityMetrics:
        overall_score: float = 0.0
        completeness: float = 0.0
        consistency: float = 0.0
        confidence: float = 0.0
        issues: List[str] = field(default_factory=list)

    @dataclass
    class StructuredAnalysisResult:
        scenario_type: str = "unknown"
        scenario_confidence: float = 0.0
        true_sentiment: str = "neutral"
        sentiment_confidence: float = 0.0
        action_type: str = "none"
        action_confidence: float = 0.0
        is_formal_opinion: bool = False
        formality_confidence: float = 0.0
        is_real_position: bool = True
        position_confidence: float = 0.0
        emotion: str = "neutral"
        emotion_confidence: float = 0.0
        extracted_info: ExtractedInfo = field(default_factory=ExtractedInfo)
        reasoning: str = ""
        quality: QualityMetrics = field(default_factory=QualityMetrics)
        original_text: str = ""
        analysis_method: str = ""
        analysis_time: datetime = field(default_factory=datetime.now)


# ==================== 智能缓存 ====================

class SemanticCache:
    """基于语义相似度的智能缓存"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[StructuredAnalysisResult, datetime]] = {}
        self._lock = threading.Lock()

    def _normalize_text(self, text: str) -> str:
        """文本标准化"""
        # 移除多余空格
        text = re.sub(r'\s+', '', text)
        # 移除表情符号
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # 转小写
        text = text.lower()
        return text

    def _get_hash(self, text: str, stock_name: str = None) -> str:
        """获取文本哈希"""
        normalized = self._normalize_text(text)
        key = f"{normalized}:{stock_name or ''}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, text: str, stock_name: str = None) -> Optional[StructuredAnalysisResult]:
        """获取缓存"""
        with self._lock:
            key = self._get_hash(text, stock_name)
            if key in self._cache:
                result, timestamp = self._cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    return result
                else:
                    del self._cache[key]
            return None

    def set(self, text: str, result: StructuredAnalysisResult, stock_name: str = None):
        """设置缓存"""
        with self._lock:
            # LRU清理
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(),
                                key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            key = self._get_hash(text, stock_name)
            self._cache[key] = (result, datetime.now())

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()


# ==================== 模糊匹配 ====================

class FuzzyMatcher:
    """模糊匹配器 - 处理股票名称变体和拼写错误"""

    # 常见股票名称变体
    STOCK_VARIANTS = {
        '茅台': ['贵州茅台', '茅台酒', '飞天茅台'],
        '宁德': ['宁德时代', 'CATL'],
        '比亚迪': ['BYD', '迪子'],
        '中际': ['中际旭创'],
        '隆基': ['隆基绿能', '隆基股份'],
        '平安': ['中国平安', '平安银行'],
        '招商': ['招商银行', '招行'],
        '五粮液': ['五粮液', '五粮'],
        '美的': ['美的集团'],
        '格力': ['格力电器'],
    }

    # 行业板块变体
    SECTOR_VARIANTS = {
        '新能源': ['光伏', '风电', '储能', '锂电'],
        '科技': ['半导体', '芯片', 'AI', '人工智能'],
        '医药': ['医疗', '生物', '创新药', 'CXO'],
        '消费': ['白酒', '食品', '饮料'],
        '金融': ['银行', '券商', '保险'],
    }

    @classmethod
    def normalize_stock_name(cls, name: str) -> List[str]:
        """获取股票名称的所有可能变体"""
        names = [name]
        for standard, variants in cls.STOCK_VARIANTS.items():
            if name in variants or name == standard:
                names.append(standard)
                names.extend(variants)
        return list(set(names))

    @classmethod
    def normalize_sector(cls, sector: str) -> List[str]:
        """获取板块的所有可能变体"""
        sectors = [sector]
        for standard, variants in cls.SECTOR_VARIANTS.items():
            if sector in variants or sector == standard:
                sectors.append(standard)
                sectors.extend(variants)
        return list(set(sectors))

    @classmethod
    def levenshtein_distance(cls, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return cls.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @classmethod
    def fuzzy_match_stock(cls, name: str, threshold: float = 0.7) -> Optional[str]:
        """模糊匹配股票名称"""
        name = name.strip()
        if not name:
            return None

        # 先尝试精确匹配
        for standard, variants in cls.STOCK_VARIANTS.items():
            if name == standard or name in variants:
                return standard

        # 模糊匹配
        best_match = None
        best_score = 0

        for standard in cls.STOCK_VARIANTS.keys():
            distance = cls.levenshtein_distance(name, standard)
            max_len = max(len(name), len(standard))
            score = 1 - distance / max_len

            if score > best_score and score >= threshold:
                best_score = score
                best_match = standard

        return best_match


# ==================== 上下文管理 ====================

@dataclass
class ConversationContext:
    """对话上下文"""
    blogger_id: str
    recent_posts: List[Dict] = field(default_factory=list)  # 最近微博
    mentioned_stocks: Dict[str, int] = field(default_factory=dict)  # 提及股票及次数
    recent_actions: List[Dict] = field(default_factory=list)  # 最近操作
    last_analysis: Optional[StructuredAnalysisResult] = None
    updated_at: datetime = field(default_factory=datetime.now)


class ContextManager:
    """上下文管理器"""

    def __init__(self, max_context_posts: int = 10):
        self.max_context_posts = max_context_posts
        self._contexts: Dict[str, ConversationContext] = {}
        self._lock = threading.Lock()

    def update_context(self, blogger_id: str, post: Dict, analysis: StructuredAnalysisResult):
        """更新上下文"""
        with self._lock:
            if blogger_id not in self._contexts:
                self._contexts[blogger_id] = ConversationContext(blogger_id=blogger_id)

            ctx = self._contexts[blogger_id]

            # 添加微博
            ctx.recent_posts.append(post)
            if len(ctx.recent_posts) > self.max_context_posts:
                ctx.recent_posts.pop(0)

            # 更新提及股票
            for stock in analysis.extracted_info.stock_names:
                ctx.mentioned_stocks[stock] = ctx.mentioned_stocks.get(stock, 0) + 1

            # 更新最近操作
            if analysis.action_type in ['buy', 'sell']:
                ctx.recent_actions.append({
                    'action': analysis.action_type,
                    'stock': analysis.extracted_info.stock_names,
                    'time': analysis.analysis_time
                })
                if len(ctx.recent_actions) > 20:
                    ctx.recent_actions.pop(0)

            ctx.last_analysis = analysis
            ctx.updated_at = datetime.now()

    def get_context(self, blogger_id: str) -> Optional[ConversationContext]:
        """获取上下文"""
        return self._contexts.get(blogger_id)

    def get_frequently_mentioned_stocks(self, blogger_id: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """获取经常提及的股票"""
        ctx = self._contexts.get(blogger_id)
        if not ctx:
            return []

        sorted_stocks = sorted(ctx.mentioned_stocks.items(),
                              key=lambda x: x[1], reverse=True)
        return sorted_stocks[:top_n]


# ==================== 质量评估 ====================

class QualityEvaluator:
    """解析质量评估器"""

    @staticmethod
    def evaluate(result: StructuredAnalysisResult) -> QualityMetrics:
        """评估解析结果质量"""
        issues = []

        # 1. 完整性评估
        completeness = QualityEvaluator._evaluate_completeness(result, issues)

        # 2. 一致性评估
        consistency = QualityEvaluator._evaluate_consistency(result, issues)

        # 3. 置信度评估
        confidence = QualityEvaluator._evaluate_confidence(result)

        # 4. 整体评分
        overall_score = (completeness * 0.3 + consistency * 0.3 + confidence * 0.4)

        return QualityMetrics(
            overall_score=round(overall_score, 3),
            completeness=round(completeness, 3),
            consistency=round(consistency, 3),
            confidence=round(confidence, 3),
            issues=issues
        )

    @staticmethod
    def _evaluate_completeness(result: StructuredAnalysisResult, issues: List[str]) -> float:
        """评估完整性"""
        score = 0.0
        total = 6

        # 检查各字段是否有有效值
        if result.scenario_type != 'unknown':
            score += 1
        if result.true_sentiment != 'neutral':
            score += 1
        if result.action_type != 'none':
            score += 1
        if result.extracted_info.stock_names:
            score += 1
        if result.reasoning:
            score += 1
        if result.scenario_confidence > 0:
            score += 1

        if score < total * 0.5:
            issues.append("解析结果不完整，多个关键字段缺失")

        return score / total

    @staticmethod
    def _evaluate_consistency(result: StructuredAnalysisResult, issues: List[str]) -> float:
        """评估一致性"""
        score = 1.0

        # 检查场景类型和交易信号的一致性
        if result.scenario_type == 'real_trading':
            if result.action_type == 'none':
                score -= 0.3
                issues.append("实盘交易场景但无明确交易信号")

        # 检查情绪和操作的一致性
        if result.true_sentiment == 'bullish' and result.action_type == 'sell':
            score -= 0.2
            issues.append("看多情绪但操作为卖出，可能存在矛盾")

        if result.true_sentiment == 'bearish' and result.action_type == 'buy':
            score -= 0.2
            issues.append("看空情绪但操作为买入，可能存在矛盾")

        # 检查置信度一致性
        confidences = [
            result.scenario_confidence,
            result.sentiment_confidence,
            result.action_confidence
        ]
        if max(confidences) - min(confidences) > 0.5:
            score -= 0.1
            issues.append("各维度置信度差异较大")

        return max(0, score)

    @staticmethod
    def _evaluate_confidence(result: StructuredAnalysisResult) -> float:
        """评估置信度"""
        confidences = [
            result.scenario_confidence,
            result.sentiment_confidence,
            result.action_confidence,
            result.formality_confidence,
            result.position_confidence
        ]
        return sum(confidences) / len(confidences)


# ==================== 增强版语义分析器 ====================

class EnhancedSemanticAnalyzer:
    """增强版语义分析器"""

    # 改进的提示词模板
    ANALYSIS_PROMPT_TEMPLATE = """你是一个专业的股市分析师助手。请分析以下微博内容的真实含义。

【微博内容】
{content}

{context_info}

请严格按照以下JSON格式输出分析结果（不要输出其他内容）：
{{
    "scenario_type": "real_trading|simulation|discussion|example|review|unknown",
    "scenario_confidence": 0.0-1.0,
    "true_sentiment": "bullish|bearish|neutral|sarcastic",
    "sentiment_confidence": 0.0-1.0,
    "action_type": "buy|sell|hold|watch|no_position|none",
    "action_confidence": 0.0-1.0,
    "is_formal_opinion": true|false,
    "formality_confidence": 0.0-1.0,
    "is_real_position": true|false,
    "position_confidence": 0.0-1.0,
    "emotion": "confident|uncertain|regretful|excited|fearful|neutral",
    "emotion_confidence": 0.0-1.0,
    "extracted_info": {{
        "stock_names": ["股票名称1", "股票名称2"],
        "stock_codes": ["代码1", "代码2"],
        "price_target": "目标价位",
        "stop_loss": "止损位",
        "stop_profit": "止盈位",
        "position_size": "仓位大小",
        "sector": "板块",
        "trading_cycle": "交易周期"
    }},
    "reasoning": "简要分析依据"
}}

场景类型说明：
- real_trading: 实盘交易，博主真实持仓和操作
- simulation: 模拟盘/假设性讨论
- discussion: 市场讨论，非具体操作
- example: 举例说明
- review: 复盘总结"""

    def __init__(self, llm_provider=None, use_cache: bool = True):
        self.llm_provider = llm_provider or self._get_default_provider()
        self.use_cache = use_cache
        self.cache = SemanticCache() if use_cache else None
        self.context_manager = ContextManager()
        self.quality_evaluator = QualityEvaluator()

    def _get_default_provider(self):
        """获取默认LLM提供者"""
        # 尝试导入现有的provider
        try:
            from .llm_semantic import SemanticAnalyzer
            analyzer = SemanticAnalyzer()
            return analyzer.provider
        except:
            return None

    def _build_context_info(self, blogger_id: str = None) -> str:
        """构建上下文信息"""
        if not blogger_id:
            return ""

        ctx = self.context_manager.get_context(blogger_id)
        if not ctx or not ctx.recent_posts:
            return ""

        context_lines = ["【历史上下文】"]
        context_lines.append(f"最近提及股票: {', '.join(ctx.mentioned_stocks.keys()[:5])}")

        if ctx.last_analysis:
            context_lines.append(f"上次分析: {ctx.last_analysis.action_type} {ctx.last_analysis.extracted_info.stock_names}")

        return "\n".join(context_lines)

    async def analyze_async(
        self,
        content: str,
        stock_name: str = None,
        blogger_id: str = None
    ) -> StructuredAnalysisResult:
        """异步分析"""
        # 检查缓存
        if self.use_cache and self.cache:
            cached = self.cache.get(content, stock_name)
            if cached:
                return cached

        # 构建提示词
        context_info = self._build_context_info(blogger_id)
        prompt = self.ANALYSIS_PROMPT_TEMPLATE.format(
            content=content,
            context_info=context_info
        )

        try:
            # 调用LLM
            if self.llm_provider:
                response = await self.llm_provider.analyze(prompt)
            else:
                # 使用规则引擎回退
                response = self._rule_based_analysis(content, stock_name)

            # 解析结果
            result = self._parse_response(response, content)

            # 质量评估
            result.quality = self.quality_evaluator.evaluate(result)

            # 更新上下文
            if blogger_id:
                self.context_manager.update_context(blogger_id, {'content': content}, result)

            # 缓存结果
            if self.use_cache and self.cache:
                self.cache.set(content, result, stock_name)

            result.analysis_method = "llm"
            return result

        except Exception as e:
            # 回退到规则分析
            result = self._fallback_analysis(content, str(e))
            result.analysis_method = "rule_fallback"
            return result

    def analyze(
        self,
        content: str,
        stock_name: str = None,
        blogger_id: str = None
    ) -> StructuredAnalysisResult:
        """同步分析"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.analyze_async(content, stock_name, blogger_id)
                    )
                    return future.result(timeout=60)
            else:
                return loop.run_until_complete(
                    self.analyze_async(content, stock_name, blogger_id)
                )
        except Exception as e:
            return self._fallback_analysis(content, str(e))

    def _parse_response(self, response: str, original_text: str) -> StructuredAnalysisResult:
        """解析LLM响应"""
        try:
            # 提取JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}

            # 提取信息
            extracted_data = data.get('extracted_info', {})
            extracted_info = ExtractedInfo(
                stock_names=extracted_data.get('stock_names', []),
                stock_codes=extracted_data.get('stock_codes', []),
                price_target=extracted_data.get('price_target'),
                stop_loss=extracted_data.get('stop_loss'),
                stop_profit=extracted_data.get('stop_profit'),
                position_size=extracted_data.get('position_size'),
                sector=extracted_data.get('sector'),
                trading_cycle=extracted_data.get('trading_cycle')
            )

            return StructuredAnalysisResult(
                scenario_type=data.get('scenario_type', 'unknown'),
                scenario_confidence=data.get('scenario_confidence', 0.5),
                true_sentiment=data.get('true_sentiment', 'neutral'),
                sentiment_confidence=data.get('sentiment_confidence', 0.5),
                action_type=data.get('action_type', 'none'),
                action_confidence=data.get('action_confidence', 0.5),
                is_formal_opinion=data.get('is_formal_opinion', False),
                formality_confidence=data.get('formality_confidence', 0.5),
                is_real_position=data.get('is_real_position', True),
                position_confidence=data.get('position_confidence', 0.5),
                emotion=data.get('emotion', 'neutral'),
                emotion_confidence=data.get('emotion_confidence', 0.5),
                extracted_info=extracted_info,
                reasoning=data.get('reasoning', ''),
                original_text=original_text
            )
        except Exception as e:
            return self._fallback_analysis(original_text, str(e))

    def _rule_based_analysis(self, content: str, stock_name: str = None) -> str:
        """基于规则的分析（回退方案）"""
        # 简单规则判断
        action = 'none'
        sentiment = 'neutral'
        scenario = 'discussion'

        buy_keywords = ['买入', '建仓', '加仓', '上车', '入手']
        sell_keywords = ['卖出', '清仓', '止盈', '止损', '离场']

        for kw in buy_keywords:
            if kw in content:
                action = 'buy'
                sentiment = 'bullish'
                scenario = 'real_trading'
                break

        for kw in sell_keywords:
            if kw in content:
                action = 'sell'
                sentiment = 'bearish'
                scenario = 'real_trading'
                break

        result = {
            'scenario_type': scenario,
            'scenario_confidence': 0.6,
            'true_sentiment': sentiment,
            'sentiment_confidence': 0.6,
            'action_type': action,
            'action_confidence': 0.6,
            'is_formal_opinion': True,
            'formality_confidence': 0.5,
            'is_real_position': True,
            'position_confidence': 0.5,
            'emotion': 'neutral',
            'emotion_confidence': 0.5,
            'extracted_info': {
                'stock_names': [stock_name] if stock_name else []
            },
            'reasoning': '基于关键词匹配'
        }

        return json.dumps(result, ensure_ascii=False)

    def _fallback_analysis(self, content: str, error: str) -> StructuredAnalysisResult:
        """回退分析"""
        response = self._rule_based_analysis(content)
        result = self._parse_response(response, content)
        result.reasoning = f"LLM分析失败({error})，使用规则回退"
        return result

    async def batch_analyze(
        self,
        contents: List[str],
        stock_names: List[str] = None,
        blogger_id: str = None
    ) -> List[StructuredAnalysisResult]:
        """批量分析（并行）"""
        tasks = []
        for i, content in enumerate(contents):
            stock_name = stock_names[i] if stock_names else None
            tasks.append(self.analyze_async(content, stock_name, blogger_id))

        return await asyncio.gather(*tasks)


# ==================== 便捷函数 ====================

def analyze_semantic_enhanced(
    content: str,
    stock_name: str = None,
    blogger_id: str = None
) -> StructuredAnalysisResult:
    """便捷函数：增强版语义分析"""
    analyzer = EnhancedSemanticAnalyzer()
    return analyzer.analyze(content, stock_name, blogger_id)


def batch_analyze_semantic(
    contents: List[str],
    stock_names: List[str] = None,
    blogger_id: str = None
) -> List[StructuredAnalysisResult]:
    """便捷函数：批量语义分析"""
    analyzer = EnhancedSemanticAnalyzer()
    return asyncio.run(analyzer.batch_analyze(contents, stock_names, blogger_id))


# ==================== 测试 ====================

if __name__ == "__main__":
    print("测试增强版语义分析器...")

    test_cases = [
        "今天买了茅台，感觉要起飞了",
        "如果明天大盘涨，我就加仓宁德时代",
        "卖飞了，可惜可惜",
        "不看好白酒板块，已经清仓了",
        "模拟盘测试一下这个策略",
    ]

    analyzer = EnhancedSemanticAnalyzer()

    for content in test_cases:
        print(f"\n分析: {content}")
        result = analyzer.analyze(content)
        print(f"  场景: {result.scenario_type} ({result.scenario_confidence:.2f})")
        print(f"  意图: {result.true_sentiment} ({result.sentiment_confidence:.2f})")
        print(f"  操作: {result.action_type} ({result.action_confidence:.2f})")
        print(f"  质量: {result.quality.overall_score:.2f}")
        if result.quality.issues:
            print(f"  问题: {result.quality.issues}")
