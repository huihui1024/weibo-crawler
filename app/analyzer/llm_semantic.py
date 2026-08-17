"""
LLM语义理解模块
用于解决关键词匹配无法处理的复杂语义问题：
- 区分模拟盘与实盘
- 区分闲聊与正式观点
- 处理反讽、隐喻、暗示等复杂语义
- 提取真实交易意图
"""
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class SemanticAnalysisResult:
    """语义分析结果"""
    # 场景类型
    scenario_type: str = "unknown"  # real_trading, simulation, discussion, example, review, unknown
    scenario_confidence: float = 0.0

    # 真实意图
    true_sentiment: str = "neutral"  # bullish, bearish, neutral, sarcastic
    sentiment_confidence: float = 0.0

    # 交易信号
    action_type: str = "none"  # buy, sell, hold, watch, no_position, none
    action_confidence: float = 0.0

    # 是否为正式观点（vs 闲聊）
    is_formal_opinion: bool = False
    formality_confidence: float = 0.0

    # 是否为实盘（vs 模拟盘）
    is_real_position: bool = True
    position_confidence: float = 0.0

    # 情绪分析
    emotion: str = "neutral"  # confident, uncertain, regretful, excited, fearful, neutral
    emotion_confidence: float = 0.0

    # 关键信息提取
    extracted_info: Dict = field(default_factory=dict)

    # 分析依据
    reasoning: str = ""

    # 原始文本
    original_text: str = ""


class LLMProvider(ABC):
    """LLM提供者抽象基类"""

    @abstractmethod
    async def analyze(self, prompt: str) -> str:
        """发送分析请求"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查服务是否可用"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API提供者（使用openai SDK，兼容讯飞/DeepSeek等）"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key)

    def analyze_sync(self, prompt: str) -> str:
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的股市分析师助手，擅长分析微博博主的交易意图和真实观点。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            timeout=60.0,
        )
        return response.choices[0].message.content

    async def analyze(self, prompt: str) -> str:
        return self.analyze_sync(prompt)


class ClaudeProvider(LLMProvider):
    """Claude API提供者"""

    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def analyze(self, prompt: str) -> str:
        if not self.is_available():
            raise ValueError("Claude API key not configured")

        import aiohttp

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["content"][0]["text"]
                else:
                    error = await resp.text()
                    raise Exception(f"Claude API error: {error}")


class LocalModelProvider(LLMProvider):
    """本地模型提供者（Ollama等）"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b"):
        self.base_url = base_url
        self.model = model
        self._failed = False

    def is_available(self) -> bool:
        return not self._failed

    async def analyze(self, prompt: str) -> str:
        import aiohttp

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1000
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("response", "")
                else:
                    raise Exception(f"Ollama API error: {resp.status}")


class SemanticAnalyzer:
    """语义分析器 - 核心类"""

    # 场景类型定义
    SCENARIO_TYPES = {
        "real_trading": "实盘交易 - 博主真实持仓和操作",
        "simulation": "模拟盘/假设 - 如果...那么...、假设性讨论",
        "discussion": "市场讨论 - 对市场/板块的分析讨论，非具体操作",
        "example": "举例说明 - 用具体股票举例说明观点",
        "review": "复盘总结 - 对过去操作的回顾总结",
        "unknown": "无法判断"
    }

    # 情绪类型定义
    SENTIMENT_TYPES = {
        "bullish": "看多 - 认为会上涨",
        "bearish": "看空 - 认为会下跌",
        "neutral": "中性 - 不明确或观望",
        "sarcastic": "反讽 - 说反话，实际意思与字面相反"
    }

    # 单例实例
    _instance = None
    _initialized = False

    def __new__(cls, provider: LLMProvider = None):
        """单例模式：确保只有一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, provider: LLMProvider = None):
        """
        初始化语义分析器

        Args:
            provider: LLM提供者，如果为None则自动选择
        """
        # 避免重复初始化
        if SemanticAnalyzer._initialized:
            return

        self.provider = provider or self._auto_select_provider()
        self._cache: Dict[str, SemanticAnalysisResult] = {}
        SemanticAnalyzer._initialized = True

    def _auto_select_provider(self) -> LLMProvider:
        """自动选择可用的LLM提供者"""
        # 优先从config.yaml读取LLM配置
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                llm_cfg = cfg.get('llm', {})
                api_key = llm_cfg.get('api_key', '')
                base_url = llm_cfg.get('base_url', '')
                model = llm_cfg.get('model', '')
                if api_key:
                    provider = OpenAIProvider(
                        api_key=api_key,
                        model=model or "gpt-4o-mini"
                    )
                    if base_url:
                        provider.base_url = base_url
                    print(f"使用配置文件LLM进行语义分析: model={model}, base_url={base_url}")
                    return provider
        except Exception as e:
            print(f"读取LLM配置失败: {e}")

        # 环境变量回退: OpenAI > Claude > Local
        openai_provider = OpenAIProvider()
        if openai_provider.is_available():
            print("使用 OpenAI API 进行语义分析")
            return openai_provider

        claude_provider = ClaudeProvider()
        if claude_provider.is_available():
            print("使用 Claude API 进行语义分析")
            return claude_provider

        # 默认使用本地模型
        print("使用本地模型(Ollama)进行语义分析")
        return LocalModelProvider()

    def _build_analysis_prompt(self, content: str, stock_name: str = None) -> str:
        """构建分析提示词"""
        prompt = f"""请分析以下微博内容的真实含义。

【微博内容】
{content}

请从以下几个维度进行分析：

1. **场景类型**（必选一个）：
   - real_trading: 实盘交易，博主真实持仓和操作
   - simulation: 模拟盘/假设性讨论，如"如果...就..."
   - discussion: 市场讨论，对市场/板块的分析，非具体操作
   - example: 举例说明，用股票举例说明观点
   - review: 复盘总结，对过去操作的回顾

2. **真实意图**（必选一个）：
   - bullish: 看多，认为会上涨
   - bearish: 看空，认为会下跌
   - neutral: 中性，不明确或观望
   - sarcastic: 反讽，说反话

3. **交易信号**（必选一个）：
   - buy: 买入/建仓信号
   - sell: 卖出/清仓信号
   - hold: 继续持有
   - watch: 关注但未操作
   - no_position: 明确不持有/已清仓
   - none: 无明确信号

4. **是否为正式观点**：
   - true: 正式观点，值得参考
   - false: 闲聊/吐槽，参考价值低

5. **是否为实盘**：
   - true: 实盘操作
   - false: 模拟盘/假设

6. **情绪状态**：
   - confident: 自信
   - uncertain: 不确定
   - regretful: 后悔
   - excited: 兴奋
   - fearful: 恐惧
   - neutral: 平静

请以JSON格式输出分析结果：
{{
    "scenario_type": "...",
    "scenario_confidence": 0.0-1.0,
    "true_sentiment": "...",
    "sentiment_confidence": 0.0-1.0,
    "action_type": "...",
    "action_confidence": 0.0-1.0,
    "is_formal_opinion": true/false,
    "formality_confidence": 0.0-1.0,
    "is_real_position": true/false,
    "position_confidence": 0.0-1.0,
    "emotion": "...",
    "emotion_confidence": 0.0-1.0,
    "extracted_info": {{
        "stock_name": "提到的股票",
        "price_target": "目标价位",
        "stop_loss": "止损位",
        "position_size": "仓位大小"
    }},
    "reasoning": "分析依据的简要说明"
}}
"""
        return prompt

    def _parse_llm_response(self, response: str, original_text: str) -> SemanticAnalysisResult:
        """解析LLM响应"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                raw_json = json_match.group()
                # 清理控制字符（换行、制表符等在JSON字符串值中是非法的）
                raw_json = re.sub(r'[\x00-\x1f]', '', raw_json)
                data = json.loads(raw_json)
            else:
                # 如果没有JSON，使用默认值
                data = {}

            return SemanticAnalysisResult(
                scenario_type=data.get("scenario_type", "unknown"),
                scenario_confidence=data.get("scenario_confidence", 0.5),
                true_sentiment=data.get("true_sentiment", "neutral"),
                sentiment_confidence=data.get("sentiment_confidence", 0.5),
                action_type=data.get("action_type", "none"),
                action_confidence=data.get("action_confidence", 0.5),
                is_formal_opinion=data.get("is_formal_opinion", False),
                formality_confidence=data.get("formality_confidence", 0.5),
                is_real_position=data.get("is_real_position", True),
                position_confidence=data.get("position_confidence", 0.5),
                emotion=data.get("emotion", "neutral"),
                emotion_confidence=data.get("emotion_confidence", 0.5),
                extracted_info=data.get("extracted_info", {}),
                reasoning=data.get("reasoning", ""),
                original_text=original_text
            )
        except Exception as e:
            print(f"解析LLM响应失败: {e}")
            return SemanticAnalysisResult(
                scenario_type="unknown",
                original_text=original_text,
                reasoning=f"解析失败: {str(e)}"
            )

    async def analyze_async(self, content: str, stock_name: str = None) -> SemanticAnalysisResult:
        """
        异步分析微博内容

        Args:
            content: 微博内容
            stock_name: 可选的股票名称

        Returns:
            SemanticAnalysisResult: 分析结果
        """
        # 检查缓存
        cache_key = f"{content[:100]}:{stock_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            if not self.provider.is_available():
                return SemanticAnalysisResult(
                    scenario_type="unknown",
                    original_text=content,
                    reasoning="LLM不可用"
                )
            prompt = self._build_analysis_prompt(content, stock_name)
            response = await self.provider.analyze(prompt)
            result = self._parse_llm_response(response, content)

            # 缓存结果
            self._cache[cache_key] = result
            return result

        except Exception as e:
            # 标记provider不可用，后续跳过
            if hasattr(self.provider, '_failed'):
                self.provider._failed = True
            return SemanticAnalysisResult(
                scenario_type="unknown",
                original_text=content,
                reasoning=f"分析失败: {str(e)}"
            )

    def analyze(self, content: str, stock_name: str = None) -> SemanticAnalysisResult:
        """
        同步分析微博内容

        Args:
            content: 微博内容
            stock_name: 可选的股票名称

        Returns:
            SemanticAnalysisResult: 分析结果
        """
        # 检查缓存
        cache_key = f"{content[:100]}:{stock_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            if not self.provider.is_available():
                return SemanticAnalysisResult(
                    scenario_type="unknown",
                    original_text=content,
                    reasoning="LLM不可用"
                )
            prompt = self._build_analysis_prompt(content, stock_name)
            # 优先使用同步接口（避免异步事件循环冲突）
            if hasattr(self.provider, 'analyze_sync'):
                response = self.provider.analyze_sync(prompt)
            else:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self.provider.analyze(prompt)
                            )
                            response = future.result(timeout=60)
                    else:
                        response = loop.run_until_complete(self.provider.analyze(prompt))
                except RuntimeError:
                    response = asyncio.run(self.provider.analyze(prompt))
            result = self._parse_llm_response(response, content)
            self._cache[cache_key] = result
            return result
        except Exception as e:
            print(f"同步分析失败: {e}")
            return SemanticAnalysisResult(
                scenario_type="unknown",
                original_text=content,
                reasoning=f"分析失败: {str(e)}"
            )

    def batch_analyze(self, contents: List[str], stock_name: str = None) -> List[SemanticAnalysisResult]:
        """
        批量分析微博内容

        Args:
            contents: 微博内容列表
            stock_name: 可选的股票名称

        Returns:
            List[SemanticAnalysisResult]: 分析结果列表
        """
        return [self.analyze(content, stock_name) for content in contents]


class HybridAnalyzer:
    """
    混合分析器 - 结合关键词匹配和LLM语义理解

    策略：
    1. 先用关键词快速匹配，如果置信度高则直接返回
    2. 如果关键词匹配置信度低或存在冲突，使用LLM进行深度分析
    3. 综合两者结果给出最终判断
    """

    # 单例实例
    _instance = None
    _initialized = False

    def __new__(cls, llm_provider: LLMProvider = None):
        """单例模式：确保只有一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, llm_provider: LLMProvider = None):
        # 避免重复初始化
        if HybridAnalyzer._initialized:
            return

        self.semantic_analyzer = SemanticAnalyzer(llm_provider)
        self._keyword_analyzer = None  # 延迟加载，避免循环导入
        HybridAnalyzer._initialized = True

    def _get_keyword_analyzer(self):
        """获取关键词分析器（延迟加载，禁用LLM避免循环）"""
        if self._keyword_analyzer is None:
            from .stock_analyzer import StockAnalyzer
            # 创建关键词分析器时禁用LLM，避免循环调用
            self._keyword_analyzer = StockAnalyzer(enable_llm=False)
        return self._keyword_analyzer

    def analyze_with_fallback(
        self,
        content: str,
        stock_name: str = None,
        keyword_confidence_threshold: float = 0.7
    ) -> Tuple[SemanticAnalysisResult, str]:
        """
        混合分析：关键词优先，LLM兜底

        Args:
            content: 微博内容
            stock_name: 股票名称
            keyword_confidence_threshold: 关键词置信度阈值

        Returns:
            Tuple[SemanticAnalysisResult, str]: (分析结果, 使用的方法)
        """
        # 1. 先尝试关键词匹配
        keyword_analyzer = self._get_keyword_analyzer()
        signal_type, keyword_conf, is_negated, reason = keyword_analyzer._analyze_signal_with_context(
            content, stock_name or ""
        )

        # 2. 如果关键词置信度高且无否定，直接返回
        if keyword_conf >= keyword_confidence_threshold and not is_negated:
            result = SemanticAnalysisResult(
                scenario_type="real_trading",  # 假设是实盘
                scenario_confidence=0.7,
                true_sentiment="bullish" if signal_type in ["buy", "hold"] else "bearish" if signal_type == "sell" else "neutral",
                sentiment_confidence=keyword_conf,
                action_type=signal_type,
                action_confidence=keyword_conf,
                is_formal_opinion=True,
                formality_confidence=0.7,
                is_real_position=True,
                position_confidence=0.7,
                emotion="neutral",
                emotion_confidence=0.5,
                reasoning=f"关键词匹配: {reason}",
                original_text=content
            )
            return result, "keyword"

        # 3. 否则使用LLM深度分析
        llm_result = self.semantic_analyzer.analyze(content, stock_name)

        # 4. 如果LLM分析失败，回退到关键词结果
        if llm_result.scenario_type == "unknown":
            result = SemanticAnalysisResult(
                scenario_type="real_trading",
                scenario_confidence=0.5,
                true_sentiment="neutral",
                sentiment_confidence=0.5,
                action_type=signal_type,
                action_confidence=keyword_conf,
                is_formal_opinion=True,
                formality_confidence=0.5,
                is_real_position=True,
                position_confidence=0.5,
                emotion="neutral",
                emotion_confidence=0.5,
                reasoning=f"LLM失败，使用关键词: {reason}",
                original_text=content
            )
            return result, "keyword_fallback"

        return llm_result, "llm"

    def should_use_llm(self, content: str, keyword_result: Tuple) -> bool:
        """
        判断是否需要使用LLM分析

        Args:
            content: 微博内容
            keyword_result: 关键词分析结果 (signal_type, confidence, is_negated, reason)

        Returns:
            bool: 是否需要LLM分析
        """
        signal_type, confidence, is_negated, reason = keyword_result

        # 以下情况需要LLM分析：
        # 1. 置信度低
        if confidence < 0.5:
            return True

        # 2. 检测到否定（可能是复杂语义）
        if is_negated:
            return True

        # 3. 内容包含可能需要深度理解的词汇
        complex_patterns = [
            r'如果.*就',  # 假设性
            r'假设',  # 假设
            r'要是',  # 假设
            r'早知道',  # 后悔
            r'可惜',  # 遗憾
            r'本来',  # 原本打算
            r'其实',  # 转折
            r'说真的',  # 强调
            r'开玩笑',  # 玩笑
            r'呵呵',  # 可能反讽
            r'哈',  # 可能反讽
            r'笑死',  # 可能反讽
            r'我就看看',  # 围观
            r'吃瓜',  # 围观
            r'模拟',  # 模拟盘
            r'虚拟',  # 虚拟盘
            r'小号',  # 可能是小号操作
        ]

        for pattern in complex_patterns:
            if re.search(pattern, content):
                return True

        return False


# 便捷函数
def analyze_semantic(content: str, stock_name: str = None) -> SemanticAnalysisResult:
    """便捷函数：分析单条微博的语义"""
    analyzer = SemanticAnalyzer()
    return analyzer.analyze(content, stock_name)


def analyze_with_hybrid(content: str, stock_name: str = None) -> Tuple[SemanticAnalysisResult, str]:
    """便捷函数：混合分析"""
    analyzer = HybridAnalyzer()
    return analyzer.analyze_with_fallback(content, stock_name)
