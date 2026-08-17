"""
多因子选股模型
基于量化因子进行股票评分和选股
支持因子：技术因子、情绪因子、资金因子、基本面因子（财报/北向/龙虎榜）
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json


@dataclass
class FactorScore:
    """因子评分结果"""
    factor_name: str
    raw_value: float  # 原始值
    normalized_score: float  # 标准化分数 (0-100)
    weight: float  # 权重
    weighted_score: float  # 加权分数
    description: str = ""


@dataclass
class StockScore:
    """股票综合评分"""
    code: str
    name: str
    total_score: float  # 总分
    factor_scores: List[FactorScore] = field(default_factory=list)
    rank: int = 0  # 排名
    recommendation: str = ""  # 推荐等级
    confidence: float = 0.0  # 置信度
    fundamental_data: Dict = field(default_factory=dict)  # 基本面数据详情


class BaseFactor(ABC):
    """因子基类"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def calculate(self, data: Dict) -> float:
        """计算因子原始值"""
        pass

    @abstractmethod
    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        """标准化因子值到0-100"""
        pass

    def get_description(self, raw_value: float, normalized_score: float) -> str:
        """获取因子描述"""
        return f"{self.name}: {raw_value:.2f} -> {normalized_score:.1f}分"


class TrendFactor(BaseFactor):
    """趋势反转因子 - IC=-0.495表明A股短线趋势延续失效，反转更强

    原逻辑（趋势延续）：均线多头=加分 → IC为负，说明追涨失效
    新逻辑（趋势反转）：均线空头/超跌=加分，均线多头=减分（可能见顶）
    """

    def __init__(self, weight: float = 0.25):
        super().__init__("趋势因子", weight)

    def calculate(self, data: Dict) -> float:
        """
        计算趋势反转得分（高分=超跌反弹机会，低分=趋势高位风险）
        """
        close = data.get('close', 0)
        ma5 = data.get('ma5', close)
        ma10 = data.get('ma10', close)
        ma20 = data.get('ma20', close)
        ma60 = data.get('ma60', close)

        score = 50  # 基准分

        # 价格在均线下方 = 超跌反弹机会（加分）
        if close < ma5 and close > ma5 * 0.97:  # 接近MA5但略低于（蓄势突破）
            score += 15
        if close < ma20 and close > ma20 * 0.93:  # 接近MA20但略低于（中期支撑）
            score += 20

        # 价格远低于MA60 = 深度超跌，反弹空间大
        if close < ma60 * 0.95 and close > ma60 * 0.85:
            score += 15

        # 均线空头排列但收敛 = 趋势即将反转
        if ma5 < ma10 < ma20 and (ma10 - ma5) < (ma20 - ma10) * 0.5:
            score += 20
        elif ma5 < ma10 and (ma10 - ma5) < ma10 * 0.02:
            score += 10

        # 均线多头排列 = 趋势高位，追涨风险大（减分）
        if ma5 > ma10 > ma20:
            score -= 25
        elif ma5 > ma10:
            score -= 10

        # 价格远高于MA20 = 短期过热（减分）
        if close > ma20 * 1.1:
            score -= 15

        # 均线斜率：下降趋势放缓=反转信号（加分），上升趋势加速=见顶风险（减分）
        ma5_slope = data.get('ma5_slope', 0)
        if -2 < ma5_slope < 0:  # 下降趋势放缓
            score += 10
        elif ma5_slope > 3:  # 上升过快
            score -= 10

        return max(0, min(100, score))

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value  # 已经是0-100


class MomentumFactor(BaseFactor):
    """动量因子 - 基于涨跌幅和RSI"""

    def __init__(self, weight: float = 0.20):
        super().__init__("动量因子", weight)

    def calculate(self, data: Dict) -> float:
        """
        计算动量得分
        考虑：近期涨跌幅、RSI、MACD
        """
        score = 0

        # 近5日涨跌幅
        change_5d = data.get('change_5d', 0)
        if change_5d > 10:
            score += 30
        elif change_5d > 5:
            score += 25
        elif change_5d > 0:
            score += 15
        elif change_5d > -5:
            score += 10
        else:
            score += 5

        # RSI指标
        rsi = data.get('rsi', 50)
        if 50 < rsi < 70:  # 强势但不超买
            score += 25
        elif 30 < rsi <= 50:  # 中性
            score += 15
        elif rsi <= 30:  # 超卖，可能反弹
            score += 20
        elif rsi > 70:  # 超买，风险
            score += 10

        # MACD
        macd = data.get('macd', 0)
        signal = data.get('macd_signal', 0)
        if macd > 0 and macd > signal:
            score += 25
        elif macd > 0:
            score += 15
        elif macd > signal:  # 金叉
            score += 20

        return min(score, 100)

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value


class VolumeFactor(BaseFactor):
    """量能因子 - 基于成交量和换手率"""

    def __init__(self, weight: float = 0.15):
        super().__init__("量能因子", weight)

    def calculate(self, data: Dict) -> float:
        """
        计算量能得分
        考虑：成交量变化、换手率、量价关系
        """
        score = 0

        # 成交量变化
        volume_change = data.get('volume_change_pct', 0)
        if volume_change > 100:  # 放量
            score += 30
        elif volume_change > 50:
            score += 25
        elif volume_change > 20:
            score += 20
        elif volume_change > 0:
            score += 15
        else:
            score += 10

        # 换手率
        turnover_rate = data.get('turnover_rate', 0)
        if 3 < turnover_rate < 10:  # 活跃但不过热
            score += 25
        elif 1 < turnover_rate <= 3:
            score += 20
        elif turnover_rate >= 10:  # 过热
            score += 15
        else:
            score += 10

        # 量价关系
        change_pct = data.get('change_pct', 0)
        if change_pct > 0 and volume_change > 0:  # 量价齐升
            score += 25
        elif change_pct < 0 and volume_change < 0:  # 缩量下跌
            score += 15
        elif change_pct > 0 and volume_change < 0:  # 缩量上涨
            score += 20

        return min(score, 100)

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value


class MoneyFlowFactor(BaseFactor):
    """资金因子 - 基于主力资金流向"""

    def __init__(self, weight: float = 0.20):
        super().__init__("资金因子", weight)

    def calculate(self, data: Dict) -> float:
        """
        计算资金得分
        考虑：主力净流入、大单占比、北向资金
        """
        score = 50  # 基础分

        # 主力净流入
        main_inflow = data.get('main_net_inflow_pct', 0)
        if main_inflow > 5:
            score += 30
        elif main_inflow > 2:
            score += 20
        elif main_inflow > 0:
            score += 10
        elif main_inflow > -2:
            score -= 5
        else:
            score -= 15

        # 大单净流入
        big_order_inflow = data.get('big_order_inflow_pct', 0)
        if big_order_inflow > 3:
            score += 15
        elif big_order_inflow > 0:
            score += 8

        # 北向资金（如果有）
        north_inflow = data.get('north_inflow', 0)
        if north_inflow > 0:
            score += 10

        return max(0, min(score, 100))

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value


class VolatilityFactor(BaseFactor):
    """波动因子 - 基于ATR和波动率"""

    def __init__(self, weight: float = 0.10):
        super().__init__("波动因子", weight)

    def calculate(self, data: Dict) -> float:
        """
        计算波动得分
        适度的波动有利于交易，过高或过低都不好
        """
        score = 50

        # ATR百分比
        atr_pct = data.get('atr_pct', 3)
        if 2 < atr_pct < 4:  # 适度波动
            score += 30
        elif 1 < atr_pct <= 2 or 4 < atr_pct <= 5:
            score += 20
        elif atr_pct > 5:  # 波动过大
            score -= 10
        else:  # 波动过小
            score += 10

        # 年化波动率
        volatility = data.get('volatility', 30)
        if 20 < volatility < 35:
            score += 20
        elif 15 < volatility <= 20 or 35 < volatility <= 45:
            score += 10
        elif volatility > 50:
            score -= 15

        return max(0, min(score, 100))

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value


class SentimentFactor(BaseFactor):
    """情绪因子 - 基于市场情绪和板块热度"""

    def __init__(self, weight: float = 0.10):
        super().__init__("情绪因子", weight)

    def calculate(self, data: Dict) -> float:
        """
        计算情绪得分
        考虑：市场情绪、板块热度、概念炒作
        """
        score = 50

        # 市场情绪
        market_sentiment = data.get('market_sentiment', 'neutral')
        if market_sentiment == 'strong_bull':
            score += 30
        elif market_sentiment == 'bull':
            score += 20
        elif market_sentiment == 'bear':
            score -= 10
        elif market_sentiment == 'strong_bear':
            score -= 20

        # 是否热门板块
        is_hot_sector = data.get('is_hot_sector', False)
        if is_hot_sector:
            score += 20

        # 概念热度
        concept_heat = data.get('concept_heat', 0)
        if concept_heat > 0:
            score += min(concept_heat * 5, 15)

        return max(0, min(score, 100))

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value


class FundamentalFactor(BaseFactor):
    """基本面因子 - 基于财报/北向资金/龙虎榜"""

    def __init__(self, weight: float = 0.15):
        super().__init__("基本面因子", weight)

    def calculate(self, data: Dict) -> float:
        """
        计算基本面得分
        考虑：财务质量、成长性、估值、北向资金、龙虎榜
        """
        score = 50

        # 财务质量评分
        quality_score = data.get('financial_quality_score', 50)
        score += (quality_score - 50) * 0.3

        # 成长性评分
        growth_score = data.get('financial_growth_score', 50)
        score += (growth_score - 50) * 0.25

        # 估值评分
        valuation_score = data.get('financial_valuation_score', 50)
        score += (valuation_score - 50) * 0.15

        # 北向资金评分
        north_score = data.get('north_bound_score', 50)
        score += (north_score - 50) * 0.2

        # 龙虎榜评分
        dragon_score = data.get('dragon_tiger_score', 50)
        score += (dragon_score - 50) * 0.1

        return max(0, min(score, 100))

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value

    def get_description(self, raw_value: float, normalized_score: float) -> str:
        return f"{self.name}: 基本面综合评分 {normalized_score:.1f}分"


class FinancialQualityFactor(BaseFactor):
    """财务质量因子 - ROE/毛利率/负债率等"""

    def __init__(self, weight: float = 0.08):
        super().__init__("财务质量因子", weight)

    def calculate(self, data: Dict) -> float:
        """计算财务质量得分"""
        score = 50

        # ROE
        roe = data.get('roe', 0)
        if roe > 20:
            score += 20
        elif roe > 15:
            score += 15
        elif roe > 10:
            score += 10
        elif roe > 5:
            score += 5
        else:
            score -= 10

        # 毛利率
        gross_margin = data.get('gross_margin', 0)
        if gross_margin > 40:
            score += 15
        elif gross_margin > 25:
            score += 10
        elif gross_margin > 15:
            score += 5

        # 净利率
        net_margin = data.get('net_margin', 0)
        if net_margin > 15:
            score += 10
        elif net_margin > 8:
            score += 5

        # 资产负债率
        debt_ratio = data.get('asset_liability_ratio', 50)
        if 30 < debt_ratio < 60:
            score += 5
        elif debt_ratio > 80:
            score -= 15

        # 现金流
        cash_flow = data.get('operating_cash_flow', 0)
        if cash_flow > 0:
            score += 5

        return max(0, min(score, 100))

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value


class GrowthFactor(BaseFactor):
    """成长因子 - 营收/利润增长率"""

    def __init__(self, weight: float = 0.07):
        super().__init__("成长因子", weight)

    def calculate(self, data: Dict) -> float:
        """计算成长性得分"""
        score = 50

        # 营收增长
        revenue_yoy = data.get('revenue_yoy', 0)
        if revenue_yoy > 50:
            score += 25
        elif revenue_yoy > 30:
            score += 20
        elif revenue_yoy > 15:
            score += 15
        elif revenue_yoy > 0:
            score += 5
        else:
            score -= 15

        # 利润增长
        profit_yoy = data.get('net_profit_yoy', 0)
        if profit_yoy > 50:
            score += 25
        elif profit_yoy > 30:
            score += 20
        elif profit_yoy > 15:
            score += 15
        elif profit_yoy > 0:
            score += 5
        else:
            score -= 15

        return max(0, min(score, 100))

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value


class NorthBoundFactor(BaseFactor):
    """北向资金因子 - 外资持股和流向"""

    def __init__(self, weight: float = 0.08):
        super().__init__("北向资金因子", weight)

    def calculate(self, data: Dict) -> float:
        """计算北向资金得分"""
        score = 50

        # 持股比例
        hold_ratio = data.get('north_hold_ratio', 0)
        if hold_ratio > 10:
            score += 20
        elif hold_ratio > 5:
            score += 15
        elif hold_ratio > 2:
            score += 10
        elif hold_ratio > 0:
            score += 5

        # 持股变化
        hold_change_5d = data.get('north_hold_change_5d', 0)
        if hold_change_5d > 5:
            score += 15
        elif hold_change_5d > 2:
            score += 10
        elif hold_change_5d > 0:
            score += 5
        elif hold_change_5d < -5:
            score -= 15

        # 净买入
        net_buy_5d = data.get('north_net_buy_5d', 0)
        if net_buy_5d > 1:
            score += 10
        elif net_buy_5d < -1:
            score -= 10

        return max(0, min(score, 100))

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value


class DragonTigerFactor(BaseFactor):
    """龙虎榜因子 - 游资/机构动向"""

    def __init__(self, weight: float = 0.05):
        super().__init__("龙虎榜因子", weight)

    def calculate(self, data: Dict) -> float:
        """计算龙虎榜得分"""
        score = 50

        # 近期是否上龙虎榜
        on_dragon_list = data.get('on_dragon_tiger_list', False)
        if not on_dragon_list:
            return 50  # 未上榜给中性分

        # 净买入
        dragon_net_buy = data.get('dragon_net_buy', 0)
        if dragon_net_buy > 5000:  # 净买入超5000万
            score += 20
        elif dragon_net_buy > 2000:
            score += 15
        elif dragon_net_buy > 0:
            score += 10
        elif dragon_net_buy < -5000:
            score -= 15

        # 龙虎榜评分
        dragon_score = data.get('dragon_tiger_score', 50)
        score += (dragon_score - 50) * 0.5

        return max(0, min(score, 100))

    def normalize(self, raw_value: float, peer_values: List[float] = None) -> float:
        return raw_value


class MultiFactorModel:
    """多因子选股模型 - 支持复盘反馈自适应权重调整"""

    def __init__(self, use_fundamental: bool = True,
                 use_feedback: bool = True):
        """
        初始化多因子模型

        Args:
            use_fundamental: 是否启用基本面因子
            use_feedback: 是否启用复盘反馈权重调整
        """
        self.use_fundamental = use_fundamental

        # 技术面因子（趋势已改为反转逻辑）
        self.factors = [
            TrendFactor(weight=0.10),        # IC=-0.495→改为反转，观察期暂低权
            # MomentumFactor — IC=0，下线
            # VolumeFactor — IC=0，下线
            # MoneyFlowFactor — IC=0，下线
            VolatilityFactor(weight=0.30),    # IC=0.30，上调为核心因子
            # SentimentFactor — IC=0，下线
        ]

        # 基本面因子（龙虎榜IC最高，大幅提权；北向暂停）
        if use_fundamental:
            self.factors.extend([
                FinancialQualityFactor(weight=0.25),  # IC=0.13
                GrowthFactor(weight=0.15),            # IC=0.29
                # NorthBoundFactor — IC=-0.047，暂停正向加权
                DragonTigerFactor(weight=0.20),       # IC=0.34，最有效，提权
            ])

        # 复盘反馈闭环：根据历史偏差调整因子权重
        if use_feedback:
            self._apply_feedback_adjustment()

        # 归一化权重
        total_weight = sum(f.weight for f in self.factors)
        for f in self.factors:
            f.weight = f.weight / total_weight

    def _apply_feedback_adjustment(self):
        """根据复盘反馈+IC检验调整因子权重

        两步调整：
        1. IC检验：用Spearman秩相关验证因子是否有预测力，剔除IC≈0的无效因子
        2. 偏差调整：根据过去预测偏差方向微调权重
        """
        # --- Step 1: IC有效性检验 ---
        eliminated = set()
        try:
            from app.selector.factor_ic_test import FactorICTester
            ic_tester = FactorICTester()
            ic_report = ic_tester.run_ic_test(lookback_days=30)

            if ic_report.total_samples >= 5:
                for r in ic_report.factor_results:
                    if r.weight_adjustment == 0.0:
                        eliminated.add(r.factor_name)
                    elif r.weight_adjustment != 1.0:
                        # IC低但非零→降权
                        for factor in self.factors:
                            if factor.name == r.factor_name:
                                factor.weight = max(0.01, factor.weight * r.weight_adjustment)

                if eliminated:
                    print(f"IC检验剔除无效因子: {', '.join(eliminated)}")

                for r in ic_report.factor_results:
                    if r.is_valid:
                        print(f"IC有效: {r.factor_name} IC={r.ic:.4f} IR={r.ic_ir:.2f} → {r.recommendation}")
                    else:
                        print(f"IC无效: {r.factor_name} IC={r.ic:.4f} IR={r.ic_ir:.2f} → {r.recommendation}")
        except Exception as e:
            print(f"IC检验失败: {e}")

        # 剔除IC≈0的因子（权重归零）
        self.factors = [f for f in self.factors if f.name not in eliminated]

        # --- Step 2: 复盘偏差调整 ---
        try:
            from app.reviewer.feedback_adjuster import FeedbackAdjuster
            adjuster = FeedbackAdjuster()
            result = adjuster.analyze_feedback(days=5)

            if result.total_samples == 0:
                return

            adj_map = {
                '趋势因子': result.trend_weight_adj,
                '动量因子': result.momentum_weight_adj,
                '情绪因子': result.sentiment_factor_adj,
            }

            for factor in self.factors:
                adj = adj_map.get(factor.name, 1.0)
                if adj != 1.0:
                    factor.weight = max(0.01, factor.weight * adj)

            print(f"偏差调整: sentiment→{result.sentiment_factor_adj:.3f}, "
                  f"trend→{result.trend_weight_adj:.3f}, "
                  f"momentum→{result.momentum_weight_adj:.3f}")
        except Exception as e:
            print(f"偏差调整失败: {e}")

    def score_stock(self, code: str, name: str, data: Dict) -> StockScore:
        """
        对单只股票进行评分

        Args:
            code: 股票代码
            name: 股票名称
            data: 股票数据字典

        Returns:
            StockScore: 评分结果
        """
        factor_scores = []
        total_weighted_score = 0
        total_weight = 0

        for factor in self.factors:
            raw_value = factor.calculate(data)
            normalized_score = factor.normalize(raw_value)
            weighted_score = normalized_score * factor.weight

            factor_scores.append(FactorScore(
                factor_name=factor.name,
                raw_value=raw_value,
                normalized_score=normalized_score,
                weight=factor.weight,
                weighted_score=weighted_score,
                description=factor.get_description(raw_value, normalized_score)
            ))

            total_weighted_score += weighted_score
            total_weight += factor.weight

        # 计算总分
        total_score = total_weighted_score / total_weight if total_weight > 0 else 0

        # 计算置信度（基于数据完整性）
        data_completeness = sum(1 for k, v in data.items() if v is not None and v != 0) / len(data)
        confidence = min(data_completeness * 0.5 + 0.5, 1.0)

        # 推荐等级
        if total_score >= 75:
            recommendation = "强烈推荐"
        elif total_score >= 60:
            recommendation = "推荐"
        elif total_score >= 45:
            recommendation = "观望"
        else:
            recommendation = "不推荐"

        return StockScore(
            code=code,
            name=name,
            total_score=total_score,
            factor_scores=factor_scores,
            recommendation=recommendation,
            confidence=confidence
        )

    def score_stocks(self, stocks_data: List[Dict]) -> List[StockScore]:
        """
        批量评分并排名

        Args:
            stocks_data: 股票数据列表，每个元素包含 code, name, data

        Returns:
            List[StockScore]: 排序后的评分结果
        """
        scores = []
        for stock in stocks_data:
            score = self.score_stock(
                stock['code'],
                stock.get('name', stock['code']),
                stock.get('data', {})
            )
            scores.append(score)

        # 按总分排序
        scores.sort(key=lambda x: x.total_score, reverse=True)

        # 设置排名
        for i, score in enumerate(scores, 1):
            score.rank = i

        return scores

    def get_top_stocks(self, scores: List[StockScore], top_n: int = 10) -> List[StockScore]:
        """获取排名前N的股票"""
        return scores[:top_n]

    def generate_score_report(self, scores: List[StockScore]) -> str:
        """生成评分报告"""
        report = f"""
# 📊 多因子选股评分报告

> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
> 评分模型：六因子模型（趋势+动量+量能+资金+波动+情绪）

---

## 评分概览

| 排名 | 代码 | 名称 | 总分 | 推荐等级 | 置信度 |
|------|------|------|------|----------|--------|
"""
        for score in scores[:20]:
            report += f"| {score.rank} | {score.code} | {score.name} | {score.total_score:.1f} | {score.recommendation} | {score.confidence:.0%} |\n"

        # 详细评分
        report += "\n---\n\n## 详细评分\n"
        for score in scores[:10]:
            report += f"\n### {score.rank}. {score.code} {score.name} ({score.total_score:.1f}分)\n\n"
            report += "| 因子 | 原始值 | 标准分 | 权重 | 加权分 |\n"
            report += "|------|--------|--------|------|--------|\n"
            for fs in score.factor_scores:
                report += f"| {fs.factor_name} | {fs.raw_value:.2f} | {fs.normalized_score:.1f} | {fs.weight:.0%} | {fs.weighted_score:.1f} |\n"
            report += f"\n**推荐等级**: {score.recommendation}\n"
            report += f"**置信度**: {score.confidence:.0%}\n"

        report += """

---

## 📝 说明

- **趋势因子(25%)**: 基于均线系统判断趋势方向和强度
- **动量因子(20%)**: 基于涨跌幅、RSI、MACD判断动量
- **量能因子(15%)**: 基于成交量、换手率判断活跃度
- **资金因子(20%)**: 基于主力资金流向判断资金面
- **波动因子(10%)**: 基于ATR、波动率判断风险收益比
- **情绪因子(10%)**: 基于市场情绪、板块热度判断市场氛围

**风险提示**: 本评分仅供参考，不构成投资建议。
"""
        return report


def prepare_stock_data_for_scoring(code: str, market_client=None,
                                    fundamental_client=None,
                                    include_fundamental: bool = True) -> Dict:
    """
    准备股票数据用于评分

    Args:
        code: 股票代码
        market_client: 市场数据客户端
        fundamental_client: 基本面数据客户端
        include_fundamental: 是否包含基本面数据

    Returns:
        Dict: 股票数据字典
    """
    data = {}

    try:
        if market_client is None:
            from app.market import get_market_client
            market_client = get_market_client()

        # 获取实时数据
        realtime = market_client.get_stock_price(code)
        if realtime:
            data['close'] = realtime.get('price', 0)
            data['change_pct'] = realtime.get('change_pct', 0)
            data['volume_change_pct'] = realtime.get('volume_change_pct', 0)
            data['turnover_rate'] = realtime.get('turnover_rate', 0)

        # 获取技术指标
        tech = market_client.get_stock_technical_indicators(code)
        if tech:
            data['ma5'] = tech.get('ma5', data.get('close', 0))
            data['ma10'] = tech.get('ma10', data.get('close', 0))
            data['ma20'] = tech.get('ma20', data.get('close', 0))
            data['ma60'] = tech.get('ma60', data.get('close', 0))
            data['rsi'] = tech.get('rsi', 50)
            data['macd'] = tech.get('macd', 0)
            data['macd_signal'] = tech.get('macd_signal', 0)
            data['ma5_slope'] = tech.get('ma5_slope', 0)
            data['change_5d'] = tech.get('change_5d', 0)

        # 获取资金流向
        try:
            flow = market_client.get_money_flow(code)
            if flow:
                data['main_net_inflow_pct'] = flow.get('main_net_inflow_pct', 0)
                data['big_order_inflow_pct'] = flow.get('big_order_inflow_pct', 0)
        except:
            pass

        # 获取波动性数据
        try:
            df = market_client.get_stock_history(code, days=30)
            if not df.empty and len(df) >= 14:
                # 计算ATR
                high = df['high'].tail(14).values
                low = df['low'].tail(14).values
                close_prev = df['close'].tail(15).values[:-1]
                tr_list = []
                for i in range(14):
                    tr = max(high[i] - low[i], abs(high[i] - close_prev[i]), abs(low[i] - close_prev[i]))
                    tr_list.append(tr)
                atr = sum(tr_list) / 14
                data['atr_pct'] = (atr / data.get('close', 1)) * 100

                # 计算年化波动率
                returns = df['close'].pct_change().dropna()
                data['volatility'] = returns.std() * np.sqrt(252) * 100
        except:
            pass

        # 获取基本面数据
        if include_fundamental:
            try:
                if fundamental_client is None:
                    from app.market.fundamental_data import get_fundamental_client
                    fundamental_client = get_fundamental_client()

                fundamental = fundamental_client.get_stock_fundamental(code)
                if fundamental:
                    # 财务数据
                    fin = fundamental.get('financial')
                    if fin:
                        data['roe'] = fin.get('roe', 0)
                        data['gross_margin'] = fin.get('gross_margin', 0)
                        data['net_margin'] = fin.get('net_margin', 0)
                        data['asset_liability_ratio'] = fin.get('asset_liability_ratio', 0)
                        data['current_ratio'] = fin.get('current_ratio', 0)
                        data['operating_cash_flow'] = fin.get('operating_cash_flow', 0)
                        data['revenue_yoy'] = fin.get('revenue_yoy', 0)
                        data['net_profit_yoy'] = fin.get('net_profit_yoy', 0)
                        data['financial_quality_score'] = fin.get('quality_score', 50)
                        data['financial_growth_score'] = fin.get('growth_score', 50)
                        data['financial_valuation_score'] = fin.get('valuation_score', 50)

                    # 北向资金
                    north = fundamental.get('north_bound')
                    if north:
                        data['north_hold_ratio'] = north.get('hold_ratio', 0)
                        data['north_hold_change_5d'] = north.get('hold_change_5d', 0)
                        data['north_net_buy_5d'] = north.get('net_buy_5d', 0)
                        data['north_bound_score'] = north.get('flow_score', 50)

                    # 龙虎榜
                    dragons = fundamental.get('dragon_tiger', [])
                    if dragons:
                        data['on_dragon_tiger_list'] = True
                        latest_dragon = dragons[0] if dragons else {}
                        data['dragon_net_buy'] = latest_dragon.get('net_buy', 0)
                        data['dragon_tiger_score'] = latest_dragon.get('dragon_score', 50)
                    else:
                        data['on_dragon_tiger_list'] = False

            except Exception as e:
                print(f"获取基本面数据失败 {code}: {e}")

    except Exception as e:
        print(f"准备股票 {code} 数据失败: {e}")

    return data


if __name__ == "__main__":
    # 测试多因子模型
    model = MultiFactorModel(use_fundamental=True)

    # 模拟数据测试（包含基本面数据）
    test_data = {
        'close': 100,
        'ma5': 98,
        'ma10': 96,
        'ma20': 94,
        'ma60': 90,
        'ma5_slope': 2.5,
        'change_5d': 8,
        'rsi': 55,
        'macd': 0.5,
        'macd_signal': 0.3,
        'volume_change_pct': 60,
        'turnover_rate': 5,
        'change_pct': 2,
        'main_net_inflow_pct': 3,
        'big_order_inflow_pct': 2,
        'atr_pct': 3.5,
        'volatility': 28,
        'market_sentiment': 'bull',
        'is_hot_sector': True,
        # 基本面数据
        'roe': 18.5,
        'gross_margin': 35.2,
        'net_margin': 12.8,
        'asset_liability_ratio': 45,
        'current_ratio': 1.8,
        'operating_cash_flow': 5.2,
        'revenue_yoy': 25.5,
        'net_profit_yoy': 35.2,
        'financial_quality_score': 72,
        'financial_growth_score': 78,
        'financial_valuation_score': 65,
        'north_hold_ratio': 3.5,
        'north_hold_change_5d': 2.1,
        'north_net_buy_5d': 0.8,
        'north_bound_score': 68,
        'on_dragon_tiger_list': True,
        'dragon_net_buy': 3500,
        'dragon_tiger_score': 72,
    }

    score = model.score_stock('300308', '中际旭创', test_data)
    print(f"总分: {score.total_score:.1f}")
    print(f"推荐等级: {score.recommendation}")
    print(f"置信度: {score.confidence:.0%}")

    print("\n因子评分详情:")
    for fs in score.factor_scores:
        print(f"  {fs.factor_name}: {fs.normalized_score:.1f}分 (权重{fs.weight:.0%})")
