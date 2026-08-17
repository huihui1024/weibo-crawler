"""智能选股引擎
结合行业方向参考和量化多因子模型，筛选潜在标的池
"""
from typing import List, Dict, Optional, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
import json
import sys
import os
import numpy as np

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd

from app.database import get_sync_db
from app.analyzer.stock_analyzer import AnalysisResult, StockAnalyzer
from app.market import get_market_client, MarketType
from app.selector.risk_metrics import (
    VaRCalculator, PortfolioRiskAnalyzer, RiskMonitor,
    VaRMethod, PortfolioRiskReport
)


@dataclass
class StockCandidate:
    """股票候选"""
    code: str
    name: str
    sector: str
    concept: str
    price: float
    change_pct: float
    style_match_score: float  # 与博主风格匹配度 (0-100)
    selection_source: str = "市场热点"  # 选股来源: "市场热点" / "行业方向" / "量化确认"
    position_weight: float = 0.0  # 仓位参考（行业方向参考时可能有值）
    selection_reasons: List[str] = field(default_factory=list)
    technical_indicators: Dict[str, Any] = field(default_factory=dict)
    market: str = "A"
    # 多因子量化评分
    fundamental_score: float = 0.0
    valuation_score: float = 0.0
    capital_flow_score: float = 0.0
    sentiment_score: float = 0.0
    pe_percentile: float = 0.0
    pb_percentile: float = 0.0
    market_regime: str = ''
    factor_detail: Dict[str, Any] = field(default_factory=dict)
    # 产业链推理 + 认知差 + 轧空信号
    cognitive_gap_score: float = 0.0
    supply_chain_score: float = 0.0
    short_interest_ratio: float = 0.0
    volume_surge_ratio: float = 0.0
    industry_chain: str = ''
    drawdown_risk_warning: str = ''  # 回撤风险强制提示


@dataclass
class StockPrediction:
    """股票预测"""
    code: str
    name: str
    prediction_date: date
    prediction_time: datetime

    # 预测内容
    expected_today_trend: str  # 上涨/下跌/震荡
    expected_today_strength: str  # 强/中/弱
    expected_today_amplitude: float  # 预期振幅
    expected_today_change: float  # 预期涨跌幅

    expected_hold_days: int  # 建议持有天数
    expected_hold_amplitude: float  # 持股周期预期振幅

    selection_reason: str  # 选股原因
    risk_warning: str  # 风险提示

    # 技术依据
    technical_basis: Dict[str, Any] = field(default_factory=dict)


class StockSelector:
    """智能选股引擎"""

    # 博主关注板块到akshare板块的映射
    SECTOR_MAPPING = {
        '科技芯片': ['半导体', '集成电路', '国产芯片'],
        '新能源': ['锂电池', '光伏', '储能', '新能源车'],
        '光模块': ['光模块', '光通信', 'CPO'],
        '算力': ['算力', '数据中心', '云计算'],
        '医药医疗': ['创新药', '医疗器械', '中药'],
        '消费白酒': ['白酒', '食品饮料'],
        '军工航天': ['军工', '航天'],
        '周期资源': ['煤炭', '有色金属', '钢铁'],
        '煤炭': ['煤炭', '煤炭开采'],
        '商业航天': ['航天', '商业航天', '卫星导航'],
        '金融券商': ['证券', '券商'],
    }

    # 热门概念关键词
    HOT_CONCEPT_KEYWORDS = [
        'AI', '人工智能', '算力', '光模块', 'CPO', '半导体',
        '新能源', '光伏', '储能', '锂电池',
        '军工', '航天', '商业航天',
        '机器人', '自动驾驶'
    ]

    def __init__(self, blogger_strategy: AnalysisResult = None,
                 market_type: MarketType = MarketType.A_SHARE):
        """
        初始化选股引擎

        Args:
            blogger_strategy: 行业方向参考（AnalysisResult），如果为None则从数据库加载
            market_type: 市场类型 A_SHARE/HK/US
        """
        self.market_type = market_type
        self.market_client = get_market_client(market_type)
        self.blogger_strategy = blogger_strategy

    def set_blogger_strategy(self, strategy: AnalysisResult):
        """设置行业方向参考"""
        self.blogger_strategy = strategy

    def generate_candidate_pool(self, max_stocks: int = 30) -> List[StockCandidate]:
        """
        生成候选股票池

        择时优先于选股：先定能不能做，再选做哪些股

        Args:
            max_stocks: 最大返回股票数量

        Returns:
            List[StockCandidate]: 候选股票列表
        """
        if not self.blogger_strategy:
            print("警告：未设置行业方向参考，使用默认策略")
            self.blogger_strategy = self._get_default_strategy()

        # 港股/美股：使用基本面深度分析路径
        if self.market_type in (MarketType.HK, MarketType.US):
            return self._generate_hk_us_candidate_pool(max_stocks)

        # === 择时优先于选股 ===
        from app.selector.market_timing import MarketTiming, TIMING_FULL_SHORT, TIMING_CAUTIOUS_SHORT
        timing_engine = MarketTiming()
        timing_result = timing_engine.get_timing()
        timing_level = timing_result['timing']
        timing_config = timing_result['config']
        print(f"\n[大盘择时] {timing_config['label']} | {timing_result['reason']}")
        print(f"  MA5={timing_result['ma5']:.0f} MA20={timing_result['ma20']:.0f} 趋势={timing_result['trend']}")

        # 全面做空时禁止建仓，直接返回空池
        if timing_level == TIMING_FULL_SHORT:
            print("  ⛔ 全面做空环境，禁止建仓，股票池清空")
            return []

        # 根据择时调整最大股票数
        adjusted_max = max(3, int(max_stocks * timing_config['pool_ratio']))
        if adjusted_max < max_stocks:
            print(f"  择时调整: 股票池上限 {max_stocks} → {adjusted_max} (pool_ratio={timing_config['pool_ratio']})")

        candidates = []

        # 1. 获取市场热点股票（唯一有效来源，胜率67%）
        market_stocks = self._get_market_hot_stocks()
        print(f"市场热点股票: {len(market_stocks)} 只")

        # 2. 获取行业方向参考（仅用于确认热点方向，不独立入选）
        direction_stocks = self._get_sector_direction_stocks()
        print(f"行业方向参考股票: {len(direction_stocks)} 只")

        # 3. 获取系统推荐股票（暂停独立入选，仅用于交叉确认）
        system_stocks = self._get_system_recommended_stocks()
        print(f"系统推荐股票（仅交叉确认）: {len(system_stocks)} 只")

        # 4. 合并：市场热点为主，行业方向和系统推荐仅作交叉确认加分
        all_stocks = self._merge_stocks(market_stocks, direction_stocks + system_stocks)
        print(f"合并后总计: {len(all_stocks)} 只股票")

        # 4. 多因子量化4层漏斗筛选
        filtered_stocks = self._select_stocks_4layer(all_stocks)
        print(f"4层漏斗筛选后剩余 {len(filtered_stocks)} 只股票")

        # 4.1 数据质量过滤：剔除价格缺失/异常的股票
        before_quality = len(filtered_stocks)
        filtered_stocks = [s for s in filtered_stocks
                          if s.get('price', 0) > 0 and s.get('price', 0) < 1000]
        if len(filtered_stocks) < before_quality:
            print(f"数据质量过滤: {before_quality} → {len(filtered_stocks)} 只（剔除价格缺失/异常）")

        # 4.5 自适应策略：根据市场状态调整基本面门槛
        regime = self._detect_market_regime()
        from app.selector.adaptive_strategy import AdaptiveStrategyManager
        asm = AdaptiveStrategyManager()
        before_adaptive = len(filtered_stocks)
        filtered_stocks = asm.apply_fundamental_filter(filtered_stocks, regime)
        if len(filtered_stocks) < before_adaptive:
            print(f"自适应策略[{regime}]筛选: {before_adaptive} → {len(filtered_stocks)} 只")

        # 4.6 大盘择时过滤：谨慎做空/全面做空时屏蔽纯热点、提高准入门槛
        before_timing = len(filtered_stocks)
        filtered_stocks = timing_engine.apply_timing_to_pool(filtered_stocks, max_stocks)
        if len(filtered_stocks) < before_timing:
            print(f"择时过滤[{timing_config['label']}]: {before_timing} → {len(filtered_stocks)} 只")
        if not timing_config['allow_hot_stocks']:
            print(f"  ⚠️ 大盘走弱，已屏蔽纯热点题材股，仅保留基本面+龙虎榜双认证标的")
        if timing_config['min_composite_score'] > 0.3:
            print(f"  ⚠️ 准入门槛提高至综合分≥{timing_config['min_composite_score']}")

        # 5. 计算风格匹配度
        for stock in filtered_stocks:
            match_score = self._calculate_style_match(stock)
            stock['style_match_score'] = match_score

        # 5.5 交叉确认：非市场热点的股票检查是否被其他来源确认
        hot_codes = {s['code'] for s in filtered_stocks if s.get('selection_source') in ('市场热点', '热点')}
        # 博主关注的行业名称集合
        blogger_sectors = set()
        if self.blogger_strategy and self.blogger_strategy.hot_industries:
            for item in self.blogger_strategy.hot_industries:
                name = item.get('name', item) if isinstance(item, dict) else item
                blogger_sectors.add(name)

        for stock in filtered_stocks:
            src = stock.get('selection_source', '')
            # 博主+热点同指向行业 → 权重+20%
            stock_sector = stock.get('sector', '')
            stock_concept = stock.get('concept', '')
            blogger_confirmed = False
            if blogger_sectors:
                for bs in blogger_sectors:
                    if bs in stock_sector or bs in stock_concept:
                        blogger_confirmed = True
                        break

            if src in ('市场热点', '热点'):
                if blogger_confirmed:
                    # 市场热点+博主方向一致 → 强力确认
                    stock['composite_score'] = stock.get('composite_score', 0) * 1.20
                    stock.setdefault('reasons', [])
                    stock['reasons'].append('博主+热点方向一致，权重+20%')
            elif src not in ('市场热点', '热点') and stock['code'] in hot_codes:
                # 非热点但被热点确认 → 加分，改标
                stock['composite_score'] = stock.get('composite_score', 0) * 1.15
                stock['style_match_score'] = min(100, stock.get('style_match_score', 50) + 5)
                stock['selection_source'] = '市场热点'
                stock.setdefault('reasons', [])
                stock['reasons'].append('多源交叉确认: 市场热点+行业方向')
            elif src not in ('市场热点', '热点') and stock['code'] not in hot_codes:
                # 未被市场热点确认 → 降权
                stock['composite_score'] = stock.get('composite_score', 0) * 0.7
                stock['style_match_score'] = max(0, stock.get('style_match_score', 50) - 10)

        # 6. 排序：按综合得分排序
        sorted_stocks = sorted(filtered_stocks, key=lambda x: -x.get('composite_score', x.get('style_match_score', 0)))

        # 7. 按来源分配名额，确保多元化（不再以博主策略为主导）
        sina_quotes = {}
        try:
            sina_quotes = self.market_client._get_realtime_quotes_sina()
        except Exception as e:
            print(f"获取新浪行情数据失败: {e}")

        # 按来源分组
        source_groups = {'市场热点': [], '行业方向': [], '量化确认': []}
        for stock in sorted_stocks:
            src = stock.get('selection_source', '市场热点')
            if src in ('市场热点', '热点'):
                src = '市场热点'
            elif src in ('基本面驱动', '博主策略', '基本面分析'):
                src = '行业方向'
            elif src in ('系统推荐', '量化策略'):
                src = '量化确认'
            if src not in source_groups:
                src = '市场热点'
            source_groups[src].append(stock)

        # 名额分配：市场热点60%（唯一有效来源）、行业方向25%（交叉确认）、量化确认15%
        # 使用择时调整后的上限
        quota = {
            '市场热点': max(int(adjusted_max * 0.60), 6),
            '行业方向': max(int(adjusted_max * 0.25), 3),
            '量化确认': max(int(adjusted_max * 0.15), 2),
        }

        selected = []
        # 市场热点优先入选
        for src in ['市场热点', '行业方向', '量化确认']:
            group = source_groups[src]
            selected.extend(group[:quota[src]])

        # 如果某来源不足配额，将空出的名额给市场热点
        for src in ['行业方向', '量化确认']:
            group = source_groups[src]
            if len(group) < quota[src]:
                freed = quota[src] - len(group)
                hot_group = source_groups['市场热点']
                already_taken = len([s for s in selected if s.get('_src', '市场热点') == '市场热点'])
                can_add = min(freed, len(hot_group) - already_taken)
                if can_add > 0:
                    selected.extend(hot_group[already_taken:already_taken + can_add])

        # P0准入门槛：择时等级决定最低综合分要求
        min_score_gate = max(0.3, timing_config['min_composite_score'])
        before_gate = len(selected)
        all_before_gate = list(selected)  # 保存门槛前列表
        selected = [s for s in selected if s.get('composite_score', 0) >= min_score_gate]
        if len(selected) < before_gate:
            print(f"准入门槛(composite≥{min_score_gate}): {before_gate} → {len(selected)} 只")
        if not selected and all_before_gate:
            print("⚠️ 无股票通过准入门槛，放宽至composite≥0")
            selected = [s for s in all_before_gate if s.get('composite_score', 0) >= 0]

        for stock in selected[:adjusted_max]:
            # 如果价格为0或未设置，尝试从新浪数据获取
            if not stock.get('price') or stock.get('price', 0) == 0:
                code = stock['code']
                if code in sina_quotes:
                    stock['price'] = sina_quotes[code]['price']
                    stock['change_pct'] = sina_quotes[code]['change_pct']
                else:
                    try:
                        df = self.market_client.get_stock_history(stock['code'], days=1)
                        if not df.empty:
                            latest = df.iloc[-1]
                            stock['price'] = latest['close']
                            stock['change_pct'] = latest['change_pct']
                    except:
                        pass

            try:
                tech = self.market_client.get_stock_technical_indicators(stock['code'])
                stock['technical_indicators'] = tech
            except:
                stock['technical_indicators'] = {}

            # 回撤风险强制提示
            risk_warn = self._generate_risk_warning(
                stock.get('technical_indicators', {}),
                None,
                stock
            )

            candidate = StockCandidate(
                code=stock['code'],
                name=stock['name'],
                sector=stock.get('sector', ''),
                concept=stock.get('concept', ''),
                price=stock.get('price', 0),
                change_pct=stock.get('change_pct', 0),
                style_match_score=stock['style_match_score'],
                selection_source=stock.get('selection_source', '市场热点'),
                position_weight=stock.get('position_weight', 0),
                selection_reasons=stock.get('reasons', []),
                technical_indicators=stock['technical_indicators'],
                market=self.market_type.value,
                fundamental_score=stock.get('fundamental_score', 0),
                valuation_score=stock.get('valuation_score', 0),
                capital_flow_score=stock.get('capital_flow_score', 0),
                sentiment_score=stock.get('sentiment_score', 0),
                pe_percentile=stock.get('pe_percentile', 0),
                pb_percentile=stock.get('pb_percentile', 0),
                market_regime=stock.get('market_regime', ''),
                factor_detail=stock.get('factor_detail', {}),
                cognitive_gap_score=stock.get('cognitive_gap_score', 0),
                supply_chain_score=stock.get('supply_chain_score', 0),
                short_interest_ratio=stock.get('short_interest_ratio', 0),
                volume_surge_ratio=stock.get('volume_surge_ratio', 0),
                industry_chain=stock.get('industry_chain', ''),
                drawdown_risk_warning=risk_warn,
            )
            candidates.append(candidate)

        # 存储择时结果供外部使用
        self._last_timing = timing_result
        return candidates[:adjusted_max]

    def _generate_hk_us_candidate_pool(self, max_stocks: int = 30) -> List[StockCandidate]:
        """港股/美股基本面深度分析选股路径

        不再仅依赖涨幅排行，而是综合基本面(质量/成长/估值) +
        技术面(趋势/动量)进行评分预测。
        """
        market_label = '港股' if self.market_type == MarketType.HK else '美股'
        print(f"[{market_label}] 使用基本面深度分析路径")

        try:
            if self.market_type == MarketType.HK:
                from ..market.hk_client import HKMarketClient
                client = HKMarketClient()
            else:
                from ..market.us_client import USMarketClient
                client = USMarketClient()

            results = client.analyze_and_predict(limit=max_stocks)
            if not results:
                print(f"[{market_label}] 未获取到分析数据")
                return []

            candidates = []
            for stock in results:
                fin = stock.get('fundamental', {})
                fin_data = fin.get('financial', {}) if fin else {}
                breakdown = stock.get('score_breakdown', {})

                # 从基本面评分中提取选股原因
                reasons = []
                q = breakdown.get('quality', {})
                for detail in q.get('details', []):
                    reasons.append(detail)
                g = breakdown.get('growth', {})
                for detail in g.get('details', []):
                    reasons.append(detail)
                v = breakdown.get('valuation', {})
                for detail in v.get('details', []):
                    reasons.append(detail)

                # 技术面原因
                if stock.get('change_pct', 0) > 3:
                    reasons.append(f"强势上涨{stock['change_pct']:+.1f}%")
                elif stock.get('change_pct', 0) > 0:
                    reasons.append("正向动量")

                if not reasons:
                    reasons.append("综合评分通过筛选")

                candidate = StockCandidate(
                    code=stock.get('code', ''),
                    name=stock.get('name', ''),
                    sector=fin_data.get('industry', market_label) if fin_data else market_label,
                    concept='基本面优质' if stock.get('prediction_score', 0) > 60 else market_label + '热点',
                    price=stock.get('price', 0),
                    change_pct=stock.get('change_pct', 0),
                    style_match_score=float(stock.get('prediction_score', 50)),
                    selection_source='基本面分析',
                    position_weight=0,
                    selection_reasons=reasons,
                    technical_indicators=stock.get('technical_indicators', {}),
                    market=self.market_type.value,
                )
                candidates.append(candidate)

            return candidates[:max_stocks]

        except Exception as e:
            print(f"[{market_label}] 基本面分析失败，降级到涨幅排行: {e}")
            # 降级：使用原来的涨幅排行逻辑
            try:
                if self.market_type == MarketType.HK:
                    from ..market.hk_client import HKMarketClient
                    client = HKMarketClient()
                    gainers = client.fetch_top_gainers(limit=max_stocks)
                else:
                    from ..market.us_client import USMarketClient
                    client = USMarketClient()
                    gainers = client.fetch_top_gainers(limit=max_stocks)

                candidates = []
                for stock in gainers:
                    candidate = StockCandidate(
                        code=stock.get('code', ''),
                        name=stock.get('name', ''),
                        sector=market_label,
                        concept=market_label + '热点',
                        price=stock.get('price', 0),
                        change_pct=stock.get('change_pct', 0),
                        style_match_score=50.0,
                        selection_source='市场热点',
                        market=self.market_type.value,
                    )
                    candidates.append(candidate)
                return candidates[:max_stocks]
            except Exception as e2:
                print(f"[{market_label}] 降级选股也失败: {e2}")
                return []

    def _get_default_strategy(self) -> AnalysisResult:
        """获取默认策略"""
        result = AnalysisResult(
            blogger_name="默认策略",
            analysis_time=datetime.now()
        )
        result.hot_industries = ['科技芯片', '新能源']
        result.selection_factors = {'看K线形态': 10, '看业绩数据': 8, '追政策热点': 6}
        result.trading_cycle_detail = '波段（1-4周）'
        result.avoid_stocks = ['ST股、退市风险股', '高位追涨股']
        return result

    def _get_sector_direction_stocks(self) -> List[Dict]:
        """根据行业方向获取股票（博主关注板块仅作方向参考，不再按仓位权重筛选和排序）"""
        all_stocks = []
        seen_codes = set()

        # 获取新浪实时行情作为价格数据源
        sina_quotes = {}
        try:
            sina_quotes = self.market_client._get_realtime_quotes_sina()
        except Exception as e:
            print(f"获取新浪行情数据失败: {e}")

        # 优先使用博主关注板块作为行业方向参考（不再使用仓位权重）
        target_sectors = []

        if self.blogger_strategy and self.blogger_strategy.hot_industries:
            target_sectors = self.blogger_strategy.hot_industries
            print(f"行业方向参考（源自博主关注板块）: {target_sectors}")
        else:
            target_sectors = ['科技芯片', '新能源']
            print(f"使用默认板块: {target_sectors}")

        # 获取行业方向板块的股票
        network_failed = False
        for industry_item in target_sectors:
            # hot_industries可能是dict列表或str列表
            industry = industry_item.get('name', industry_item) if isinstance(industry_item, dict) else industry_item
            akshare_sectors = self.SECTOR_MAPPING.get(industry, [industry])

            for sector_name in akshare_sectors:
                try:
                    stocks = self.market_client._retry_request(
                        self.market_client.get_sector_stocks, sector_name
                    )
                    if not stocks:  # 返回空列表也视为失败
                        network_failed = True
                        continue
                    for stock in stocks:
                        if stock['code'] not in seen_codes:
                            seen_codes.add(stock['code'])
                            # 补充价格数据
                            if not stock.get('price') or stock.get('price', 0) == 0:
                                code = stock['code']
                                if code in sina_quotes:
                                    stock['price'] = sina_quotes[code]['price']
                                    stock['change_pct'] = sina_quotes[code]['change_pct']
                            stock['sector'] = industry
                            stock['concept'] = sector_name
                            stock['selection_source'] = '基本面驱动'
                            stock['position_weight'] = 0
                            stock['reasons'] = [f"行业方向: {industry}"]
                            all_stocks.append(stock)
                except Exception as e:
                    print(f"获取博主板块 {sector_name} 失败: {e}")
                    network_failed = True
                    continue

        # 如果网络请求失败，从新浪数据中通过关键词匹配
        if network_failed and sina_quotes and len(all_stocks) == 0:
            print("网络不稳定，从新浪数据中通过关键词匹配博主板块股票")
            # 板块关键词映射
            sector_keywords = {
                '煤炭': ['神华', '煤业', '能源', '焦煤', '焦炭', '煤炭'],
                '商业航天': ['航天', '卫星', '火箭', '导航', '宇航', '航空'],
                '科技芯片': ['芯片', '半导体', '集成电路', '微电子', '光刻'],
                '新能源': ['锂电', '光伏', '储能', '新能源', '电池'],
                '消费白酒': ['茅台', '五粮液', '白酒', '酒'],
                '金融券商': ['证券', '券商', '银行', '金融'],
            }

            for industry_item in target_sectors:
                industry = industry_item.get('name', industry_item) if isinstance(industry_item, dict) else industry_item
                keywords = sector_keywords.get(industry, [industry])
                matched_count = 0

                for code, quote in sina_quotes.items():
                    name = quote.get('name', '')
                    # 检查是否匹配关键词
                    if any(kw in name for kw in keywords):
                        if code not in seen_codes:
                            seen_codes.add(code)
                            stock = {
                                'code': code,
                                'name': name,
                                'price': quote.get('price', 0),
                                'change_pct': quote.get('change_pct', 0),
                                'sector': industry,
                                'concept': f'关键词匹配: {industry}',
                                'selection_source': '基本面驱动',
                                'position_weight': 0,
                                'reasons': [f"行业方向: {industry}"]
                            }
                            all_stocks.append(stock)
                            matched_count += 1
                            if matched_count >= 10:  # 每个板块最多10只
                                break

        print(f"行业方向参考股票获取完成: {len(all_stocks)} 只")
        return all_stocks

    def _get_market_hot_stocks(self) -> List[Dict]:
        """获取市场热点股票

        热点持续性判断：
        - 新鲜度评分：今日涨幅越大越新鲜(1-3天)，连续多日涨幅递减→衰减期(5天+)
        - 只推热点第1-3天的标的，衰减期股票降权
        """
        all_stocks = []
        seen_codes = set()

        # 获取新浪实时行情作为价格数据源
        sina_quotes = {}
        try:
            sina_quotes = self.market_client._get_realtime_quotes_sina()
        except Exception as e:
            print(f"获取新浪行情数据失败: {e}")

        # 1. 获取热门概念板块排行
        concept_freshness = {}
        try:
            hot_concepts = self.market_client._retry_request(
                self.market_client.get_hot_concepts, limit=10
            )
            concept_names = [c['name'] for c in hot_concepts[:5]]
            print(f"热门概念板块: {concept_names}")

            for i, concept in enumerate(hot_concepts[:10]):
                concept_name = concept['name']
                concept_change = concept.get('change_pct', 0)

                # 新鲜度评分：涨幅>5%=极新鲜, 3-5%=新鲜, 1.5-3%=观察, <1.5%=衰减
                if concept_change >= 5:
                    freshness = 100
                elif concept_change >= 3:
                    freshness = 80
                elif concept_change >= 1.5:
                    freshness = 60
                elif concept_change >= 0.5:
                    freshness = 30
                else:
                    freshness = 10
                freshness += max(0, (5 - i) * 3)  # 排名加分
                freshness = min(100, freshness)
                concept_freshness[concept_name] = freshness

                # 衰减期概念不取成分股
                if freshness < 30:
                    print(f"  跳过衰减概念: {concept_name}({concept_change:+.2f}%) 新鲜度={freshness}")
                    continue

                try:
                    stocks = self.market_client._retry_request(
                        self.market_client.get_concept_stocks, concept_name
                    )
                    for stock in stocks:
                        if stock['code'] not in seen_codes:
                            seen_codes.add(stock['code'])
                            if not stock.get('price') or stock.get('price', 0) == 0:
                                code = stock['code']
                                if code in sina_quotes:
                                    stock['price'] = sina_quotes[code]['price']
                                    stock['change_pct'] = sina_quotes[code]['change_pct']
                            stock['sector'] = '市场热点'
                            stock['concept'] = concept_name
                            stock['selection_source'] = '市场热点'
                            stock['position_weight'] = 0
                            stock['hot_freshness'] = freshness
                            fl = '极新鲜' if freshness >= 90 else '新鲜' if freshness >= 70 else '观察' if freshness >= 50 else '衰减'
                            stock['reasons'] = [f"热门概念: {concept_name}({concept_change:+.2f}%) [{fl}]"]
                            all_stocks.append(stock)
                except Exception as e:
                    print(f"获取概念 {concept_name} 成分股失败: {e}")
                    continue
        except Exception as e:
            print(f"获取热门概念失败: {e}")

        # 2. 获取涨幅榜前30只股票（市场强势股），排除ST股
        try:
            top_gainers = self.market_client._retry_request(
                self.market_client.get_top_gainers, limit=50
            )
            non_st_count = 0
            for stock in top_gainers:
                name = stock.get('name', '')
                if 'ST' in name or '退' in name:
                    continue
                if stock['code'] not in seen_codes:
                    seen_codes.add(stock['code'])
                    change = stock.get('change_pct', 0)
                    if change >= 9.9:
                        freshness = 100
                    elif change >= 5:
                        freshness = 85
                    elif change >= 3:
                        freshness = 70
                    else:
                        freshness = 50
                    stock['sector'] = '涨幅榜'
                    stock['concept'] = '强势股'
                    stock['selection_source'] = '市场热点'
                    stock['position_weight'] = 0
                    stock['hot_freshness'] = freshness
                    stock['reasons'] = [f"涨幅榜: {change:+.2f}%"]
                    all_stocks.append(stock)
                    non_st_count += 1
                    if non_st_count >= 30:
                        break
        except Exception as e:
            print(f"获取涨幅榜失败: {e}")

        # 3. 衰减期股票标记
        for stock in all_stocks:
            freshness = stock.get('hot_freshness', 50)
            stock['hot_decay'] = freshness < 30

        fresh_count = sum(1 for s in all_stocks if s.get('hot_freshness', 0) >= 50)
        decay_count = sum(1 for s in all_stocks if s.get('hot_freshness', 0) < 50)
        print(f"市场热点股票获取完成: {len(all_stocks)} 只 (新鲜{fresh_count}/衰减{decay_count})")
        return all_stocks

    # ==================== 系统推荐 — 主流量化策略 ====================

    def _get_system_recommended_stocks(self) -> List[Dict]:
        """获取系统推荐股票 — 基于主流市场量化策略"""
        all_stocks = []
        seen_codes = set()

        # 获取新浪实时行情作为价格数据源
        sina_quotes = {}
        try:
            sina_quotes = self.market_client._get_realtime_quotes_sina()
        except Exception as e:
            print(f"获取新浪行情数据失败: {e}")

        # 按优先级依次执行各策略
        strategies = [
            ("龙虎榜机构净买入", self._strategy_lhb_institutional),
            ("机构调研热度", self._strategy_institutional_research),
            ("强势股(涨停新高)", self._strategy_strong_stocks),
            ("连续量价齐升", self._strategy_volume_price_rise),
            ("涨停池", self._strategy_limit_up_pool),
        ]

        for strategy_name, strategy_func in strategies:
            try:
                stocks = strategy_func(sina_quotes)
                count = 0
                for stock in stocks:
                    code = stock.get('code', '')
                    if code and code not in seen_codes:
                        seen_codes.add(code)
                        # 补充价格
                        if not stock.get('price') or stock.get('price', 0) == 0:
                            if code in sina_quotes:
                                stock['price'] = sina_quotes[code]['price']
                                stock['change_pct'] = sina_quotes[code]['change_pct']
                        stock['selection_source'] = '系统推荐'
                        stock['position_weight'] = 0
                        all_stocks.append(stock)
                        count += 1
                print(f"  系统推荐[{strategy_name}]: {count} 只")
            except Exception as e:
                print(f"  系统推荐[{strategy_name}]失败: {e}")

        print(f"系统推荐股票获取完成: {len(all_stocks)} 只")
        return all_stocks

    def _strategy_lhb_institutional(self, sina_quotes: Dict) -> List[Dict]:
        """策略1: 龙虎榜机构净买入 — 机构资金认可的股票"""
        import akshare as ak
        stocks = []
        try:
            df = ak.stock_lhb_stock_statistic_em(symbol='近一月')
            if df is None or df.empty:
                return stocks

            # 筛选: 机构买入净额>0, 上榜次数>=3, 排除ST/退市
            for _, row in df.iterrows():
                name = str(row.get('名称', ''))
                code = str(row.get('代码', ''))
                if 'ST' in name or '退' in name:
                    continue
                buy_net = float(row.get('机构买入净额', 0) or 0)
                count = int(row.get('上榜次数', 0) or 0)
                if buy_net <= 0 or count < 3:
                    continue

                stocks.append({
                    'code': code,
                    'name': name,
                    'price': float(row.get('收盘价', 0) or 0),
                    'change_pct': float(row.get('涨跌幅', 0) or 0),
                    'sector': str(row.get('所属行业', '')) if '所属行业' in df.columns else '',
                    'concept': '龙虎榜机构买入',
                    'reasons': [f"龙虎榜: 机构净买入{buy_net/1e8:.2f}亿, 上榜{count}次"],
                })
                if len(stocks) >= 10:
                    break
        except Exception as e:
            print(f"龙虎榜策略异常: {e}")
        return stocks

    def _strategy_institutional_research(self, sina_quotes: Dict) -> List[Dict]:
        """策略2: 机构调研热度 — 机构密集调研意味着关注度上升"""
        import akshare as ak
        stocks = []
        try:
            df = ak.stock_jgdy_tj_em()
            if df is None or df.empty:
                return stocks

            # 筛选: 接待机构数量>=20, 按机构数降序
            df_sorted = df.sort_values(by='接待机构数量', ascending=False)
            for _, row in df_sorted.iterrows():
                name = str(row.get('名称', ''))
                code = str(row.get('代码', ''))
                if 'ST' in name or '退' in name:
                    continue
                inst_count = int(row.get('接待机构数量', 0) or 0)
                if inst_count < 20:
                    continue

                stocks.append({
                    'code': code,
                    'name': name,
                    'price': float(row.get('最新价', 0) or 0),
                    'change_pct': float(row.get('涨跌幅', 0) or 0),
                    'sector': '',
                    'concept': '机构调研',
                    'reasons': [f"机构调研: {inst_count}家机构调研"],
                })
                if len(stocks) >= 10:
                    break
        except Exception as e:
            print(f"机构调研策略异常: {e}")
        return stocks

    def _strategy_strong_stocks(self, sina_quotes: Dict) -> List[Dict]:
        """策略3: 强势股(涨停新高) — 突破新高是趋势确认信号"""
        import akshare as ak
        stocks = []
        try:
            # 找最近有数据的交易日
            df = None
            from datetime import date, timedelta
            for days_back in range(0, 7):
                d = (date.today() - timedelta(days=days_back)).strftime('%Y%m%d')
                try:
                    df = ak.stock_zt_pool_strong_em(date=d)
                    if df is not None and not df.empty:
                        break
                except Exception:
                    continue
            if df is None or df.empty:
                return stocks

            for _, row in df.iterrows():
                name = str(row.get('名称', ''))
                code = str(row.get('代码', ''))
                if 'ST' in name or '退' in name:
                    continue
                is_new_high = str(row.get('是否新高', '')) == '是'
                turnover = float(row.get('换手率', 0) or 0)
                if not is_new_high or turnover > 20:
                    continue

                reason = str(row.get('入选理由', ''))
                industry = str(row.get('所属行业', '')) if '所属行业' in df.columns else ''
                stocks.append({
                    'code': code,
                    'name': name,
                    'price': float(row.get('最新价', 0) or 0),
                    'change_pct': float(row.get('涨跌幅', 0) or 0),
                    'sector': industry,
                    'concept': '强势股',
                    'reasons': [f"强势股: {reason}, 换手{turnover:.1f}%"],
                })
                if len(stocks) >= 10:
                    break
        except Exception as e:
            print(f"强势股策略异常: {e}")
        return stocks

    def _strategy_volume_price_rise(self, sina_quotes: Dict) -> List[Dict]:
        """策略4: 连续量价齐升 — 量价配合是最可靠的趋势信号"""
        import akshare as ak
        stocks = []
        try:
            df = ak.stock_rank_ljqs_ths()
            if df is None or df.empty:
                return stocks

            for _, row in df.iterrows():
                name = str(row.get('股票简称', ''))
                code = str(row.get('股票代码', ''))
                if 'ST' in name or '退' in name:
                    continue
                days = int(row.get('量价齐升天数', 0) or 0)
                if days < 3:
                    continue
                gain = float(row.get('阶段涨幅', 0) or 0)
                industry = str(row.get('所属行业', '')) if '所属行业' in df.columns else ''

                stocks.append({
                    'code': code,
                    'name': name,
                    'price': float(row.get('最新价', 0) or 0),
                    'change_pct': 0,
                    'sector': industry,
                    'concept': '量价齐升',
                    'reasons': [f"量价齐升: 连续{days}天, 阶段涨幅{gain:.1f}%"],
                })
                if len(stocks) >= 10:
                    break
        except Exception as e:
            print(f"量价齐升策略异常: {e}")
        return stocks

    def _strategy_limit_up_pool(self, sina_quotes: Dict) -> List[Dict]:
        """策略5: 涨停池 — 连板股/首板热门板块，跟踪市场最强方向"""
        import akshare as ak
        stocks = []
        try:
            # 找最近有数据的交易日
            df = None
            from datetime import date, timedelta
            for days_back in range(0, 7):
                d = (date.today() - timedelta(days=days_back)).strftime('%Y%m%d')
                try:
                    df = ak.stock_zt_pool_em(date=d)
                    if df is not None and not df.empty:
                        break
                except Exception:
                    continue
            if df is None or df.empty:
                return stocks

            for _, row in df.iterrows():
                name = str(row.get('名称', ''))
                code = str(row.get('代码', ''))
                if 'ST' in name or '退' in name:
                    continue
                consecutive = int(row.get('连板数', 0) or 0)
                industry = str(row.get('所属行业', '')) if '所属行业' in df.columns else ''
                # 连板>=2 或 首板+行业热门
                if consecutive >= 2:
                    pass
                elif consecutive == 1 and any(kw in industry for kw in self.HOT_CONCEPT_KEYWORDS):
                    pass
                else:
                    continue

                stocks.append({
                    'code': code,
                    'name': name,
                    'price': float(row.get('最新价', 0) or 0),
                    'change_pct': float(row.get('涨跌幅', 0) or 0),
                    'sector': industry,
                    'concept': '涨停',
                    'reasons': [f"涨停: {consecutive}连板, {industry}"],
                })
                if len(stocks) >= 5:
                    break
        except Exception as e:
            print(f"涨停池策略异常: {e}")
        return stocks

    def _merge_stocks(self, blogger_stocks: List[Dict], market_stocks: List[Dict]) -> List[Dict]:
        """合并股票池（来源平等，去重即可）"""
        merged = {}

        # 先添加博主关注板块的股票（作为基本面驱动来源之一）
        for stock in blogger_stocks:
            code = stock['code']
            # 将"博主策略"来源改为"基本面驱动"
            if stock.get('selection_source') == '博主策略':
                stock['selection_source'] = '基本面驱动'
            merged[code] = stock

        # 再添加其他来源股票（不覆盖已有）
        for stock in market_stocks:
            code = stock['code']
            if code not in merged:
                merged[code] = stock

        return list(merged.values())

    def _filter_by_factors(self, stocks: List[Dict]) -> List[Dict]:
        """根据博主选股因子筛选股票"""
        if not self.blogger_strategy:
            return stocks

        filtered = []
        tech_failed_count = 0  # 技术指标获取失败计数

        for stock in stocks:
            # 保留原有的选股原因
            original_reasons = stock.get('reasons', [])
            reasons = list(original_reasons)  # 复制原有原因
            score = 0

            # 获取技术指标
            try:
                tech = self.market_client.get_stock_technical_indicators(stock['code'])
            except:
                tech = {}
                tech_failed_count += 1

            # 因子1: K线形态
            if self.blogger_strategy.selection_factors.get('看K线形态', 0) > 0:
                if tech.get('is_above_ma5') and tech.get('is_above_ma20'):
                    reasons.append("均线多头排列")
                    score += 20
                if tech.get('trend') == '上涨':
                    reasons.append("上升趋势")
                    score += 15
                # 如果技术指标获取失败，给予基础分
                if not tech and tech_failed_count > len(stocks) * 0.5:
                    score += 10  # 网络不稳定时降低筛选条件

            # 因子2: 业绩数据（通过PE判断）
            if self.blogger_strategy.selection_factors.get('看业绩数据', 0) > 0:
                pe = stock.get('pe_ratio', 0)
                if 0 < pe < 50:  # 合理PE范围
                    reasons.append(f"PE合理({pe:.1f})")
                    score += 10
                elif not tech:  # 网络不稳定时给予基础分
                    score += 5

            # 因子3: 政策热点
            if self.blogger_strategy.selection_factors.get('追政策热点', 0) > 0:
                concept = stock.get('concept', '')
                if any(kw in concept for kw in self.HOT_CONCEPT_KEYWORDS):
                    reasons.append("热门概念")
                    score += 15
                elif stock.get('sector'):  # 有板块信息也加分
                    reasons.append(f"板块: {stock.get('sector')}")
                    score += 10

            # 排除ST股
            if 'ST' in stock['name'] or 'st' in stock['name']:
                continue

            # 排除涨幅过大的股票
            if stock.get('change_pct', 0) > 9.5:  # 涨停附近不追
                continue

            # 如果网络不稳定（大部分技术指标获取失败），降低筛选门槛
            if tech_failed_count > len(stocks) * 0.5:
                # 只要不是ST且没涨停就通过
                if score == 0:
                    reasons.append("符合基本条件")
                    score = 10

            if score > 0:
                stock['reasons'] = reasons
                stock['factor_score'] = score
                filtered.append(stock)

        # 如果筛选后为空，返回原始股票列表（排除ST和涨停）
        if not filtered and stocks:
            print("网络不稳定，降低筛选条件，保留所有非ST非涨停股票")
            for stock in stocks:
                if 'ST' not in stock['name'] and 'st' not in stock['name']:
                    if stock.get('change_pct', 0) <= 9.5:
                        stock['reasons'] = stock.get('reasons', []) + ['符合基本筛选条件']
                        stock['factor_score'] = 10
                        filtered.append(stock)

        return filtered

    # ==================== 多因子量化4层漏斗 ====================

    def _select_stocks_4layer(self, candidates: List[Dict]) -> List[Dict]:
        """多因子量化4层漏斗选股"""
        if not candidates:
            return candidates

        try:
            # L0: 热点生命周期标记（启动/发酵/尾声）
            self._mark_hot_lifecycle(candidates)

            # 预取基本面数据（批量缓存，后续L1/L2/L3直接读取）
            self._prefetch_fundamental_data(candidates)

            # L1: 基本面硬筛选（含质押率、商誉、审计意见）
            l1_pass = self._l1_fundamental_filter(candidates)

            # L1全部被淘汰时回退：保留所有候选，给予默认低分
            if not l1_pass:
                print("L1基本面筛选全部淘汰，回退保留所有候选")
                for s in candidates:
                    s['fundamental_score'] = 30
                l1_pass = list(candidates)

            # L2: 估值百分位评分
            l2_scored = self._l2_valuation_score(l1_pass)

            # L3: 资金流因子
            l3_scored = self._l3_capital_flow_score(l2_scored)

            # L3.5: 认知差因子 (博主认知领先/机构覆盖真空)
            l3_5_scored = self._l3_5_cognitive_gap_score(l3_scored)

            # L4: 情绪增强
            l4_scored = self._l4_sentiment_enhance(l3_5_scored)

            # L4.5: 产业链推理 (发现瓶颈节点/定价权)
            l4_5_scored = self._l4_5_chain_reasoning_score(l4_scored)

            # 因子标准化 + 动态加权
            final = self._combine_factors_dynamic(l4_5_scored)

            return final

        except Exception as e:
            print(f"4层漏斗异常，回退到基础筛选: {e}")
            return self._filter_by_factors(candidates)

    def _prefetch_fundamental_data(self, candidates: List[Dict]):
        """预取所有候选股票的基本面数据并缓存，供L1/L2/L3使用
        使用轻量版（2次API调用/股），避免完整版5+次调用导致超时"""
        import time
        fd_client = self._get_fundamental_client()
        if not fd_client:
            return

        start_time = time.time()
        fetched = 0
        cached = 0
        failed = 0

        for stock in candidates:
            code = stock.get('code', '')
            if not code:
                continue

            # 检查缓存是否已有数据
            if fd_client._use_cache and fd_client._cache:
                cached_data = fd_client._cache.get('financial', code)
                if cached_data:
                    stock['_prefetched_fundamental'] = fd_client.get_stock_fundamental(code)
                    cached += 1
                    continue

            try:
                data = fd_client.get_stock_fundamental_lite(code)
                if data and data.get('financial'):
                    stock['_prefetched_fundamental'] = data
                    fetched += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            # 总预取时间限制120秒（轻量版每只约3秒，120秒可处理约40只）
            if time.time() - start_time > 120:
                print(f"基本面预取超时({time.time()-start_time:.0f}s)，已获取{fetched}只，剩余跳过")
                break

        elapsed = time.time() - start_time
        if fetched + cached > 0:
            print(f"基本面数据预取完成: {fetched}只新获取+{cached}只缓存命中+{failed}只失败, 耗时{elapsed:.0f}s")

    def _mark_hot_lifecycle(self, candidates: List[Dict]):
        """L0: 热点生命周期标记 — 启动/发酵/尾声
        基于hot_freshness和概念日涨跌幅判断阶段:
        - 启动(launch): freshness>=80 且 概念日涨幅>3%
        - 发酵(ferment): freshness 50-80 或 概念日涨幅1-3%
        - 尾声(ending): freshness<50 或 概念日涨幅<0%
        """
        for stock in candidates:
            freshness = stock.get('hot_freshness', 50)
            concept_change = 0
            # 从选股原因中提取概念涨跌幅
            for reason in stock.get('reasons', []):
                if '概念' in reason or '涨幅' in reason:
                    import re
                    match = re.search(r'([+-]?\d+\.?\d*)%', reason)
                    if match:
                        concept_change = float(match.group(1))
                        break

            if freshness >= 80 and concept_change > 3:
                lifecycle = 'launch'
                lifecycle_label = '启动'
            elif freshness >= 50 or concept_change > 0:
                lifecycle = 'ferment'
                lifecycle_label = '发酵'
            else:
                lifecycle = 'ending'
                lifecycle_label = '尾声'

            stock['hot_lifecycle'] = lifecycle
            stock['hot_lifecycle_label'] = lifecycle_label

        launch_count = sum(1 for s in candidates if s.get('hot_lifecycle') == 'launch')
        ferment_count = sum(1 for s in candidates if s.get('hot_lifecycle') == 'ferment')
        ending_count = sum(1 for s in candidates if s.get('hot_lifecycle') == 'ending')
        if launch_count + ferment_count + ending_count > 0:
            print(f"热点生命周期: 启动{launch_count}/发酵{ferment_count}/尾声{ending_count}")

    def _get_fundamental_client(self):
        """获取基本面数据客户端（懒加载）"""
        try:
            from app.market.fundamental_data import get_fundamental_client
            return get_fundamental_client()
        except Exception as e:
            print(f"获取基本面客户端失败: {e}")
            return None

    def _l1_fundamental_filter(self, candidates: List[Dict]) -> List[Dict]:
        """L1: 基本面排除法 — 不用于选股加分，只排除财务有问题的股票

        排除条件（任一满足即排除）：
        - ROE < 5%：盈利能力严重不足
        - 营收增长 < -20%：营收大幅下滑
        - OCF < 0 且 ROE < 8%：现金流为负且盈利弱
        - 质押率 > 50%：大股东质押比例过高，流动性风险
        - 商誉占净资产 > 30%：商誉减值风险
        - 非标审计意见：财务可信度存疑

        通过排除的股票不加分，fundamental_score统一设为50（中性）
        """
        fd_client = self._get_fundamental_client()

        passed = []
        for i, stock in enumerate(candidates):
            try:
                # 优先使用预取数据
                data = stock.get('_prefetched_fundamental')
                if not data and fd_client:
                    data = fd_client.get_stock_fundamental(stock['code'])

                if data:
                    fin = data.get('financial', {}) or {}
                    roe = fin.get('roe', 0) or 0
                    rev_growth = fin.get('revenue_yoy', 0) or 0
                    ocf = fin.get('operating_cash_flow', 0) or 0
                    goodwill_ratio = data.get('financial', {}).get('goodwill_ratio', 0) or 0
                    # 从财报数据中获取质押率
                    pledge_ratio = 0  # 新浪源暂无质押率数据
                    audit_opinion = ''  # 新浪源暂无审计意见

                    should_exclude = False
                    exclude_reasons = []

                    if roe < 5:
                        should_exclude = True
                        exclude_reasons.append(f'ROE={roe:.1f}%<5%')
                    if rev_growth < -20:
                        should_exclude = True
                        exclude_reasons.append(f'营收增长={rev_growth:.1f}%<-20%')
                    if ocf < 0 and roe < 8:
                        should_exclude = True
                        exclude_reasons.append(f'OCF={ocf:.1f}<0且ROE={roe:.1f}%<8%')
                    if pledge_ratio > 50:
                        should_exclude = True
                        exclude_reasons.append(f'质押率={pledge_ratio:.1f}%>50%')
                    if goodwill_ratio > 30:
                        should_exclude = True
                        exclude_reasons.append(f'商誉占比={goodwill_ratio:.1f}%>30%')
                    if audit_opinion and '非标' in audit_opinion:
                        should_exclude = True
                        exclude_reasons.append(f'审计意见={audit_opinion}')

                    if should_exclude:
                        stock['l1_pass'] = False
                        stock['l1_reject_reason'] = '; '.join(exclude_reasons)
                    else:
                        stock['fundamental_score'] = 50
                        stock['fundamental_data'] = data
                        stock['l1_pass'] = True
                        passed.append(stock)
                else:
                    stock['fundamental_score'] = 50
                    stock['l1_pass'] = True
                    passed.append(stock)
            except Exception as e:
                stock['fundamental_score'] = 50
                stock['l1_pass'] = True
                passed.append(stock)

        excluded_count = sum(1 for s in candidates if s.get('l1_pass') == False)
        print(f"L1基本面排除: {len(passed)}只通过, {excluded_count}只排除")
        return passed

    def _l2_valuation_score(self, candidates: List[Dict]) -> List[Dict]:
        """L2: 估值百分位评分 — PE/PB 3年百分位越低分越高
        优先使用预取的基本面数据，避免重复API调用
        """
        from app.market.fundamental_data import get_valuation_percentile, get_fundamental_client

        import time
        start_time = time.time()

        fd_client = get_fundamental_client()
        for i, stock in enumerate(candidates):
            # 优先从预取数据中直接计算估值
            prefetched = stock.get('_prefetched_fundamental')
            if prefetched and prefetched.get('financial'):
                try:
                    pe_pct, pb_pct = get_valuation_percentile(stock['code'], fd_client)
                    stock['pe_percentile'] = pe_pct
                    stock['pb_percentile'] = pb_pct
                    stock['valuation_score'] = round(100 - (pe_pct * 0.6 + pb_pct * 0.4), 1)
                    continue
                except Exception:
                    pass

            # 回退：直接调用（有缓存则快）
            try:
                if time.time() - start_time > 120:
                    stock['valuation_score'] = 50
                    stock['pe_percentile'] = 50
                    stock['pb_percentile'] = 50
                    continue
                pe_pct, pb_pct = get_valuation_percentile(stock['code'], fd_client)
                stock['pe_percentile'] = pe_pct
                stock['pb_percentile'] = pb_pct
                stock['valuation_score'] = round(100 - (pe_pct * 0.6 + pb_pct * 0.4), 1)
            except Exception:
                stock['valuation_score'] = 50
                stock['pe_percentile'] = 50
                stock['pb_percentile'] = 50

        return candidates

    def _l3_capital_flow_score(self, candidates: List[Dict]) -> List[Dict]:
        """L3: 资金流评分 — 北向资金持股变化+5日净流入
        优先使用预取数据中的北向资金，避免重复API调用
        """
        for i, stock in enumerate(candidates):
            # 优先使用预取数据
            prefetched = stock.get('_prefetched_fundamental')
            if prefetched and prefetched.get('north_bound'):
                nb = prefetched['north_bound']
                stock['capital_flow_score'] = min(100, max(0, nb.get('flow_score', 50)))
                continue

            # 回退：从基本面客户端获取
            try:
                fd_client = self._get_fundamental_client()
                if fd_client:
                    flow = fd_client.get_north_bound_flow(stock['code'])
                    if flow and hasattr(flow, 'flow_score'):
                        stock['capital_flow_score'] = min(100, max(0, flow.flow_score))
                    else:
                        stock['capital_flow_score'] = 50
                else:
                    stock['capital_flow_score'] = 50
            except Exception:
                stock['capital_flow_score'] = 50

        return candidates

    def _l4_sentiment_enhance(self, candidates: List[Dict]) -> List[Dict]:
        """L4: 情绪增强 — 博主信号+板块热度+市场情绪，看空信号降级"""
        # 获取看空列表（用于降级）
        bearish_codes = set()
        if self.blogger_strategy and hasattr(self.blogger_strategy, 'bearish_positions'):
            for pos in (self.blogger_strategy.bearish_positions or []):
                if isinstance(pos, dict) and pos.get('code'):
                    bearish_codes.add(pos['code'])

        for stock in candidates:
            try:
                base = self._calculate_style_match(stock)

                # 看空信号降级：博主明确看空/已清仓的股票大幅扣分
                code = stock.get('code', '')
                if code in bearish_codes:
                    base = max(0, base * 0.3)  # 降级到30%水平
                    stock.setdefault('reasons', [])
                    if '博主看空' not in stock.get('reasons', []):
                        stock['reasons'] = stock.get('reasons', []) + ['博主看空/已清仓']

                # position_weight=0 的博主策略股票（仅提及无仓位）应降级
                if stock.get('selection_source') == '博主策略' and stock.get('position_weight', 0) == 0:
                    base = max(0, base * 0.6)  # 降级到60%水平

                stock['sentiment_score'] = min(100, base)
            except Exception:
                stock['sentiment_score'] = 50

        return candidates

    # ==================== L3.5 认知差因子 ====================
    def _l3_5_cognitive_gap_score(self, candidates: List[Dict]) -> List[Dict]:
        """L3.5: 认知差评分 — 博主讨论热度/机构覆盖度 × 市值分组

        核心逻辑(Serenity案例): 小市值+低机构覆盖+博主频繁提及 = 定价真空 = 认知差优势
        """
        import time
        start_time = time.time()
        MAX_TOTAL_TIME = 60

        # 获取博主提及的股票频次
        mention_counts = {}
        if self.blogger_strategy:
            for pos in (self.blogger_strategy.current_positions or []):
                name = pos.get('name', '')
                if name:
                    mention_counts[name] = pos.get('mention_count', 1)
            for pos in (self.blogger_strategy.watch_positions or []):
                name = pos.get('name', '')
                if name:
                    mention_counts[name] = pos.get('mention_count', 1)
            for pos in (self.blogger_strategy.likely_positions or []):
                name = pos.get('name', '')
                if name:
                    mention_counts[name] = pos.get('mention_count', 1)

        for i, stock in enumerate(candidates):
            elapsed = time.time() - start_time
            if elapsed > MAX_TOTAL_TIME:
                remaining = candidates[i:]
                for s in remaining:
                    s['cognitive_gap_score'] = 50
                    s['volume_surge_ratio'] = 0
                print(f"L3.5认知差评分超时({elapsed:.0f}s)，{len(remaining)}只股票使用默认值")
                break

            score = 50  # 基础分

            # 1. 市值分组 — 从fundamental_data获取或从价格估算
            market_cap = self._estimate_market_cap(stock)
            if market_cap < 50:     # <50亿: 小市值,机构难以覆盖
                score += 25
            elif market_cap < 200:  # 50-200亿: 中小市值
                score += 15
            elif market_cap < 500:  # 200-500亿: 中市值
                score += 5
            # >500亿: 大市值,机构覆盖充分,认知差低

            # 2. 博主提及频次 — 仅作轻量参考（非主导因子）
            name = stock.get('name', '')
            mention_count = mention_counts.get(name, 0)
            if mention_count >= 5:
                score += 10
            elif mention_count >= 3:
                score += 7
            elif mention_count >= 1:
                score += 3

            # 3. 机构覆盖度估算 — 用市值近似
            # 小市值 → 默认低覆盖 → 认知差加分(已在市值部分计算)
            # 尝试获取研报数量(akshare),超时则跳过
            try:
                analyst_count = self._get_analyst_coverage(stock['code'])
                if analyst_count == 0:
                    score += 10  # 无研报覆盖 = 机构认知真空
                elif analyst_count <= 2:
                    score += 5
                elif analyst_count >= 10:
                    score -= 5  # 过多覆盖 = 无认知差
            except Exception:
                pass

            stock['cognitive_gap_score'] = min(100, max(0, score))

            # 5. 成交量异动率(对所有候选股计算，不再仅限博主策略)
            if mention_count > 0:
                stock['volume_surge_ratio'] = self._calculate_volume_surge(stock)
            else:
                stock['volume_surge_ratio'] = 0

        return candidates

    def _estimate_market_cap(self, stock: Dict) -> float:
        """估算市值(亿元) — 从价格×fundamental_data获取,或默认值"""
        try:
            fd_client = self._get_fundamental_client()
            if fd_client:
                data = fd_client.get_stock_fundamental(stock['code'])
                if data and data.get('total_market_cap'):
                    return float(data['total_market_cap']) / 1e8  # 转换为亿元
        except Exception:
            pass

        # 默认: 假设中市值(200亿)
        return 200

    def _get_analyst_coverage(self, code: str) -> int:
        """获取券商研报覆盖数量(akshare)"""
        try:
            import akshare as ak
            df = ak.stock_comment_detail_zhfp(code)
            if df is not None and not df.empty:
                return len(df)
        except Exception:
            pass
        return 0

    def _calculate_volume_surge(self, stock: Dict) -> float:
        """计算成交量异动率 — 近5日vs前5日成交量变化倍数"""
        try:
            df = self.market_client.get_stock_history(stock['code'], days=20)
            if df.empty or len(df) < 15:
                return 0

            recent_vol = df['volume'].tail(5).mean()
            prev_vol = df['volume'].iloc[-15:-10].mean()
            if prev_vol > 0:
                surge = recent_vol / prev_vol
                return round(min(surge, 20), 2)  # 钳位到20倍
        except Exception:
            pass
        return 0

    # ==================== L4.5 产业链推理评分 ====================
    def _l4_5_chain_reasoning_score(self, candidates: List[Dict]) -> List[Dict]:
        """L4.5: 产业链推理评分 — 发现瓶颈节点(定价权)

        核心逻辑(Serenity案例): GPU→互联→CPO→激光器怕热→ELS→SIVE
        处于瓶颈节点的公司有定价权, 能卡住下游脖子
        """
        from app.selector.chain_reasoner import ChainReasoner

        reasoner = ChainReasoner()

        # 获取博主提及的板块列表
        mentioned_sectors = []
        if self.blogger_strategy and self.blogger_strategy.hot_industries:
            for ind in self.blogger_strategy.hot_industries:
                name = ind.get('name', ind) if isinstance(ind, dict) else ind
                mentioned_sectors.append(name)

        import time
        start_time = time.time()
        MAX_TOTAL_TIME = 60

        for i, stock in enumerate(candidates):
            elapsed = time.time() - start_time
            if elapsed > MAX_TOTAL_TIME:
                remaining = candidates[i:]
                for s in remaining:
                    s['supply_chain_score'] = 50
                    s['industry_chain'] = ''
                print(f"L4.5产业链推理超时({elapsed:.0f}s)，{len(remaining)}只股票使用默认值")
                break

            sector = stock.get('sector', '')
            name = stock.get('name', '')

            result = reasoner.reason_chain(sector, name, mentioned_sectors)

            stock['supply_chain_score'] = result.supply_chain_score
            stock['industry_chain'] = result.chain_path

            # 瓶颈节点 + 认知差高 → 双重加分
            if result.is_bottleneck and stock.get('cognitive_gap_score', 50) > 60:
                stock['supply_chain_score'] = min(100, result.supply_chain_score + 15)

            # 写入推理信息到reasons
            if result.chain_position != "unknown":
                stock.setdefault('reasons', [])
                if result.is_bottleneck:
                    stock['reasons'] = stock.get('reasons', []) + [
                        f"产业链瓶颈: {result.bottleneck_node}"
                    ]
                stock['reasons'] = stock.get('reasons', []) + [
                    f"产业链: {result.chain_path}"
                ]

        return candidates

    # ==================== 轧空信号 ====================
    def _calculate_short_squeeze_score(self, stock: Dict) -> float:
        """轧空信号评分 — 做空比例 × 小市值 × 博主看多

        核心逻辑: 做空比例高 + 小市值 + 博主看多 = 潜在轧空爆发
        """
        score = 0

        # 1. 做空比例(融资融券数据)
        short_ratio = self._get_short_interest_ratio(stock['code'])
        stock['short_interest_ratio'] = short_ratio

        if short_ratio > 10:
            score += 30  # 高做空比例 = 轧空潜力大
        elif short_ratio > 5:
            score += 15
        elif short_ratio > 2:
            score += 5

        # 2. 市值(小市值更容易轧空)
        market_cap = self._estimate_market_cap(stock)
        if market_cap < 100:
            score += 20
        elif market_cap < 300:
            score += 10

        # 3. 市值(小市值更容易轧空)
        market_cap = self._estimate_market_cap(stock)
        if market_cap < 100:
            score += 20
        elif market_cap < 300:
            score += 10

        # 4. 成交量异动(空头开始回补的迹象)
        surge = stock.get('volume_surge_ratio', 0)
        if surge > 3:
            score += 15  # 成交量暴涨 = 可能空头回补
        elif surge > 1.5:
            score += 5

        return min(100, max(0, score))

    def _get_short_interest_ratio(self, code: str) -> float:
        """获取融券余量/流通股本 = 做空比例(%)"""
        try:
            import akshare as ak
            # 尝试获取融资融券数据
            if code.startswith('6'):
                df = ak.stock_margin_sse(start_date="", end_date="", code=code)
            else:
                df = ak.stock_margin_underlying_info_szse(code)

            if df is not None and not df.empty:
                recent = df.tail(1)
                short_volume = recent['融券余量'].values[0]
                total_volume = recent['融券余量'].values[0] + recent['融资买入额'].values[0] if '融资买入额' in recent.columns else 0
                if total_volume > 0:
                    return round(short_volume / total_volume * 100, 2)
        except Exception:
            pass
        return 0

    def _combine_factors_dynamic(self, candidates: List[Dict]) -> List[Dict]:
        """因子标准化 + 基于市场状态的动态加权

        小样本(<10)使用百分位替代z-score，避免极端值扭曲
        """
        from app.market.fundamental_data import standardize_factors

        if len(candidates) < 3:
            # 候选太少，无法做有意义的标准化
            for s in candidates:
                s.setdefault('composite_score', 50)
                s.setdefault('market_regime', 'oscillation')
                s['style_match_score'] = s.get('sentiment_score', 50)
            return candidates

        # 检测市场状态
        regime = self._detect_market_regime()
        weights = self._get_regime_weights(regime)

        # 自适应策略：权重偏移叠加
        from app.selector.adaptive_strategy import AdaptiveStrategyManager
        asm = AdaptiveStrategyManager()
        weights = asm.apply_weight_adjustment(weights, regime)

        # 构建DataFrame做标准化（fundamental_score已改为排除法，不参与评分）
        df = pd.DataFrame(candidates)
        factor_cols = [
            'valuation_score', 'capital_flow_score',
            'sentiment_score', 'cognitive_gap_score', 'supply_chain_score'
        ]

        # 确保列存在并填充默认值
        for col in factor_cols:
            if col not in df.columns:
                df[col] = 50
            df[col] = df[col].fillna(50)

        # 小样本(<10): 使用百分位替代z-score，避免极端值扭曲
        if len(candidates) < 10:
            for col in factor_cols:
                # 百分位排名: (rank - 1) / (n - 1)，映射到[-1, 1]
                ranked = df[col].rank(pct=True)
                df[col] = (ranked - 0.5) * 2  # 百分位→[-1, 1]，原地替换
        else:
            # 大样本: winsorize + zscore 标准化
            df = standardize_factors(df, factor_cols)

        # 加权求和 → 综合得分（基本面已改为排除法，不参与加权）
        for i, stock in enumerate(candidates):
            z_val = df.iloc[i].get('valuation_score', 0)
            z_flow = df.iloc[i].get('capital_flow_score', 0)
            z_sent = df.iloc[i].get('sentiment_score', 0)
            z_cog = df.iloc[i].get('cognitive_gap_score', 0)
            z_chain = df.iloc[i].get('supply_chain_score', 0)

            composite = (
                z_val * weights['valuation'] +
                z_flow * weights['capital_flow'] +
                z_cog * weights['cognitive_gap'] +
                z_sent * weights['sentiment'] +
                z_chain * weights['chain_reasoning']
            )
            stock['composite_score'] = round(composite, 4)
            stock['market_regime'] = regime

            # 信号强度折扣
            dyn = get_dynamic_win_rate()
            signal = dyn.get('signal_strength', 'normal')
            if signal == 'suspended':
                stock['composite_score'] *= 0.5   # 暂停：分数减半
            elif signal == 'degraded':
                stock['composite_score'] *= 0.75   # 降级：分数打75%

            # 热点衰减惩罚：尾声阶段热点大幅降权
            lifecycle = stock.get('hot_lifecycle', 'ferment')
            if lifecycle == 'ending':
                stock['composite_score'] *= 0.3
            elif lifecycle == 'launch' and stock.get('selection_source') == '市场热点':
                # 启动阶段热点小幅提权
                stock['composite_score'] *= 1.15

            # 热点新鲜度加分：极新鲜热点小幅提权（兼容旧逻辑）
            freshness = stock.get('hot_freshness', 50)
            if freshness >= 90 and stock.get('selection_source') == '市场热点' and lifecycle != 'launch':
                stock['composite_score'] *= 1.1

            # 映射到 style_match_score: 50 + composite*20, 钳位[0,100]
            stock['style_match_score'] = round(max(0, min(100, 50 + composite * 20)), 1)

            # 因子明细
            stock['factor_detail'] = {
                'z_valuation': round(z_val, 3),
                'z_capital_flow': round(z_flow, 3),
                'z_sentiment': round(z_sent, 3),
                'z_cognitive_gap': round(z_cog, 3),
                'z_chain_reasoning': round(z_chain, 3),
                'regime': regime,
                'weights': weights,
                'l1_pass': stock.get('l1_pass', True),
                'l1_reject_reason': stock.get('l1_reject_reason', ''),
                'industry_chain': stock.get('industry_chain', ''),
                'short_interest_ratio': stock.get('short_interest_ratio', 0),
                'volume_surge_ratio': stock.get('volume_surge_ratio', 0),
                'hot_lifecycle': lifecycle,
                'hot_lifecycle_label': stock.get('hot_lifecycle_label', ''),
            }

        # 按综合得分排序
        candidates.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        return candidates

    def _detect_market_regime(self) -> str:
        """检测市场状态: bull / bear / oscillation"""
        try:
            regime = self.market_client.detect_market_regime()
            return regime if regime in ('bull', 'bear', 'oscillation') else 'oscillation'
        except Exception as e:
            print(f"市场状态检测失败: {e}")
            return 'oscillation'

    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """市场状态 → 因子权重（基本面已改为排除法，不参与加权）

        认知差+产业链合计≤5%（低IC因子），估值+资金流+情绪占95%
        """
        if regime == 'bull':
            return {
                'valuation': 0.28,
                'capital_flow': 0.35,
                'cognitive_gap': 0.02,
                'sentiment': 0.32,
                'chain_reasoning': 0.03,
            }
        elif regime == 'bear':
            return {
                'valuation': 0.40,
                'capital_flow': 0.25,
                'cognitive_gap': 0.02,
                'sentiment': 0.29,
                'chain_reasoning': 0.04,
            }
        else:  # oscillation
            return {
                'valuation': 0.34,
                'capital_flow': 0.30,
                'cognitive_gap': 0.02,
                'sentiment': 0.30,
                'chain_reasoning': 0.04,
            }

    def _calculate_style_match(self, stock: Dict) -> float:
        """计算风格匹配度 — 博主信号仅作轻量参考(≤10分)，以因子驱动为主"""
        if not self.blogger_strategy:
            return 50.0

        score = 30.0  # 基础分

        # 博主信号仅作行业方向参考（最多+10分）
        if stock.get('selection_source') == '博主策略':
            position_weight = stock.get('position_weight', 0)
            if position_weight >= 10:
                score += 10  # 重仓方向参考
            elif position_weight >= 5:
                score += 7
            elif position_weight > 0:
                score += 3
            # position_weight=0 仅提及，不加分

        # 板块匹配（按热度分级）
        stock_sector = stock.get('sector', '')
        hot_industries = self.blogger_strategy.hot_industries
        if isinstance(hot_industries, list):
            for i, ind in enumerate(hot_industries):
                ind_name = ind.get('name', ind) if isinstance(ind, dict) else ind
                if stock_sector == ind_name:
                    # 排名越前加分越多: 第1名+15, 第2名+10, 第3名+5
                    score += max(15 - i * 5, 3)
                    break
        elif stock_sector in hot_industries:
            score += 10

        # 因子匹配（按比例映射到0-25，而非直接加）
        factor_score = stock.get('factor_score', 0)
        # factor_score范围约0-50，映射到0-25
        score += min(factor_score * 0.5, 25)

        return min(score, 100.0)

    def analyze_stock(self, code: str) -> Optional[StockPrediction]:
        """
        分析单只股票，生成预测

        Args:
            code: 股票代码

        Returns:
            StockPrediction: 预测结果
        """
        try:
            # 获取股票信息
            info = self.market_client.get_stock_info(code)
            if not info:
                return None

            name = info.get('股票简称', '')

            # 获取历史数据
            df = self.market_client.get_stock_history(code, days=30)
            if df.empty:
                return None

            # 获取技术指标
            tech = self.market_client.get_stock_technical_indicators(code)

            # 分析预测
            prediction = self._generate_prediction(code, name, df, tech)

            return prediction
        except Exception as e:
            print(f"分析股票 {code} 失败: {e}")
            return None

    def _generate_prediction(self, code: str, name: str,
                             history_df, tech: Dict) -> StockPrediction:
        """生成预测"""
        now = datetime.now()
        today = date.today()

        # 分析当日走势预测
        today_trend, today_strength, today_amplitude, today_change = \
            self._predict_today(history_df, tech)

        # 分析持股周期
        hold_days, hold_amplitude = self._predict_hold_period(history_df, tech)

        # 生成选股原因
        reasons = self._generate_selection_reasons(tech, history_df)

        # 生成风险提示
        risk = self._generate_risk_warning(tech, history_df)

        return StockPrediction(
            code=code,
            name=name,
            prediction_date=today,
            prediction_time=now,
            expected_today_trend=today_trend,
            expected_today_strength=today_strength,
            expected_today_amplitude=today_amplitude,
            expected_today_change=today_change,
            expected_hold_days=hold_days,
            expected_hold_amplitude=hold_amplitude,
            selection_reason=reasons,
            risk_warning=risk,
            technical_basis=tech
        )

    def _predict_today(self, history_df, tech: Dict) -> tuple:
        """预测当日走势"""
        # 基于技术指标判断
        trend = "震荡"
        strength = "中"
        amplitude = 3.0
        change = 0.0

        if tech.get('trend') == '上涨':
            trend = "上涨"
            change = 2.0
            amplitude = 4.0
            if tech.get('is_above_ma5') and tech.get('is_above_ma20'):
                strength = "强"
                change = 3.0
                amplitude = 5.0
        elif tech.get('trend') == '下跌':
            trend = "下跌"
            change = -1.5
            amplitude = 4.0
            strength = "弱"

        # 根据成交量调整
        volume_change = tech.get('volume_change_pct', 0)
        if volume_change > 50:  # 放量
            amplitude *= 1.3
            if trend == "上涨":
                change *= 1.2

        return trend, strength, round(amplitude, 2), round(change, 2)

    def _predict_hold_period(self, history_df, tech: Dict) -> tuple:
        """预测持股周期"""
        # 根据博主交易周期调整
        if self.blogger_strategy:
            cycle = self.blogger_strategy.trading_cycle_detail
            if '超短线' in cycle:
                hold_days = 2
            elif '短线' in cycle:
                hold_days = 5
            elif '波段' in cycle:
                hold_days = 15
            elif '中长线' in cycle:
                hold_days = 60
            else:
                hold_days = 10
        else:
            hold_days = 10

        # 根据趋势调整振幅预期
        if tech.get('trend') == '上涨':
            hold_amplitude = hold_days * 0.5  # 每天平均0.5%
        elif tech.get('trend') == '下跌':
            hold_amplitude = -hold_days * 0.3
        else:
            hold_amplitude = hold_days * 0.2

        return hold_days, round(hold_amplitude, 2)

    def _generate_selection_reasons(self, tech: Dict, history_df) -> str:
        """生成选股原因"""
        reasons = []

        if tech.get('trend') == '上涨':
            reasons.append("处于上升趋势")
        if tech.get('is_above_ma5') and tech.get('is_above_ma20'):
            reasons.append("均线多头排列")
        if tech.get('volume_change_pct', 0) > 30:
            reasons.append("成交量放大")

        if reasons:
            return "；".join(reasons)
        return "符合博主风格筛选条件"

    def _generate_risk_warning(self, tech: Dict, history_df,
                                fundamental_data: Dict = None) -> str:
        """生成风险提示 — 使用动态胜率"""
        warnings = []

        # 动态胜率
        dyn = get_dynamic_win_rate()
        wr = dyn['win_rate'] or dyn['backtest_win_rate']
        signal = dyn['signal_strength']

        # 核心风险提示：使用实际胜率
        if dyn['total_samples'] > 0:
            warnings.append(f"系统近期胜率{wr}%，单次判断失败概率{100-wr:.0f}%，请严格止损")
            fail_prob_3 = round((1 - wr/100) ** 3 * 100, 1)
            warnings.append(f"{wr}%胜率下连续3次失败概率约{fail_prob_3}%，建议单笔仓位≤总资金5%")
        else:
            warnings.append(f"系统回测胜率{wr}%（无实盘验证），非确定性预测，请严格止损")

        # 信号降级/暂停警告
        if signal == 'suspended':
            warnings.append("⚠️ 荐股信号已暂停：实盘胜率严重偏离回测，请勿跟投")
        elif signal == 'degraded':
            warnings.append("⚠️ 荐股信号降级：实盘表现弱于回测，建议减仓观望")

        # 基本面风险
        if fundamental_data:
            roe = fundamental_data.get('roe', 0) or 0
            rev_growth = fundamental_data.get('revenue_growth', 0) or 0
            if roe < 8:
                warnings.append(f"ROE={roe:.1f}%<8%，盈利能力偏弱")
            if rev_growth < 0:
                warnings.append(f"营收增长={rev_growth:.1f}%<0，基本面存疑")

        # 技术面风险
        if tech.get('amplitude_20d', 0) > 30:
            warnings.append("近期波动较大，注意风险")
        if tech.get('trend') == '下跌':
            warnings.append("当前处于下跌趋势")
        if not tech.get('is_above_ma20'):
            warnings.append("股价低于20日均线")

        if warnings:
            return "；".join(warnings)
        dyn = get_dynamic_win_rate()
        wr = dyn['win_rate'] or dyn['backtest_win_rate']
        return f"系统胜率{wr}%，非确定性预测，请严格止损"

    def save_candidates_to_db(self, candidates: List[StockCandidate], blogger_name: str = None, db_path: str = None):
        """保存候选股票到数据库"""
        conn = get_sync_db(db_path)
        cursor = conn.cursor()

        # 【修复】如果候选股票为空（网络失败），不清空旧数据
        if not candidates:
            print(f"候选股票为空，保留旧数据（博主: {blogger_name or '默认'}）")
            conn.close()
            return

        # 【修复】质量门槛：新候选数量远少于旧数据时保留旧数据
        # 避免因API临时故障选出少量垃圾股覆盖正常数据
        try:
            if blogger_name:
                cursor.execute("SELECT COUNT(*) FROM stock_pool WHERE blogger_name = ? AND status = 'active'", (blogger_name,))
            else:
                cursor.execute("SELECT COUNT(*) FROM stock_pool WHERE status = 'active'")
            old_count = cursor.fetchone()[0]
            if old_count > 0 and len(candidates) < old_count * 0.3:
                # 新候选不到旧数据的30%，可能是API故障
                # 检查新候选质量：如果大部分是*ST/ST股，说明数据源异常
                st_count = sum(1 for c in candidates if 'ST' in c.name or '*ST' in c.name)
                if st_count > len(candidates) * 0.5:
                    print(f"新候选仅{len(candidates)}只（旧{old_count}只），且{st_count}只为ST股，疑似数据源异常，保留旧数据")
                    conn.close()
                    return
                elif len(candidates) < 5:
                    print(f"新候选仅{len(candidates)}只（旧{old_count}只），数量过少，保留旧数据")
                    conn.close()
                    return
        except Exception:
            pass

        # 清空所有旧数据（不再按博主区分，避免残留垃圾数据）
        cursor.execute("DELETE FROM stock_pool WHERE status = 'active'")

        # 插入新数据
        for c in candidates:
            # 转换技术指标为可序列化格式
            tech_indicators = {}
            for k, v in c.technical_indicators.items():
                if isinstance(v, (bool, np.bool_)):
                    tech_indicators[k] = bool(v)
                elif isinstance(v, (np.integer, np.floating)):
                    tech_indicators[k] = float(v)
                else:
                    tech_indicators[k] = v

            cursor.execute("""
                INSERT INTO stock_pool
                (stock_code, stock_name, sector, concept, price, change_pct,
                 style_match_score, selection_source, position_weight, selection_reasons,
                 technical_indicators, added_date, blogger_name, market,
                 fundamental_score, valuation_score, capital_flow_score, sentiment_score,
                 pe_percentile, pb_percentile, market_regime, factor_detail,
                 cognitive_gap_score, supply_chain_score, short_interest_ratio,
                 volume_surge_ratio, industry_chain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                c.code, c.name, c.sector, c.concept, c.price, c.change_pct,
                c.style_match_score, c.selection_source, c.position_weight,
                json.dumps(c.selection_reasons, ensure_ascii=False),
                json.dumps(tech_indicators, ensure_ascii=False), date.today(),
                blogger_name if c.selection_source in ('基本面驱动', '量化策略') else None,
                c.market,
                c.fundamental_score, c.valuation_score, c.capital_flow_score, c.sentiment_score,
                c.pe_percentile, c.pb_percentile, c.market_regime,
                json.dumps(c.factor_detail, ensure_ascii=False) if isinstance(c.factor_detail, dict) else str(c.factor_detail),
                c.cognitive_gap_score, c.supply_chain_score, c.short_interest_ratio,
                c.volume_surge_ratio, c.industry_chain
            ))

        conn.commit()
        conn.close()
        print(f"已保存 {len(candidates)} 只候选股票到数据库（博主: {blogger_name or '默认'}）")

    def calculate_portfolio_risk(self, candidates: List[StockCandidate],
                                 total_capital: float = 1000000.0) -> Optional[PortfolioRiskReport]:
        """计算候选股票池的组合风险

        Args:
            candidates: 候选股票列表
            total_capital: 总资金（默认100万）

        Returns:
            PortfolioRiskReport，如果数据不足返回None
        """
        if not candidates:
            return None

        positions = {}
        price_data = {}
        for c in candidates:
            if not c.price or c.price <= 0:
                continue
            weight = c.position_weight if c.position_weight > 0 else 5.0
            position_value = total_capital * weight / 100
            positions[c.code] = position_value
            try:
                df = self.market_client.get_stock_history(c.code, days=60)
                if not df.empty and len(df) > 20:
                    price_data[c.code] = df
            except Exception:
                continue

        if len(price_data) < 2:
            print("[风险度量] 有效历史数据不足，无法计算组合风险")
            return None

        analyzer = PortfolioRiskAnalyzer(confidence_level=0.95)
        report = analyzer.analyze_portfolio(positions, price_data)
        return report


def generate_risk_report(blogger_name: str = None, total_capital: float = 1000000.0,
                         db_path: str = None) -> Optional[Dict]:
    """生成组合风险报告并保存到数据库

    Args:
        blogger_name: 博主名称
        total_capital: 总资金
        db_path: 数据库路径

    Returns:
        Dict格式的风险报告摘要，数据不足返回None
    """
    pool = get_stock_pool_by_blogger(blogger_name, db_path=db_path)
    if not pool:
        print(f"[风险度量] 股票池为空（博主: {blogger_name or '全部'}）")
        return None

    market_client = get_market_client()
    positions = {}
    price_data = {}
    import time as _time
    _history_cache = {}
    for stock in pool:
        code = stock['code']
        price = stock.get('price', 0)
        if not price or price <= 0:
            continue
        weight = stock.get('position_weight', 0)
        if weight <= 0:
            weight = 5.0
        position_value = total_capital * weight / 100
        positions[code] = position_value
        cache_key = f"{code}_60"
        if cache_key in _history_cache:
            price_data[code] = _history_cache[cache_key]
            continue
        try:
            df = market_client.get_stock_history(code, days=60)
            if not df.empty and len(df) > 20:
                price_data[code] = df
                _history_cache[cache_key] = df
        except Exception:
            continue

    if len(price_data) < 2:
        print("[风险度量] 有效历史数据不足")
        return None

    analyzer = PortfolioRiskAnalyzer(confidence_level=0.95)
    report = analyzer.analyze_portfolio(positions, price_data)

    # 保存到数据库
    _save_risk_report_to_db(report, blogger_name, db_path)

    return _risk_report_to_dict(report)


def _save_risk_report_to_db(report: PortfolioRiskReport, blogger_name: str = None,
                             db_path: str = None):
    """保存风险报告到数据库"""
    conn = get_sync_db(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date DATE NOT NULL,
            blogger_name TEXT,
            portfolio_value REAL,
            var_value REAL,
            var_absolute REAL,
            cvar_value REAL,
            confidence_level REAL,
            risk_level TEXT,
            risk_score REAL,
            risk_rating TEXT,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            max_drawdown REAL,
            calmar_ratio REAL,
            stress_test JSON,
            risk_contribution JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    var_info = report.portfolio_var
    stress_json = json.dumps(
        {k: round(v, 2) for k, v in report.stress_test_results.items()},
        ensure_ascii=False
    ) if report.stress_test_results else '{}'
    contrib_json = json.dumps(
        {k: round(v, 2) for k, v in report.risk_contribution.items()},
        ensure_ascii=False
    ) if report.risk_contribution else '{}'

    cursor.execute("""
        INSERT INTO risk_reports
        (report_date, blogger_name, portfolio_value, var_value, var_absolute,
         cvar_value, confidence_level, risk_level, risk_score, risk_rating,
         sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
         stress_test, risk_contribution)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        date.today().isoformat(), blogger_name, report.portfolio_value,
        var_info.var_value if var_info else 0,
        var_info.var_absolute if var_info else 0,
        var_info.cvar if var_info else 0,
        var_info.confidence_level if var_info else 0.95,
        var_info.risk_level if var_info else '未知',
        var_info.risk_score if var_info else 50,
        report.risk_rating,
        report.sharpe_ratio, report.sortino_ratio,
        report.max_drawdown, report.calmar_ratio,
        stress_json, contrib_json
    ))

    conn.commit()
    conn.close()
    print(f"[风险度量] 报告已保存（博主: {blogger_name or '全部'}）")


def _risk_report_to_dict(report: PortfolioRiskReport) -> Dict:
    """将PortfolioRiskReport转换为可序列化的Dict"""
    var_info = report.portfolio_var
    result = {
        'portfolio_value': report.portfolio_value,
        'calculation_time': report.calculation_time.isoformat(),
        'risk_rating': report.risk_rating,
        'overall_risk_score': report.overall_risk_score,
        'sharpe_ratio': report.sharpe_ratio,
        'sortino_ratio': report.sortino_ratio,
        'max_drawdown': report.max_drawdown,
        'calmar_ratio': report.calmar_ratio,
        'stress_test_results': {k: round(v, 2) for k, v in report.stress_test_results.items()},
        'risk_contribution': {k: round(v, 2) for k, v in report.risk_contribution.items()},
        'marginal_var': {k: round(v, 4) for k, v in report.marginal_var.items()},
    }
    if var_info:
        result['portfolio_var'] = {
            'var_value': var_info.var_value,
            'var_absolute': var_info.var_absolute,
            'cvar': var_info.cvar,
            'confidence_level': var_info.confidence_level,
            'time_horizon': var_info.time_horizon,
            'method': var_info.method,
            'risk_level': var_info.risk_level,
            'risk_score': var_info.risk_score,
            'mean_return': var_info.mean_return,
            'std_return': var_info.std_return,
            'skewness': var_info.skewness,
            'kurtosis': var_info.kurtosis,
        }
    result['individual_risks'] = {}
    for code, var in report.individual_risks.items():
        result['individual_risks'][code] = {
            'var_value': var.var_value,
            'cvar': var.cvar,
            'risk_level': var.risk_level,
        }
    if not report.correlation_matrix.empty:
        result['correlation_matrix'] = report.correlation_matrix.round(2).to_dict()
    return result


def get_risk_report_from_db(blogger_name: str = None, days: int = 30,
                             db_path: str = None) -> List[Dict]:
    """从数据库获取历史风险报告

    Args:
        blogger_name: 博主名称
        days: 获取最近多少天
        db_path: 数据库路径

    Returns:
        List[Dict]: 风险报告列表
    """
    if db_path is None:
        db_path = 'data/weibo.db'

    conn = get_sync_db(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_reports'")
    if not cursor.fetchone():
        conn.close()
        return []

    start_date = (date.today() - timedelta(days=days)).isoformat()

    if blogger_name:
        cursor.execute("""
            SELECT report_date, blogger_name, portfolio_value, var_value, var_absolute,
                   cvar_value, confidence_level, risk_level, risk_score, risk_rating,
                   sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
                   stress_test, risk_contribution
            FROM risk_reports
            WHERE blogger_name = ? AND report_date >= ?
            ORDER BY report_date DESC
        """, (blogger_name, start_date))
    else:
        cursor.execute("""
            SELECT report_date, blogger_name, portfolio_value, var_value, var_absolute,
                   cvar_value, confidence_level, risk_level, risk_score, risk_rating,
                   sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
                   stress_test, risk_contribution
            FROM risk_reports
            WHERE report_date >= ?
            ORDER BY report_date DESC
        """, (start_date,))

    results = []
    for row in cursor.fetchall():
        results.append({
            'report_date': row[0],
            'blogger_name': row[1],
            'portfolio_value': row[2],
            'var_value': row[3],
            'var_absolute': row[4],
            'cvar_value': row[5],
            'confidence_level': row[6],
            'risk_level': row[7],
            'risk_score': row[8],
            'risk_rating': row[9],
            'sharpe_ratio': row[10],
            'sortino_ratio': row[11],
            'max_drawdown': row[12],
            'calmar_ratio': row[13],
            'stress_test': json.loads(row[14]) if row[14] else {},
            'risk_contribution': json.loads(row[15]) if row[15] else {},
        })

    conn.close()
    return results


def _get_current_pool_status(blogger_name: str = None) -> Dict:
    """暂停时获取当前股票池状态（只读，不刷新），仅供内部监控"""
    try:
        pool = get_stock_pool_by_blogger(blogger_name)
        dyn = get_dynamic_win_rate()
        return {
            'candidates': pool,
            'dynamic_win_rate': dyn,
            'signal_strength': dyn.get('signal_strength', 'suspended'),
            'lock_reason': dyn.get('lock_reason', ''),
            'warning': dyn.get('warning', ''),
            'pool_frozen': True,
        }
    except Exception:
        return {'candidates': [], 'pool_frozen': True, 'signal_strength': 'suspended'}


def run_stock_selector(blogger_name: str = None):
    """运行选股引擎

    Args:
        blogger_name: 博主名称，如果为None则使用第一个有数据的博主
    """
    # 信号强度检查
    dyn = get_dynamic_win_rate()
    signal = dyn.get('signal_strength', 'normal')
    lock_reason = dyn.get('lock_reason', '')
    if signal == 'suspended':
        print(f"🔴 荐股信号已暂停锁定: {dyn.get('warning', '实盘胜率严重偏离回测')}")
        if lock_reason:
            print(f"   锁定原因: {lock_reason}")
        print("   系统暂停对外荐股输出，仅保留内部监控功能")
        print("   解锁需人工审计并确认系统问题已修复")
        # 暂停时仍返回当前池信息但不刷新，用于内部监控
        return _get_current_pool_status(blogger_name)
    elif signal == 'degraded':
        print(f"🟡 荐股信号降级: {dyn.get('warning', '实盘表现弱于回测')}")
        print("   推荐数量将减半，建议减仓观望")

    # 从数据库加载微博数据
    conn = get_sync_db()
    cursor = conn.cursor()

    # 【改进】获取所有博主列表
    cursor.execute("SELECT DISTINCT nickname FROM bloggers WHERE nickname IS NOT NULL")
    all_bloggers = [row[0] for row in cursor.fetchall()]
    print(f"数据库中的博主列表: {all_bloggers}")

    # 获取微博数据
    cursor.execute("""
        SELECT b.nickname, w.content, w.publish_time
        FROM weibo_posts w
        LEFT JOIN bloggers b ON w.blogger_id = b.id
        ORDER BY w.crawl_time DESC
    """)

    posts = []
    for row in cursor.fetchall():
        posts.append({
            'content': row[1],
            'publish_time': str(row[2]) if row[2] else '',
            'blogger': row[0]
        })

    # 如果没有指定博主，使用第一个有数据的博主
    if not blogger_name:
        bloggers_with_posts = list(set(p.get('blogger') for p in posts if p.get('blogger')))
        if bloggers_with_posts:
            blogger_name = bloggers_with_posts[0]
            print(f"未指定博主，使用第一个有数据的博主: {blogger_name}")
        else:
            print("没有找到任何博主数据，使用默认策略")
            blogger_name = None

    conn.close()

    # 分析博主策略
    analyzer = StockAnalyzer()

    if blogger_name:
        blogger_posts = [p for p in posts if p.get('blogger') == blogger_name]
        print(f"博主 {blogger_name} 的微博数量: {len(blogger_posts)}")
        strategy = analyzer.analyze_posts(blogger_posts, blogger_name)
    else:
        # 使用默认策略
        strategy = None

    print("=" * 70)
    print(f"行业方向参考（博主: {blogger_name or '默认'}）— 博主信号仅作参考，非选股主导")
    print("=" * 70)
    if strategy:
        print(f"关注板块: {strategy.hot_industries}")
        print(f"选股因子: {list(strategy.selection_factors.keys())}")
        print(f"交易周期: {strategy.trading_cycle_detail}")
    else:
        print("使用默认选股策略")

    # 创建选股引擎
    selector = StockSelector(strategy)

    # 生成候选股票池
    print("\n" + "=" * 70)
    print("生成候选股票池")
    print("=" * 70)

    candidates = selector.generate_candidate_pool(max_stocks=30)

    # 显示大盘择时结果
    timing_info = getattr(selector, '_last_timing', None)
    if timing_info:
        t_cfg = timing_info['config']
        print(f"\n{'='*70}")
        print(f"[大盘择时] {t_cfg['label']} | {timing_info['reason']}")
        print(f"  股票池比例: {t_cfg['pool_ratio']:.0%} | 总仓位上限: {t_cfg['max_position']:.0%} | 单票上限: {t_cfg['single_position']:.0%}")
        if not t_cfg['allow_hot_stocks']:
            print(f"  ⚠️ 纯热点题材已屏蔽 | 准入门槛: 综合分≥{t_cfg['min_composite_score']}")
        print(f"{'='*70}")

    # 信号强度门控
    dyn = get_dynamic_win_rate()
    signal = dyn.get('signal_strength', 'normal')
    if signal == 'suspended' and len(candidates) > 5:
        # 暂停状态：仅保留前5只观察级股票
        candidates = candidates[:5]
        print(f"\n[信号暂停] 仅保留前5只观察级股票")
    elif signal == 'degraded' and len(candidates) > 15:
        # 降级状态：推荐数量减半
        candidates = candidates[:len(candidates)//2]
        print(f"\n[信号降级] 推荐数量减半至{len(candidates)}只")

    print(f"\n共筛选出 {len(candidates)} 只候选股票:")
    print("-" * 70)
    for i, c in enumerate(candidates, 1):
        source_tag = "【基本面驱动】" if c.selection_source == '基本面驱动' else ("【量化策略】" if c.selection_source == '量化策略' else "【市场热点】")
        print(f"{i}. {source_tag} {c.code} {c.name}")
        print(f"   板块: {c.sector} | 概念: {c.concept}")
        print(f"   价格: {c.price}元 | 涨跌: {c.change_pct:+.2f}%")
        print(f"   风格匹配度: {c.style_match_score:.1f}分")
        if c.position_weight > 0:
            print(f"   仓位权重: {c.position_weight}%")
        print(f"   选股原因: {'; '.join(c.selection_reasons)}")
        print()

    # 组合构建：仓位管理 + 风险预算 + 板块分散
    from app.selector.portfolio_constructor import PortfolioConstructor
    constructor = PortfolioConstructor()
    regime = selector._detect_market_regime()
    candidate_dicts = [
        {
            'code': c.code, 'name': c.name, 'sector': c.sector,
            'concept': c.concept, 'price': c.price, 'change_pct': c.change_pct,
            'style_match_score': c.style_match_score,
            'selection_source': c.selection_source,
            'fundamental_score': c.fundamental_score,
            'capital_flow_score': c.capital_flow_score,
            'valuation_score': c.valuation_score,
            'technical_indicators': c.technical_indicators,
            'reasons': c.selection_reasons,
            'composite_score': c.style_match_score,  # 近似
        }
        for c in candidates
    ]
    portfolio = constructor.construct(candidate_dicts, market_regime=regime)

    # 择时仓位上限覆盖：如果择时要求更严格，截断组合仓位
    if timing_info:
        timing_max_pos = timing_info['config']['max_position'] * 100  # 转为百分比
        timing_single = timing_info['config']['single_position'] * 100
        if portfolio.total_position_pct > timing_max_pos:
            print(f"\n[择时截仓] 总仓位 {portfolio.total_position_pct:.1f}% → {timing_max_pos:.1f}% (择时上限)")
            # 按仓位从大到小削减，直到总仓位满足择时上限
            for pos in sorted(portfolio.positions, key=lambda p: -p.position_pct):
                if portfolio.total_position_pct <= timing_max_pos:
                    break
                cut = min(pos.position_pct, portfolio.total_position_pct - timing_max_pos)
                pos.position_pct -= cut
                portfolio.total_position_pct -= cut
            # 移除仓位为0的
            portfolio.positions = [p for p in portfolio.positions if p.position_pct > 0]
        # 单票仓位上限
        for pos in portfolio.positions:
            if pos.position_pct > timing_single:
                pos.position_pct = timing_single

    print("\n" + "=" * 70)
    print("组合构建结果")
    print("=" * 70)
    # 自适应策略信息
    from app.selector.adaptive_strategy import AdaptiveStrategyManager
    asm = AdaptiveStrategyManager()
    guidance = asm.get_position_guidance(regime, dyn.get('signal_strength', 'normal'))
    print(f"策略: {guidance['strategy_name']} | 止损{guidance['stop_loss_pct']}% | 止盈+{guidance['take_profit_pct']}% | 移动止损{guidance['trailing_stop_pct']}%")
    print(f"持仓周期: {guidance['default_hold_days']}天 | 最大总仓位: {guidance['max_total_position']}%")
    print(f"说明: {guidance['description']}")
    print("-" * 70)
    for i, pos in enumerate(portfolio.positions, 1):
        grade_icon = {'A': '🟢', 'B': '🔵', 'C': '🟡', 'D': '🔴'}.get(pos.position_grade, '⚪')
        print(f"{grade_icon} {i}. [{pos.position_grade}级] {pos.code} {pos.name}")
        print(f"   建议仓位: {pos.position_pct:.1f}% | 信号评分: {pos.signal_score:.1f} | 波动率: {pos.volatility:.1f}%")
        print(f"   风险预算: {pos.risk_budget:.2f}% | 板块: {pos.sector}")
        print()
    print(f"总仓位: {portfolio.total_position_pct:.1f}% | 覆盖行业: {portfolio.num_sectors} | 最大板块仓位: {portfolio.max_sector_pct:.1f}%")
    print(f"组合CVaR: {portfolio.portfolio_cvar:.2f}% | 集中度评分: {portfolio.concentration_score:.1f}")
    if portfolio.risk_warning:
        print(f"⚠️  {portfolio.risk_warning}")

    # 保存组合到数据库
    constructor.save_portfolio_to_db(portfolio, blogger_name=blogger_name)

    # 交易决策：买入时机评估
    print("\n" + "=" * 70)
    print("交易决策引擎")
    print("=" * 70)
    from app.selector.trading_decision import TradingDecisionEngine
    engine = TradingDecisionEngine()
    candidate_codes = [c.code for c in candidates]
    decision = engine.generate_decision_report(
        candidate_codes=candidate_codes,
        regime=regime,
    )
    if decision.buy_signals:
        print(f"\n买入信号({len(decision.buy_signals)}只, 时机评分>{engine.TIMING_SCORE_THRESHOLD}):")
        for s in sorted(decision.buy_signals, key=lambda x: -x.timing_score):
            icon = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(s.confidence, '⚪')
            print(f"  {icon} {s.code} {s.name} [{s.signal_type}] 时机{s.timing_score:.0f}分")
            print(f"     入场{s.entry_price:.2f} | 止损{s.stop_loss_price:.2f} | 止盈{s.take_profit_price:.2f} | 仓位{s.position_pct:.1f}%")
            print(f"     {'; '.join(s.reasons)}")
    else:
        print("\n暂无满足条件的买入信号(时机评分不足)")

    if decision.portfolio_actions:
        print(f"\n操作建议:")
        for action in decision.portfolio_actions:
            print(f"  → {action}")

    if decision.risk_warnings:
        for w in decision.risk_warnings:
            print(f"  ⚠️  {w}")

    # 保存决策报告
    engine.save_decision_report(decision, blogger_name=blogger_name)

    # 保存到数据库（按博主区分）
    selector.save_candidates_to_db(candidates, blogger_name=blogger_name)

    return candidates


def get_stock_pool_by_blogger(blogger_name: str = None, db_path: str = None) -> List[Dict]:
    """获取指定博主的股票池

    Args:
        blogger_name: 博主名称，如果为None则获取所有
        db_path: 数据库路径

    Returns:
        List[Dict]: 股票池列表
    """
    if db_path is None:
        db_path = 'data/weibo.db'

    conn = get_sync_db(db_path)
    cursor = conn.cursor()

    if blogger_name:
        cursor.execute("""
            SELECT stock_code, stock_name, sector, concept, price, change_pct,
                   style_match_score, selection_source, position_weight, selection_reasons, added_date
            FROM stock_pool
            WHERE blogger_name = ? AND status = 'active'
            ORDER BY
                CASE WHEN selection_source = '博主策略' THEN 0 ELSE 1 END,
                position_weight DESC,
                style_match_score DESC
        """, (blogger_name,))
    else:
        cursor.execute("""
            SELECT stock_code, stock_name, sector, concept, price, change_pct,
                   style_match_score, selection_source, position_weight, selection_reasons, added_date, blogger_name
            FROM stock_pool
            WHERE status = 'active'
            ORDER BY blogger_name,
                CASE WHEN selection_source = '基本面驱动' THEN 0 ELSE 1 END,
                position_weight DESC,
                style_match_score DESC
        """)

    results = []
    for row in cursor.fetchall():
        results.append({
            'code': row[0],
            'name': row[1],
            'sector': row[2],
            'concept': row[3],
            'price': row[4],
            'change_pct': row[5],
            'style_match_score': row[6],
            'selection_source': row[7] if len(row) > 7 else '市场热点',
            'position_weight': row[8] if len(row) > 8 else 0,
            'selection_reasons': json.loads(row[9]) if len(row) > 9 and row[9] else [],
            'added_date': row[10] if len(row) > 10 else None,
            'blogger_name': row[11] if len(row) > 11 else blogger_name
        })

    conn.close()
    return results


def update_all_blogger_pools(db_path: str = None):
    """更新所有博主的股票池

    Args:
        db_path: 数据库路径
    """
    import sqlite3

    if db_path is None:
        db_path = 'data/weibo.db'

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有博主
    cursor.execute("SELECT DISTINCT nickname FROM bloggers WHERE nickname IS NOT NULL")
    bloggers = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"开始更新所有博主的股票池，共 {len(bloggers)} 个博主")

    for blogger in bloggers:
        print(f"\n{'='*70}")
        print(f"更新博主: {blogger}")
        print('='*70)
        try:
            run_stock_selector(blogger_name=blogger)
        except Exception as e:
            print(f"更新博主 {blogger} 失败: {e}")

    print("\n所有博主股票池更新完成")


def get_dynamic_win_rate(db_path: str = None, recent_days: int = 30) -> Dict:
    """动态胜率计算 — 从daily_reviews表实时滚动计算，不再硬编码

    Args:
        db_path: 数据库路径
        recent_days: 滚动窗口天数

    Returns:
        Dict: {
            "win_rate": float,           # 实际滚动胜率(0-100)
            "total_samples": int,        # 总预测样本数
            "total_correct": int,        # 正确预测数
            "backtest_win_rate": 64.0,   # 回测胜率(固定参考值)
            "deviation": float,          # 实盘与回测偏差
            "signal_strength": str,      # "normal"|"degraded"|"suspended"
            "consecutive_low_days": int, # 连续低胜率天数(<50%)
            "warning": str or None,      # 风险警告
        }
    """
    BACKTEST_WIN_RATE = 64.0  # 回测基准胜率，仅作参考（已作废，待回测审计后更新）
    LOW_WIN_RATE_THRESHOLD = 45.0  # 降至45%（原50%）
    CONSECUTIVE_LOW_LIMIT = 10     # 连续10天低胜率→永久暂停（原5天）
    DEVIATION_SUSPEND_THRESHOLD = 15.0
    DIRECTION_ACCURACY_SUSPEND = 0.30  # 方向准确率<30%→永久暂停

    if db_path is None:
        db_path = 'data/weibo.db'

    conn = get_sync_db(db_path)
    cursor = conn.cursor()

    try:
        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_reviews'")
        if not cursor.fetchone():
            return {
                'win_rate': 0,
                'total_samples': 0,
                'total_correct': 0,
                'backtest_win_rate': BACKTEST_WIN_RATE,
                'deviation': BACKTEST_WIN_RATE,
                'signal_strength': 'suspended',
                'consecutive_low_days': 0,
                'warning': '无实盘数据，系统胜率未知，请勿直接跟投',
            }

        # 滚动窗口胜率
        cursor.execute("""
            SELECT review_date, accuracy_rate, total_predictions, correct_predictions
            FROM daily_reviews
            ORDER BY review_date DESC
            LIMIT ?
        """, (recent_days,))
        rows = cursor.fetchall()

        if not rows:
            return {
                'win_rate': 0,
                'total_samples': 0,
                'total_correct': 0,
                'backtest_win_rate': BACKTEST_WIN_RATE,
                'deviation': BACKTEST_WIN_RATE,
                'signal_strength': 'suspended',
                'consecutive_low_days': 0,
                'warning': '无实盘数据，系统胜率未知，请勿直接跟投',
            }

        total_samples = sum(r[2] for r in rows if r[2])
        total_correct = sum(r[3] for r in rows if r[3])
        win_rate = round(total_correct / total_samples * 100, 1) if total_samples > 0 else 0

        # 计算连续低胜率天数
        consecutive_low = 0
        for r in rows:
            if (r[1] or 0) < LOW_WIN_RATE_THRESHOLD:
                consecutive_low += 1
            else:
                break

        # 偏差计算
        deviation = round(BACKTEST_WIN_RATE - win_rate, 1)

        # 信号强度判定（严格规则，人工解锁需审计）
        signal_strength = 'normal'
        warning = None
        lock_reason = None  # 锁定原因

        # 紧急暂停规则1：方向准确率<30%
        try:
            from app.selector.recommendation_feedback import RecommendationFeedback
            fb_q = RecommendationFeedback(db_path)
            latest_q = fb_q.get_latest_quality_report()
            if latest_q and latest_q.get('direction_accuracy', 1) < DIRECTION_ACCURACY_SUSPEND:
                signal_strength = 'suspended'
                lock_reason = f'方向准确率{latest_q["direction_accuracy"]:.0%}<30%，系统预测方向严重失效，暂停锁定'
                warning = lock_reason
        except Exception:
            pass

        # 紧急暂停规则2：连续10天胜率<45%
        if lock_reason is None and consecutive_low >= CONSECUTIVE_LOW_LIMIT:
            signal_strength = 'suspended'
            lock_reason = f'连续{consecutive_low}天胜率低于{LOW_WIN_RATE_THRESHOLD}%，永久暂停锁定（需人工审计解锁）'
            warning = lock_reason

        # 紧急暂停规则3：实盘与回测偏差>15%
        if lock_reason is None and deviation > DEVIATION_SUSPEND_THRESHOLD:
            signal_strength = 'suspended'
            lock_reason = f'实盘胜率({win_rate}%)与回测({BACKTEST_WIN_RATE}%)偏差>{DEVIATION_SUSPEND_THRESHOLD}%，回测失真，暂停锁定'
            warning = lock_reason

        # 降级规则1：连续3天低胜率
        if lock_reason is None and consecutive_low >= 3:
            signal_strength = 'degraded'
            warning = f'近{consecutive_low}天胜率低于{LOW_WIN_RATE_THRESHOLD}%，荐股信号降级'

        # 降级规则2：偏差>10%
        if lock_reason is None and deviation > 10:
            signal_strength = 'degraded'
            warning = f'实盘胜率({win_rate}%)低于回测({BACKTEST_WIN_RATE}%)超10%，信号降级'

        # 强制降级：模拟交易未满90天
        try:
            sim = get_simulation_status(blogger_name=None, db_path=db_path)
            if sim.get('days', 0) < 90:
                if signal_strength == 'normal':
                    signal_strength = 'degraded'
                    warning = (warning or '') + f'；实盘模拟仅{sim["days"]}天/需90天，强制降级'
        except Exception:
            pass

        # 质量反馈闭环叠加
        try:
            from app.selector.recommendation_feedback import RecommendationFeedback
            fb = RecommendationFeedback(db_path)
            latest = fb.get_latest_quality_report()
            if latest:
                if latest.get('system_health') == 'suspended':
                    signal_strength = 'suspended'
                    lock_reason = '推荐质量严重不足，暂停锁定'
                    warning = (warning or '') + '；推荐质量严重不足，信号暂停锁定'
                elif latest.get('system_health') == 'degraded':
                    if signal_strength == 'normal':
                        signal_strength = 'degraded'
                        warning = (warning or '') + '；推荐质量偏低，信号降级'
                    elif signal_strength == 'degraded':
                        signal_strength = 'suspended'
                        lock_reason = '质量持续偏差，升级为暂停'
                        warning = (warning or '') + '；质量持续偏差，升级为暂停锁定'
        except Exception:
            pass

        return {
            'win_rate': win_rate,
            'total_samples': total_samples,
            'total_correct': total_correct,
            'backtest_win_rate': BACKTEST_WIN_RATE,
            'deviation': deviation,
            'signal_strength': signal_strength,
            'consecutive_low_days': consecutive_low,
            'warning': warning,
            'lock_reason': lock_reason,
        }
    finally:
        conn.close()


def get_simulation_status(blogger_name: str = None, db_path: str = None) -> Dict:
    """检查实盘模拟验证状态

    Args:
        blogger_name: 博主名称
        db_path: 数据库路径

    Returns:
        Dict: {"status": "not_started"|"in_progress"|"3m_completed",
               "days": 0, "win_rate": 0, "max_drawdown": 0}
    """
    if db_path is None:
        db_path = 'data/weibo.db'

    conn = get_sync_db(db_path)
    cursor = conn.cursor()

    try:
        # 检查是否有模拟交易记录
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='simulated_orders'")
        has_orders = cursor.fetchone() is not None

        # 检查复盘记录
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_reviews'")
        has_reviews = cursor.fetchone() is not None

        if not has_orders and not has_reviews:
            return {
                'status': 'not_started',
                'days': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'required_days': 90,
                'warning': '系统尚未开始实盘模拟验证，当前荐股仅供参考，请勿直接跟投',
            }

        # 查询模拟交易天数
        days = 0
        win_rate = 0
        max_drawdown = 0

        if has_reviews:
            query = """
                SELECT COUNT(DISTINCT review_date) as days,
                       AVG(accuracy_rate) as avg_accuracy,
                       MIN(0) as max_dd
                FROM daily_reviews
            """
            params = []
            if blogger_name:
                query += " WHERE blogger_name = ?"
                params.append(blogger_name)

            cursor.execute(query, params)
            row = cursor.fetchone()
            if row and row[0]:
                days = row[0]
                win_rate = round(row[1] or 0, 1)

        # 查询风险报告中的最大回撤
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_reports'")
        if cursor.fetchone():
            cursor.execute("SELECT MAX(max_drawdown) FROM risk_reports")
            dd_row = cursor.fetchone()
            if dd_row and dd_row[0]:
                max_drawdown = round(abs(dd_row[0]), 2)

        if days >= 90:
            return {
                'status': '3m_completed',
                'days': days,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'required_days': 90,
                'warning': None,
            }
        elif days > 0:
            return {
                'status': 'in_progress',
                'days': days,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'required_days': 90,
                'warning': f'系统实盘模拟仅{days}天，未达3个月({90}天)验证要求，请勿直接跟投',
            }
        else:
            return {
                'status': 'not_started',
                'days': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'required_days': 90,
                'warning': '系统尚未开始实盘模拟验证，当前荐股仅供参考，请勿直接跟投',
            }
    finally:
        conn.close()


if __name__ == "__main__":
    run_stock_selector()
