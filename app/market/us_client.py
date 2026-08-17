"""
美股市场数据客户端 - 基本面分析 + 预测
"""
from typing import Dict, List, Optional, Any
import traceback

from .base_client import MarketClientBase, MarketType
import pandas as pd


class USMarketClient(MarketClientBase):
    """美股市场数据客户端（基本面分析+预测）"""
    market_type = MarketType.US

    def __init__(self):
        super().__init__()
        self.market_name = "美股"
        self._sina_client = None

    def _get_sina_client(self):
        """懒加载新浪客户端，用于实时行情和历史数据"""
        if self._sina_client is None:
            from .sina_tencent_client import SinaTencentUSClient
            self._sina_client = SinaTencentUSClient()
        return self._sina_client

    def get_stock_price(self, code: str) -> Dict[str, Any]:
        return self._get_sina_client().get_stock_price(code)

    def get_realtime_quotes(self, limit: int = 100) -> List[Dict[str, Any]]:
        gainers = self.fetch_top_gainers(limit=limit)
        return gainers if gainers else []

    def get_stock_history(self, code: str, days: int = 30, adjust: str = "qfq") -> pd.DataFrame:
        return self._get_sina_client().get_stock_history(code, days, adjust)

    def get_market_overview(self) -> Dict[str, Any]:
        try:
            gainers = self.fetch_top_gainers(limit=5)
            return {
                'market': 'US',
                'top_gainers': [{'code': g['code'], 'name': g['name'],
                                  'change_pct': g.get('change_pct', 0)} for g in gainers],
            }
        except Exception:
            return {'market': 'US'}

    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        return self._get_sina_client().search_stock(keyword)

    def get_index_quote(self, index_code: str = "") -> Dict[str, Any]:
        return self._get_sina_client().get_index_quote(index_code)

    def get_stock_technical_indicators(self, code: str) -> Dict[str, Any]:
        return self._get_sina_client().get_stock_technical_indicators(code)

    def fetch_top_gainers(self, limit: int = 10) -> List[Dict]:
        """获取美股涨幅排行榜（保留兼容）"""
        try:
            import akshare as ak
            df = ak.stock_us_spot_em()
            if df.empty:
                return []

            df = df.sort_values(by='涨跌幅', ascending=False)
            results = []
            for _, row in df.head(limit).iterrows():
                results.append({
                    'code': str(row.get('代码', '')),
                    'name': str(row.get('名称', '')),
                    'price': float(row.get('最新价', 0)),
                    'change_pct': float(row.get('涨跌幅', 0)),
                    'change_amt': float(row.get('涨跌额', 0)),
                    'volume': float(row.get('成交量', 0)),
                    'amount': float(row.get('成交额', 0)),
                    'amplitude': float(row.get('振幅', 0)),
                    'high': float(row.get('最高', 0)),
                    'low': float(row.get('最低', 0)),
                    'open': float(row.get('今开', 0)),
                    'prev_close': float(row.get('昨收', 0)),
                    'market': 'US',
                })
            return results
        except Exception as e:
            print(f"获取美股涨幅排行失败: {e}")
            return []

    def fetch_stock_data(self, code: str) -> Optional[Dict]:
        """获取美股个股实时数据"""
        try:
            import akshare as ak
            df = ak.stock_us_spot_em()
            row = df[df['代码'] == code]
            if row.empty:
                return None
            r = row.iloc[0]
            return {
                'code': str(r.get('代码', code)),
                'name': str(r.get('名称', '')),
                'price': float(r.get('最新价', 0)),
                'change_pct': float(r.get('涨跌幅', 0)),
                'volume': float(r.get('成交量', 0)),
                'amount': float(r.get('成交额', 0)),
                'high': float(r.get('最高', 0)),
                'low': float(r.get('最低', 0)),
                'open': float(r.get('今开', 0)),
                'prev_close': float(r.get('昨收', 0)),
                'pe_ratio': float(r.get('市盈率-动态', 0)) if r.get('市盈率-动态') else 0,
                'pb_ratio': float(r.get('市净率', 0)) if r.get('市净率') else 0,
                'market_cap': float(r.get('总市值', 0)),
                'market': 'US',
            }
        except Exception as e:
            print(f"获取美股 {code} 数据失败: {e}")
            return None

    def fetch_top_gainers_with_fundamental(self, limit: int = 20) -> List[Dict]:
        """获取美股涨幅排行 + 基本面数据，用于深度分析"""
        gainers = self.fetch_top_gainers(limit=limit)
        if not gainers:
            return []

        from .fundamental_data import FundamentalDataClient
        fetcher = FundamentalDataClient()

        results = []
        for stock in gainers:
            try:
                fundamental = fetcher.get_us_stock_fundamental(stock['code'])
                stock['fundamental'] = fundamental
                stock['fundamental_score'] = fundamental.get('fundamental_score', 50)
            except Exception as e:
                print(f"获取美股 {stock['code']} 基本面失败: {e}")
                stock['fundamental'] = None
                stock['fundamental_score'] = 50
            results.append(stock)

        return results

    def analyze_and_predict(self, limit: int = 20) -> List[Dict]:
        """基于基本面分析的美股深度选股与预测

        不再仅依赖涨幅排行，而是综合基本面(质量/成长/估值) +
        技术面(趋势/动量) + 资金面(成交额)进行评分预测。
        """
        # 1. 获取候选池
        candidates = self.fetch_top_gainers(limit=limit * 3)
        if not candidates:
            return []

        # 2. 注入基本面数据并评分
        from .fundamental_data import FundamentalDataClient
        fetcher = FundamentalDataClient()

        scored = []
        for stock in candidates:
            try:
                fundamental = fetcher.get_us_stock_fundamental(stock['code'])
                stock['fundamental'] = fundamental
                score_result = self._calculate_us_prediction_score(stock, fundamental)
                stock.update(score_result)
                scored.append(stock)
            except Exception as e:
                print(f"美股 {stock['code']} 分析失败: {e}")
                stock['fundamental'] = None
                stock['fundamental_score'] = 50
                stock['prediction_score'] = 50
                stock['predicted_trend'] = 'neutral'
                scored.append(stock)

        # 3. 按综合评分排序
        scored.sort(key=lambda x: x.get('prediction_score', 0), reverse=True)
        return scored[:limit]

    def _calculate_us_prediction_score(self, stock: Dict, fundamental: Optional[Dict]) -> Dict:
        """计算美股预测评分

        评分维度:
        - 基本面质量(25%): ROE/净利率/资产负债率/现金流
        - 成长性(35%): 营收增速/利润增速（美股成长股权重更高）
        - 估值合理性(20%): PE/PB
        - 技术面动量(20%): 涨幅/成交额/振幅
        """
        scores = {
            'quality_score': 50,
            'growth_score': 50,
            'valuation_score': 50,
            'momentum_score': 50,
            'prediction_score': 50,
            'predicted_trend': 'neutral',
            'predicted_change': 0.0,
            'confidence': 0.3,
            'score_breakdown': {},
        }

        if not fundamental or not fundamental.get('financial'):
            momentum = self._calc_momentum_score(stock)
            scores['momentum_score'] = momentum
            scores['prediction_score'] = int(momentum * 0.4 + 50 * 0.6)
            scores['predicted_trend'] = 'bullish' if momentum > 65 else ('bearish' if momentum < 35 else 'neutral')
            scores['confidence'] = 0.3
            scores['score_breakdown'] = {'note': '无基本面数据，仅技术面评分'}
            return scores

        fin = fundamental['financial']

        # 基本面质量分
        quality = 50
        quality_details = []

        roe = fin.get('roe', 0)
        if roe > 25:
            quality += 15
            quality_details.append(f'ROE={roe:.1f}%优秀')
        elif roe > 15:
            quality += 8
            quality_details.append(f'ROE={roe:.1f}%良好')
        elif roe > 0:
            quality += 3

        net_margin = fin.get('net_margin', 0)
        if net_margin > 20:
            quality += 12
            quality_details.append(f'净利率={net_margin:.1f}%优秀')
        elif net_margin > 10:
            quality += 6
        elif net_margin > 0:
            quality += 2

        debt_ratio = fin.get('asset_liability_ratio', 0)
        if 0 < debt_ratio < 40:
            quality += 8
            quality_details.append(f'负债率={debt_ratio:.1f}%低风险')
        elif 40 <= debt_ratio < 60:
            quality += 4
        elif debt_ratio >= 70:
            quality -= 8
            quality_details.append(f'负债率={debt_ratio:.1f}%偏高')

        current_ratio = fin.get('current_ratio', 0)
        if current_ratio > 2:
            quality += 5
            quality_details.append(f'流动比率={current_ratio:.1f}充裕')
        elif current_ratio > 1:
            quality += 2

        ocf = fin.get('operating_cash_flow', 0)
        if ocf > 0:
            quality += 5
            quality_details.append('经营现金流为正')

        quality = max(0, min(100, quality))
        scores['quality_score'] = quality

        # 成长性评分（美股成长权重更高 35%）
        growth = 50
        growth_details = []

        rev_yoy = fin.get('revenue_yoy', 0)
        if rev_yoy > 30:
            growth += 20
            growth_details.append(f'营收同比+{rev_yoy:.1f}%高增')
        elif rev_yoy > 15:
            growth += 12
            growth_details.append(f'营收同比+{rev_yoy:.1f}%稳健')
        elif rev_yoy > 0:
            growth += 5
        elif rev_yoy < -10:
            growth -= 10
            growth_details.append(f'营收同比{rev_yoy:.1f}%下滑')

        profit_yoy = fin.get('net_profit_yoy', 0)
        if profit_yoy > 30:
            growth += 20
            growth_details.append(f'利润同比+{profit_yoy:.1f}%高增')
        elif profit_yoy > 15:
            growth += 12
            growth_details.append(f'利润同比+{profit_yoy:.1f}%稳健')
        elif profit_yoy > 0:
            growth += 5
        elif profit_yoy < -10:
            growth -= 10
            growth_details.append(f'利润同比{profit_yoy:.1f}%下滑')

        gross_margin = fin.get('gross_margin', 0)
        if gross_margin > 60:
            growth += 10
            growth_details.append(f'毛利率={gross_margin:.1f}%极高(科技/医药)')
        elif gross_margin > 40:
            growth += 6
        elif gross_margin > 20:
            growth += 3

        growth = max(0, min(100, growth))
        scores['growth_score'] = growth

        # 估值评分
        valuation = 50
        valuation_details = []

        pe = fin.get('pe_ratio', 0)
        if 0 < pe < 15:
            valuation += 20
            valuation_details.append(f'PE={pe:.1f}低估值')
        elif 15 <= pe < 25:
            valuation += 10
            valuation_details.append(f'PE={pe:.1f}合理')
        elif 25 <= pe < 40:
            valuation += 0
        elif pe >= 40:
            valuation -= 8
            valuation_details.append(f'PE={pe:.1f}偏高')
        elif pe < 0:
            valuation -= 15
            valuation_details.append('亏损股')

        pb = fin.get('pb_ratio', 0)
        if 0 < pb < 2:
            valuation += 8
            valuation_details.append(f'PB={pb:.1f}合理')
        elif 2 <= pb < 5:
            valuation += 3
        elif pb >= 8:
            valuation -= 5

        # 美股PS指标（市销率）对未盈利公司重要
        market_cap = fin.get('total_market_cap', 0)
        revenue = fin.get('revenue', 0)
        if revenue > 0 and market_cap > 0:
            ps = market_cap / revenue
            if ps < 3:
                valuation += 8
                valuation_details.append(f'PS={ps:.1f}低估值')
            elif ps < 8:
                valuation += 3
            elif ps > 15:
                valuation -= 5

        valuation = max(0, min(100, valuation))
        scores['valuation_score'] = valuation

        # 技术面动量评分
        momentum = self._calc_momentum_score(stock)
        scores['momentum_score'] = momentum

        # 综合评分（美股成长权重更高）
        total = int(
            quality * 0.25 +
            growth * 0.35 +
            valuation * 0.20 +
            momentum * 0.20
        )
        scores['prediction_score'] = total

        # 预测方向
        if total >= 70:
            scores['predicted_trend'] = 'bullish'
            scores['predicted_change'] = round((total - 50) * 0.1, 2)
        elif total >= 55:
            scores['predicted_trend'] = 'slightly_bullish'
            scores['predicted_change'] = round((total - 50) * 0.06, 2)
        elif total >= 45:
            scores['predicted_trend'] = 'neutral'
            scores['predicted_change'] = 0.0
        elif total >= 30:
            scores['predicted_trend'] = 'slightly_bearish'
            scores['predicted_change'] = round((total - 50) * 0.06, 2)
        else:
            scores['predicted_trend'] = 'bearish'
            scores['predicted_change'] = round((total - 50) * 0.1, 2)

        # 置信度
        data_quality = fundamental.get('data_quality', 0)
        data_completeness = fundamental.get('data_completeness', 0)
        scores['confidence'] = round(0.3 + data_completeness * 0.4 + data_quality * 0.3, 2)
        scores['confidence'] = min(scores['confidence'], 0.9)

        scores['score_breakdown'] = {
            'quality': {'score': quality, 'details': quality_details},
            'growth': {'score': growth, 'details': growth_details},
            'valuation': {'score': valuation, 'details': valuation_details},
            'momentum': {'score': momentum},
        }

        return scores

    def _calc_momentum_score(self, stock: Dict) -> int:
        """技术面动量评分"""
        momentum = 50

        change_pct = stock.get('change_pct', 0)
        if change_pct > 5:
            momentum += 15
        elif change_pct > 2:
            momentum += 8
        elif change_pct > 0:
            momentum += 3
        elif change_pct < -5:
            momentum -= 15
        elif change_pct < -2:
            momentum -= 8

        amount = stock.get('amount', 0)
        if amount > 1e9:
            momentum += 10
        elif amount > 5e8:
            momentum += 5

        amplitude = stock.get('amplitude', 0)
        if 2 < amplitude < 8:
            momentum += 5
        elif amplitude > 10:
            momentum -= 5

        return max(0, min(100, momentum))
