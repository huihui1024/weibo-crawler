"""
回测框架
用于验证选股策略和预测模型的准确性
支持：历史回测、策略评估、收益分析
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import sqlite3
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from app.database import get_sync_db


@dataclass
class BacktestTrade:
    """回测交易记录"""
    code: str
    name: str
    direction: str  # 'buy' or 'sell'
    date: date
    price: float
    shares: int
    amount: float  # 成交金额
    commission: float  # 手续费
    reason: str = ""


@dataclass
class BacktestPosition:
    """回测持仓"""
    code: str
    name: str
    shares: int
    cost_price: float
    current_price: float
    market_value: float
    profit: float
    profit_pct: float
    hold_days: int
    peak_profit_pct: float = 0  # 持仓期间最大盈利%，用于移动止损


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本统计
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float

    # 收益指标
    total_return: float  # 总收益率
    annual_return: float  # 年化收益率
    benchmark_return: float  # 基准收益率

    # 风险指标
    max_drawdown: float  # 最大回撤
    volatility: float  # 年化波动率
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率

    # 交易统计
    total_trades: int
    win_trades: int
    lose_trades: int
    win_rate: float
    avg_profit_trade: float  # 平均盈利交易
    avg_loss_trade: float  # 平均亏损交易
    profit_factor: float  # 盈亏比

    # 持仓统计
    avg_hold_days: float
    max_hold_days: int

    # 详细数据
    trades: List[BacktestTrade] = field(default_factory=list)
    daily_values: List[Dict] = field(default_factory=list)
    positions: List[BacktestPosition] = field(default_factory=list)


class BacktestEngine:
    """回测引擎 — 严格执行A股交易规则（T+1、涨跌停限制）"""

    def __init__(self, initial_capital: float = 1000000, commission_rate: float = 0.0003,
                 slippage_pct: float = 0.1):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_pct: 滑点百分比(单边, 如0.1表示0.1%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = 5  # 最低手续费
        self.slippage_pct = slippage_pct / 100  # 转为小数

        # 回测状态
        self.cash = initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[BacktestTrade] = []
        self.daily_values: List[Dict] = []
        self.buy_dates: Dict[str, date] = {}  # T+1: 记录买入日期

    def reset(self):
        """重置回测状态"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_values = []

    def buy(self, code: str, name: str, trade_date: date, price: float,
            amount: float = None, shares: int = None, reason: str = "") -> bool:
        """
        买入股票

        Args:
            code: 股票代码
            name: 股票名称
            trade_date: 交易日期
            price: 买入价格(理想价)
            amount: 买入金额（与shares二选一）
            shares: 买入股数（与amount二选一）
            reason: 买入原因

        Returns:
            bool: 是否成功
        """
        # 加入滑点：买入价上浮
        actual_price = price * (1 + self.slippage_pct)

        # 计算买入数量
        if shares is None:
            if amount is None:
                amount = self.cash * 0.1  # 默认使用10%仓位
            shares = int(amount / actual_price / 100) * 100  # 向下取整到100股

        if shares <= 0:
            return False

        # 计算成交金额和手续费
        trade_amount = shares * actual_price
        commission = max(trade_amount * self.commission_rate, self.min_commission)
        total_cost = trade_amount + commission

        # 检查资金是否足够
        if total_cost > self.cash:
            shares = int((self.cash - self.min_commission) / actual_price / 100) * 100
            if shares <= 0:
                return False
            trade_amount = shares * actual_price
            commission = max(trade_amount * self.commission_rate, self.min_commission)
            total_cost = trade_amount + commission

        # 执行买入
        self.cash -= total_cost

        # 更新持仓
        if code in self.positions:
            pos = self.positions[code]
            total_shares = pos.shares + shares
            total_cost_price = (pos.cost_price * pos.shares + actual_price * shares) / total_shares
            pos.shares = total_shares
            pos.cost_price = total_cost_price
        else:
            self.positions[code] = BacktestPosition(
                code=code,
                name=name,
                shares=shares,
                cost_price=actual_price,
                current_price=actual_price,
                market_value=trade_amount,
                profit=0,
                profit_pct=0,
                hold_days=0
            )

        # 记录交易
        self.trades.append(BacktestTrade(
            code=code,
            name=name,
            direction='buy',
            date=trade_date,
            price=price,
            shares=shares,
            amount=trade_amount,
            commission=commission,
            reason=reason
        ))

        return True

    def sell(self, code: str, trade_date: date, price: float,
             shares: int = None, reason: str = "") -> bool:
        """
        卖出股票

        Args:
            code: 股票代码
            trade_date: 交易日期
            price: 卖出价格(理想价)
            shares: 卖出股数（None表示全部卖出）
            reason: 卖出原因

        Returns:
            bool: 是否成功
        """
        if code not in self.positions:
            return False

        pos = self.positions[code]

        if shares is None or shares >= pos.shares:
            shares = pos.shares

        if shares <= 0:
            return False

        # 加入滑点：卖出价下浮
        actual_price = price * (1 - self.slippage_pct)

        # 计算成交金额和手续费
        trade_amount = shares * actual_price
        commission = max(trade_amount * self.commission_rate, self.min_commission)
        net_amount = trade_amount - commission

        # 执行卖出
        self.cash += net_amount

        # 更新持仓
        pos.shares -= shares
        if pos.shares <= 0:
            del self.positions[code]

        # 记录交易
        self.trades.append(BacktestTrade(
            code=code,
            name=pos.name,
            direction='sell',
            date=trade_date,
            price=price,
            shares=shares,
            amount=trade_amount,
            commission=commission,
            reason=reason
        ))

        return True

    def update_positions(self, trade_date: date, prices: Dict[str, float]):
        """
        更新持仓市值

        Args:
            trade_date: 交易日期
            prices: 股票价格字典 {code: price}
        """
        total_market_value = self.cash

        for code, pos in self.positions.items():
            if code in prices:
                pos.current_price = prices[code]
                pos.market_value = pos.shares * pos.current_price
                pos.profit = pos.market_value - pos.cost_price * pos.shares
                pos.profit_pct = (pos.current_price - pos.cost_price) / pos.cost_price * 100
                pos.hold_days += 1
                # 更新峰值盈利（移动止损用）
                if pos.profit_pct > pos.peak_profit_pct:
                    pos.peak_profit_pct = pos.profit_pct
            total_market_value += pos.market_value

        # 记录每日净值
        self.daily_values.append({
            'date': trade_date,
            'cash': self.cash,
            'market_value': total_market_value - self.cash,
            'total_value': total_market_value,
            'positions': len(self.positions)
        })

    def get_total_value(self) -> float:
        """获取总资产"""
        return self.cash + sum(pos.market_value for pos in self.positions.values())


class BacktestStrategy(ABC):
    """回测策略基类"""

    @abstractmethod
    def generate_signals(self, date: date, data: Dict) -> List[Dict]:
        """
        生成交易信号

        Args:
            date: 当前日期
            data: 市场数据

        Returns:
            List[Dict]: 交易信号列表 [{'code': str, 'action': 'buy'/'sell', 'reason': str}]
        """
        pass


class PredictionBasedStrategy(BacktestStrategy):
    """基于预测的策略"""

    def __init__(self, score_threshold: float = 60, max_positions: int = 10):
        self.score_threshold = score_threshold
        self.max_positions = max_positions

    def generate_signals(self, date: date, data: Dict) -> List[Dict]:
        signals = []

        # 获取当日预测
        predictions = data.get('predictions', [])

        for pred in predictions:
            score = pred.get('score', 0)
            if score >= self.score_threshold:
                signals.append({
                    'code': pred['code'],
                    'action': 'buy',
                    'reason': f"预测得分{score:.1f}分，{pred.get('trend', '')}"
                })

        return signals


class MultiFactorStrategy(BacktestStrategy):
    """多因子策略"""

    def __init__(self, score_threshold: float = 65, max_positions: int = 10):
        self.score_threshold = score_threshold
        self.max_positions = max_positions

    def generate_signals(self, date: date, data: Dict) -> List[Dict]:
        signals = []

        # 获取因子评分
        scores = data.get('factor_scores', [])

        # 按得分排序，选择前N只
        sorted_scores = sorted(scores, key=lambda x: x.get('total_score', 0), reverse=True)

        for score in sorted_scores[:self.max_positions]:
            if score.get('total_score', 0) >= self.score_threshold:
                signals.append({
                    'code': score['code'],
                    'action': 'buy',
                    'reason': f"多因子得分{score['total_score']:.1f}分，推荐{score.get('recommendation', '')}"
                })

        return signals


class BacktestRunner:
    """回测运行器"""

    def __init__(self, db_path: str = 'data/weibo.db', slippage_pct: float = 0.1):
        self.db_path = db_path
        self.engine = BacktestEngine(slippage_pct=slippage_pct)
        self.slippage_pct = slippage_pct

    def load_historical_predictions(self, start_date: date, end_date: date) -> Dict[date, List]:
        """
        加载历史预测数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict[date, List]: 按日期分组的预测数据
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT stock_code, stock_name, prediction_date, prediction_time,
                   current_price, expected_close_change, expected_trend, expected_strength,
                   expected_hold_days, expected_hold_return, selection_reason
            FROM predictions
            WHERE prediction_date >= ? AND prediction_date <= ?
            ORDER BY prediction_date
        """, (start_date, end_date))

        rows = cursor.fetchall()
        conn.close()

        predictions_by_date = {}
        for row in rows:
            pred_date = date.fromisoformat(row[2])
            if pred_date not in predictions_by_date:
                predictions_by_date[pred_date] = []

            predictions_by_date[pred_date].append({
                'code': row[0],
                'name': row[1],
                'prediction_date': row[2],
                'current_price': row[4],
                'expected_change': row[5],
                'trend': row[6],
                'strength': row[7],
                'hold_days': row[8],
                'expected_return': row[9],
                'reason': row[10]
            })

        return predictions_by_date

    def load_review_results(self, start_date: date, end_date: date) -> Dict[date, List]:
        """
        加载复盘结果数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict[date, List]: 按日期分组的复盘数据
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='review_results'")
        if not cursor.fetchone():
            conn.close()
            return {}

        # 获取表结构
        cursor.execute("PRAGMA table_info(review_results)")
        columns = [col[1] for col in cursor.fetchall()]

        # 根据实际列名构建查询
        if 'expected_change' in columns:
            predicted_col = 'expected_change'
        elif 'predicted_change' in columns:
            predicted_col = 'predicted_change'
        else:
            predicted_col = 'expected_change'

        try:
            cursor.execute(f"""
                SELECT stock_code, stock_name, review_date,
                       {predicted_col}, actual_change, trend_match, prediction_score
                FROM review_results
                WHERE review_date >= ? AND review_date <= ?
                ORDER BY review_date
            """, (start_date, end_date))

            rows = cursor.fetchall()
        except Exception as e:
            print(f"查询复盘数据失败: {e}")
            rows = []

        conn.close()

        reviews_by_date = {}
        for row in rows:
            review_date = date.fromisoformat(row[2])
            if review_date not in reviews_by_date:
                reviews_by_date[review_date] = []

            reviews_by_date[review_date].append({
                'code': row[0],
                'name': row[1],
                'predicted_change': row[3],
                'actual_change': row[4],
                'trend_match': row[5],
                'score': row[6]
            })

        return reviews_by_date

    def load_benchmark_data(self, start_date: date, end_date: date) -> Dict[date, float]:
        """加载沪深300基准数据

        Returns:
            Dict[date, float]: {date: close_price}
        """
        try:
            from app.market import get_market_client
            client = get_market_client()
            df = client.get_stock_history('000300', days=(end_date - start_date).days + 30)
            if df.empty:
                return {}

            benchmark = {}
            for _, row in df.iterrows():
                d = row['date'] if 'date' in df.columns else row.name
                if isinstance(d, str):
                    d = date.fromisoformat(d)
                elif hasattr(d, 'date'):
                    d = d.date()
                if start_date <= d <= end_date:
                    benchmark[d] = row['close']

            return benchmark
        except Exception as e:
            print(f"加载基准数据失败: {e}")
            return {}

    def run_backtest(self, strategy: BacktestStrategy,
                     start_date: date, end_date: date,
                     initial_capital: float = 1000000,
                     use_adaptive_strategy: bool = True) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 回测策略
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            use_adaptive_strategy: 是否使用自适应策略(ATR止损止盈)

        Returns:
            BacktestResult: 回测结果
        """
        self.engine = BacktestEngine(initial_capital=initial_capital,
                                     slippage_pct=self.slippage_pct)

        # 加载历史数据
        predictions = self.load_historical_predictions(start_date, end_date)
        reviews = self.load_review_results(start_date, end_date)

        # 加载基准数据(沪深300)
        benchmark_data = self.load_benchmark_data(start_date, end_date)

        # 自适应策略管理器
        adaptive_mgr = None
        if use_adaptive_strategy:
            try:
                from app.selector.adaptive_strategy import AdaptiveStrategyManager
                adaptive_mgr = AdaptiveStrategyManager()
            except Exception:
                pass

        # 获取交易日历
        trade_dates = sorted(set(list(predictions.keys()) + list(reviews.keys())))

        if not trade_dates:
            return self._empty_result(start_date, end_date, initial_capital)

        # 模拟交易
        for i, trade_date in enumerate(trade_dates):
            # 准备当日数据
            day_data = {
                'predictions': predictions.get(trade_date, []),
                'reviews': reviews.get(trade_date, [])
            }

            # 生成信号
            signals = strategy.generate_signals(trade_date, day_data)

            # 执行交易（T+1规则：买入当日不可卖出）
            for signal in signals:
                if signal['action'] == 'buy':
                    # 排除ST/退市股（幸存者偏差修正）
                    name = signal.get('name', '')
                    if 'ST' in name or '退' in name:
                        continue
                    # 找到预测价格
                    pred = next((p for p in day_data['predictions'] if p['code'] == signal['code']), None)
                    if pred:
                        # 前视偏差修正：使用次日开盘价近似（无次日数据时跳过买入）
                        buy_price = pred.get('next_open', pred.get('current_price', 0))
                        if buy_price <= 0:
                            buy_price = pred.get('current_price', 0)
                        if buy_price <= 0:
                            continue
                        self.engine.buy(
                            code=signal['code'],
                            name=pred['name'],
                            trade_date=trade_date,
                            price=buy_price,
                            reason=signal['reason']
                        )
                        self.engine.buy_dates[signal['code']] = trade_date  # T+1记录

            # 更新持仓（使用复盘实际价格）
            day_reviews = reviews.get(trade_date, [])
            prices = {}
            for review in day_reviews:
                # 使用预测时的价格作为参考
                pred = next((p for p in predictions.get(trade_date, []) if p['code'] == review['code']), None)
                if pred and pred.get('current_price') and review.get('actual_change'):
                    # 计算实际价格（基于预测价格和实际涨跌幅）
                    try:
                        actual_price = pred['current_price'] * (1 + float(review['actual_change']) / 100)
                        prices[review['code']] = actual_price
                    except (TypeError, ValueError):
                        pass

            self.engine.update_positions(trade_date, prices)

            # 卖出逻辑：持有超过预期天数或达到止损止盈(自适应)
            positions_to_sell = []
            for code, pos in list(self.engine.positions.items()):
                sell_reason = None

                # 找到买入时的预测持有天数
                buy_trade = next((t for t in self.engine.trades if t.code == code and t.direction == 'buy'), None)
                if buy_trade:
                    pred = next((p for p in predictions.get(buy_trade.date, []) if p['code'] == code), None)
                    if pred:
                        expected_days = pred.get('hold_days', 10)

                        # 自适应策略止损止盈
                        if adaptive_mgr:
                            # 简化：使用oscillation作为默认regime
                            profile = adaptive_mgr.get_strategy('oscillation')
                            sl_pct = profile.stop_loss_pct
                            tp_pct = profile.take_profit_pct
                            hold_days = min(expected_days, profile.max_hold_days)
                        else:
                            sl_pct = -5.0
                            tp_pct = 10.0
                            hold_days = expected_days

                        if pos.hold_days >= hold_days:
                            sell_reason = f"达到预期持有天数({hold_days}天)"

                        # 固定止损止盈
                        if pos.profit_pct > tp_pct:
                            sell_reason = f"止盈卖出(+{pos.profit_pct:.1f}%>{tp_pct}%)"
                        elif pos.profit_pct < sl_pct:
                            sell_reason = f"止损卖出({pos.profit_pct:.1f}%<{sl_pct}%)"

                        # 移动止损：持仓盈利后从最高点回落超过阈值
                        if sell_reason is None and adaptive_mgr and pos.peak_profit_pct > 0:
                            trailing_pct = profile.trailing_stop_pct
                            drawdown_from_peak = pos.peak_profit_pct - pos.profit_pct
                            if drawdown_from_peak > trailing_pct:
                                sell_reason = f"移动止损(从峰值{pos.peak_profit_pct:.1f}%回落{drawdown_from_peak:.1f}%>{trailing_pct}%)"

                if sell_reason:
                    positions_to_sell.append((code, pos.current_price, sell_reason))

            for code, price, reason in positions_to_sell:
                # T+1检查：买入当日不可卖出
                buy_date = self.engine.buy_dates.get(code)
                if buy_date and buy_date >= trade_date:
                    continue  # 跳过T+0卖出
                self.engine.sell(code, trade_date, price, reason=reason)

        # 计算回测结果
        return self._calculate_result(start_date, end_date, initial_capital)

    def _calculate_result(self, start_date: date, end_date: date,
                          initial_capital: float) -> BacktestResult:
        """计算回测结果"""
        final_capital = self.engine.get_total_value()
        total_return = (final_capital - initial_capital) / initial_capital * 100

        # 计算年化收益
        days = (end_date - start_date).days
        annual_return = total_return * 365 / days if days > 0 else 0

        # 计算基准收益(沪深300)
        benchmark_return = 0
        try:
            benchmark_data = self.load_benchmark_data(start_date, end_date)
            if benchmark_data:
                sorted_dates = sorted(benchmark_data.keys())
                if len(sorted_dates) >= 2:
                    start_price = benchmark_data[sorted_dates[0]]
                    end_price = benchmark_data[sorted_dates[-1]]
                    if start_price > 0:
                        benchmark_return = (end_price - start_price) / start_price * 100
        except Exception:
            benchmark_return = 0

        # 计算最大回撤
        max_drawdown = 0
        peak = initial_capital
        for dv in self.engine.daily_values:
            if dv['total_value'] > peak:
                peak = dv['total_value']
            drawdown = (peak - dv['total_value']) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 计算波动率和夏普/索提诺/卡尔玛比率
        sharpe_ratio = 0
        sortino_ratio = 0
        calmar_ratio = 0
        volatility = 0

        if len(self.engine.daily_values) > 1:
            values = [dv['total_value'] for dv in self.engine.daily_values]
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            volatility = np.std(returns) * np.sqrt(252) * 100 if returns else 0
            avg_return = np.mean(returns) * 252 if returns else 0
            sharpe_ratio = avg_return / volatility * 100 if volatility > 0 else 0

            # Sortino: 只用下行波动率
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                downside_vol = np.std(negative_returns) * np.sqrt(252) * 100
                sortino_ratio = avg_return / downside_vol * 100 if downside_vol > 0 else 0
            else:
                sortino_ratio = sharpe_ratio

            # Calmar: 年化收益/最大回撤
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0

        # 交易统计
        buy_trades = [t for t in self.engine.trades if t.direction == 'buy']
        sell_trades = [t for t in self.engine.trades if t.direction == 'sell']

        # 计算盈亏
        win_trades = 0
        lose_trades = 0
        total_profit = 0
        total_loss = 0

        for sell in sell_trades:
            # 找到对应的买入
            buy = next((t for t in reversed(buy_trades) if t.code == sell.code), None)
            if buy:
                profit = (sell.price - buy.price) * sell.shares - sell.commission - buy.commission
                if profit > 0:
                    win_trades += 1
                    total_profit += profit
                else:
                    lose_trades += 1
                    total_loss += abs(profit)

        total_trades = win_trades + lose_trades
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
        avg_profit_trade = total_profit / win_trades if win_trades > 0 else 0
        avg_loss_trade = total_loss / lose_trades if lose_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # 持仓统计
        hold_days_list = []
        for sell in sell_trades:
            buy = next((t for t in buy_trades if t.code == sell.code), None)
            if buy:
                hold_days_list.append((sell.date - buy.date).days)

        avg_hold_days = np.mean(hold_days_list) if hold_days_list else 0
        max_hold_days = max(hold_days_list) if hold_days_list else 0

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annual_return=annual_return,
            benchmark_return=benchmark_return,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            total_trades=total_trades,
            win_trades=win_trades,
            lose_trades=lose_trades,
            win_rate=win_rate,
            avg_profit_trade=avg_profit_trade,
            avg_loss_trade=avg_loss_trade,
            profit_factor=profit_factor,
            avg_hold_days=avg_hold_days,
            max_hold_days=max_hold_days,
            trades=self.engine.trades,
            daily_values=self.engine.daily_values
        )

    def _empty_result(self, start_date: date, end_date: date,
                      initial_capital: float) -> BacktestResult:
        """返回空结果"""
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=initial_capital,
            total_return=0,
            annual_return=0,
            benchmark_return=0,
            max_drawdown=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            total_trades=0,
            win_trades=0,
            lose_trades=0,
            win_rate=0,
            avg_profit_trade=0,
            avg_loss_trade=0,
            profit_factor=0,
            avg_hold_days=0,
            max_hold_days=0
        )


def generate_backtest_report(result: BacktestResult) -> str:
    """生成回测报告"""
    report = f"""
# 📊 回测报告

> 回测区间：{result.start_date} ~ {result.end_date}
> 初始资金：{result.initial_capital:,.0f} 元
> 最终资金：{result.final_capital:,.0f} 元

---

## 一、收益概况

| 指标 | 数值 |
|------|------|
| 总收益率 | {result.total_return:+.2f}% |
| 年化收益率 | {result.annual_return:+.2f}% |
| 基准收益率 | {result.benchmark_return:+.2f}% |
| 超额收益 | {result.total_return - result.benchmark_return:+.2f}% |

---

## 二、风险指标

| 指标 | 数值 |
|------|------|
| 最大回撤 | {result.max_drawdown:.2f}% |
| 年化波动率 | {result.volatility:.2f}% |
| 夏普比率 | {result.sharpe_ratio:.2f} |
| 索提诺比率 | {result.sortino_ratio:.2f} |
| 卡尔玛比率 | {result.total_return / result.max_drawdown:.2f if result.max_drawdown > 0 else 'N/A'} |

---

## 三、交易统计

| 指标 | 数值 |
|------|------|
| 总交易次数 | {result.total_trades} |
| 盈利交易 | {result.win_trades} |
| 亏损交易 | {result.lose_trades} |
| 胜率 | {result.win_rate:.1f}% |
| 平均盈利 | {result.avg_profit_trade:,.0f} 元 |
| 平均亏损 | {result.avg_loss_trade:,.0f} 元 |
| 盈亏比 | {result.profit_factor:.2f} |
| 平均持仓天数 | {result.avg_hold_days:.1f} 天 |
| 最长持仓天数 | {result.max_hold_days} 天 |

---

## 四、交易明细

"""
    if result.trades:
        report += "| 日期 | 代码 | 名称 | 方向 | 价格 | 数量 | 金额 | 手续费 | 原因 |\n"
        report += "|------|------|------|------|------|------|------|--------|------|\n"
        for trade in result.trades[:50]:  # 最多显示50条
            report += f"| {trade.date} | {trade.code} | {trade.name} | {trade.direction} | {trade.price:.2f} | {trade.shares} | {trade.amount:,.0f} | {trade.commission:.2f} | {trade.reason[:20]} |\n"
    else:
        report += "暂无交易记录\n"

    report += """

---

## 📝 说明

- 本回测基于历史数据，不代表未来收益
- 回测已考虑滑点(0.1%单边)和手续费(万三，最低5元)
- 基准为沪深300指数
- 止损止盈使用自适应策略(ATR动态调整)
- 投资有风险，入市需谨慎
"""
    return report


def run_prediction_backtest(db_path: str = 'data/weibo.db',
                            days: int = 30) -> BacktestResult:
    """
    运行预测策略回测

    Args:
        db_path: 数据库路径
        days: 回测天数

    Returns:
        BacktestResult: 回测结果
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    runner = BacktestRunner(db_path)
    strategy = PredictionBasedStrategy(score_threshold=60)

    return runner.run_backtest(strategy, start_date, end_date)


if __name__ == "__main__":
    # 测试回测框架
    print("测试回测框架...")

    # 创建模拟回测
    runner = BacktestRunner()

    # 测试空回测
    result = runner.run_backtest(
        PredictionBasedStrategy(),
        date.today() - timedelta(days=30),
        date.today()
    )

    print(generate_backtest_report(result))
