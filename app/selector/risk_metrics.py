"""
风险度量模块
实现VaR（风险价值）、CVaR、压力测试、相关性矩阵等专业风险度量功能
支持：历史模拟法、参数法、蒙特卡洛模拟
"""
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm, t as t_dist


class VaRMethod(Enum):
    """VaR计算方法"""
    HISTORICAL = "historical"  # 历史模拟法
    PARAMETRIC = "parametric"  # 参数法（正态分布假设）
    MONTE_CARLO = "monte_carlo"  # 蒙特卡洛模拟
    CORNISH_FISHER = "cornish_fisher"  # Cornish-Fisher展开


class RiskMetricType(Enum):
    """风险指标类型"""
    VAR = "VaR"  # 风险价值
    CVAR = "CVaR"  # 条件风险价值（Expected Shortfall）
    MAX_DRAWDOWN = "MaxDrawdown"  # 最大回撤
    SHARPE = "Sharpe"  # 夏普比率
    SORTINO = "Sortino"  # 索提诺比率
    CALMAR = "Calmar"  # 卡玛比率
    VOLATILITY = "Volatility"  # 波动率
    BETA = "Beta"  # 贝塔系数


@dataclass
class VaRResult:
    """VaR计算结果"""
    var_value: float  # VaR值（百分比）
    var_absolute: float  # VaR值（绝对金额）
    confidence_level: float  # 置信水平
    time_horizon: int  # 时间跨度（天）
    method: str  # 计算方法

    # 扩展指标
    cvar: float = 0.0  # 条件VaR（Expected Shortfall）
    var_range: Tuple[float, float] = (0, 0)  # VaR置信区间

    # 分布参数
    mean_return: float = 0.0
    std_return: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    # 风险等级
    risk_level: str = "中等"
    risk_score: float = 50.0


@dataclass
class PortfolioRiskReport:
    """组合风险报告"""
    portfolio_value: float
    calculation_time: datetime

    # 单资产风险
    individual_risks: Dict[str, VaRResult] = field(default_factory=dict)

    # 组合风险
    portfolio_var: Optional[VaRResult] = None
    portfolio_cvar: float = 0.0

    # 风险贡献
    risk_contribution: Dict[str, float] = field(default_factory=dict)
    marginal_var: Dict[str, float] = field(default_factory=dict)

    # 相关性
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    covariance_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 压力测试
    stress_test_results: Dict[str, float] = field(default_factory=dict)

    # 风险指标
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # 风险评级
    overall_risk_score: float = 50.0
    risk_rating: str = "中等风险"


class VaRCalculator:
    """VaR计算器"""

    def __init__(self, confidence_level: float = 0.95,
                 time_horizon: int = 1,
                 annual_trading_days: int = 252):
        """
        Args:
            confidence_level: 置信水平（默认95%）
            time_horizon: 时间跨度（天）
            annual_trading_days: 年交易日数
        """
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.annual_trading_days = annual_trading_days

    def calculate_var(self, returns: pd.Series,
                      method: VaRMethod = VaRMethod.HISTORICAL,
                      position_value: float = 100000) -> VaRResult:
        """
        计算VaR

        Args:
            returns: 收益率序列
            method: 计算方法
            position_value: 持仓金额

        Returns:
            VaRResult
        """
        if len(returns) < 20:
            return self._default_result(position_value)

        returns = returns.dropna()

        # 计算分布参数
        mean_return = returns.mean()
        std_return = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # 根据方法计算VaR
        if method == VaRMethod.HISTORICAL:
            var_pct = self._historical_var(returns)
            cvar_pct = self._historical_cvar(returns)
        elif method == VaRMethod.PARAMETRIC:
            var_pct = self._parametric_var(mean_return, std_return)
            cvar_pct = self._parametric_cvar(mean_return, std_return)
        elif method == VaRMethod.MONTE_CARLO:
            var_pct, cvar_pct = self._monte_carlo_var(returns)
        elif method == VaRMethod.CORNISH_FISHER:
            var_pct = self._cornish_fisher_var(mean_return, std_return, skewness, kurtosis)
            cvar_pct = var_pct * 1.1  # 近似
        else:
            var_pct = self._historical_var(returns)
            cvar_pct = self._historical_cvar(returns)

        # 时间跨度调整
        var_pct = var_pct * np.sqrt(self.time_horizon)
        cvar_pct = cvar_pct * np.sqrt(self.time_horizon)

        # 计算绝对VaR
        var_absolute = position_value * var_pct / 100

        # VaR置信区间
        n = len(returns)
        se = std_return / np.sqrt(n)
        var_range = (
            var_pct - 1.96 * se * 100,
            var_pct + 1.96 * se * 100
        )

        # 风险等级评估
        risk_level, risk_score = self._assess_risk_level(var_pct)

        return VaRResult(
            var_value=round(var_pct, 4),
            var_absolute=round(var_absolute, 2),
            confidence_level=self.confidence_level,
            time_horizon=self.time_horizon,
            method=method.value,
            cvar=round(cvar_pct, 4),
            var_range=(round(var_range[0], 4), round(var_range[1], 4)),
            mean_return=round(mean_return * 100, 4),
            std_return=round(std_return * 100, 4),
            skewness=round(skewness, 4),
            kurtosis=round(kurtosis, 4),
            risk_level=risk_level,
            risk_score=risk_score
        )

    def _historical_var(self, returns: pd.Series) -> float:
        """历史模拟法计算VaR"""
        alpha = 1 - self.confidence_level
        var = np.percentile(returns, alpha * 100)
        return -var * 100  # 转换为正数百分比

    def _historical_cvar(self, returns: pd.Series) -> float:
        """历史模拟法计算CVaR（Expected Shortfall）"""
        alpha = 1 - self.confidence_level
        var_threshold = np.percentile(returns, alpha * 100)
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) > 0:
            cvar = -tail_returns.mean() * 100
        else:
            cvar = -var_threshold * 100
        return cvar

    def _parametric_var(self, mean: float, std: float) -> float:
        """参数法计算VaR（正态分布假设）"""
        z_score = norm.ppf(1 - self.confidence_level)
        var = -(mean + z_score * std) * 100
        return var

    def _parametric_cvar(self, mean: float, std: float) -> float:
        """参数法计算CVaR"""
        z_score = norm.ppf(1 - self.confidence_level)
        phi_z = norm.pdf(z_score)
        cvar = -(mean - std * phi_z / (1 - self.confidence_level)) * 100
        return cvar

    def _monte_carlo_var(self, returns: pd.Series, n_simulations: int = 10000) -> Tuple[float, float]:
        """蒙特卡洛模拟计算VaR"""
        mean = returns.mean()
        std = returns.std()

        # 生成模拟收益率
        simulated_returns = np.random.normal(mean, std, n_simulations)

        # 计算VaR和CVaR
        alpha = 1 - self.confidence_level
        var = -np.percentile(simulated_returns, alpha * 100) * 100

        var_threshold = np.percentile(simulated_returns, alpha * 100)
        tail_returns = simulated_returns[simulated_returns <= var_threshold]
        cvar = -tail_returns.mean() * 100 if len(tail_returns) > 0 else var

        return var, cvar

    def _cornish_fisher_var(self, mean: float, std: float,
                            skewness: float, kurtosis: float) -> float:
        """Cornish-Fisher展开计算VaR（考虑偏度和峰度）"""
        z = norm.ppf(1 - self.confidence_level)

        # Cornish-Fisher展开
        z_cf = (z +
                (z**2 - 1) * skewness / 6 +
                (z**3 - 3*z) * kurtosis / 24 -
                (2*z**3 - 5*z) * skewness**2 / 36)

        var = -(mean + z_cf * std) * 100
        return var

    def _assess_risk_level(self, var_pct: float) -> Tuple[str, float]:
        """评估风险等级"""
        if var_pct <= 2:
            return "低风险", 20
        elif var_pct <= 4:
            return "中低风险", 40
        elif var_pct <= 6:
            return "中等风险", 60
        elif var_pct <= 10:
            return "中高风险", 75
        else:
            return "高风险", 90

    def _default_result(self, position_value: float) -> VaRResult:
        """默认结果（数据不足时）"""
        return VaRResult(
            var_value=5.0,
            var_absolute=position_value * 0.05,
            confidence_level=self.confidence_level,
            time_horizon=self.time_horizon,
            method="default",
            cvar=7.0,
            risk_level="未知",
            risk_score=50.0
        )


class PortfolioRiskAnalyzer:
    """组合风险分析器"""

    def __init__(self, confidence_level: float = 0.95):
        self.var_calculator = VaRCalculator(confidence_level)
        self.confidence_level = confidence_level

    def analyze_portfolio(self, positions: Dict[str, float],
                         price_data: Dict[str, pd.DataFrame],
                         benchmark_data: pd.DataFrame = None) -> PortfolioRiskReport:
        """
        分析组合风险

        Args:
            positions: {code: position_value} 持仓
            price_data: {code: DataFrame} 价格数据
            benchmark_data: 基准数据（可选）

        Returns:
            PortfolioRiskReport
        """
        portfolio_value = sum(positions.values())

        report = PortfolioRiskReport(
            portfolio_value=portfolio_value,
            calculation_time=datetime.now()
        )

        # 计算各资产收益率
        returns_data = {}
        for code, df in price_data.items():
            if len(df) > 20:
                returns_data[code] = df['close'].pct_change().dropna()

        # 1. 计算单资产VaR
        for code, position_value in positions.items():
            if code in returns_data:
                var_result = self.var_calculator.calculate_var(
                    returns_data[code],
                    method=VaRMethod.HISTORICAL,
                    position_value=position_value
                )
                report.individual_risks[code] = var_result

        # 2. 计算组合VaR
        if len(returns_data) > 1:
            portfolio_returns = self._calculate_portfolio_returns(
                positions, returns_data, portfolio_value
            )
            if len(portfolio_returns) > 20:
                report.portfolio_var = self.var_calculator.calculate_var(
                    portfolio_returns,
                    method=VaRMethod.HISTORICAL,
                    position_value=portfolio_value
                )
                report.portfolio_cvar = report.portfolio_var.cvar

        # 3. 计算相关性矩阵
        if len(returns_data) > 1:
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            if len(returns_df) > 20:
                report.correlation_matrix = returns_df.corr()
                report.covariance_matrix = returns_df.cov() * 252  # 年化协方差

        # 4. 计算风险贡献
        report.risk_contribution = self._calculate_risk_contribution(
            positions, returns_data, report.correlation_matrix
        )

        # 5. 计算边际VaR
        report.marginal_var = self._calculate_marginal_var(
            positions, returns_data, portfolio_value
        )

        # 6. 压力测试
        report.stress_test_results = self._stress_test(
            positions, returns_data, portfolio_value
        )

        # 7. 计算风险指标
        if report.portfolio_var:
            portfolio_returns = self._calculate_portfolio_returns(
                positions, returns_data, portfolio_value
            )
            if len(portfolio_returns) > 20:
                report.sharpe_ratio = self._calculate_sharpe(portfolio_returns)
                report.sortino_ratio = self._calculate_sortino(portfolio_returns)
                report.max_drawdown = self._calculate_max_drawdown(portfolio_returns)
                report.calmar_ratio = self._calculate_calmar(portfolio_returns)

        # 8. 综合风险评级
        report.overall_risk_score, report.risk_rating = self._calculate_overall_risk(report)

        return report

    def _calculate_portfolio_returns(self, positions: Dict[str, float],
                                     returns_data: Dict[str, pd.Series],
                                     portfolio_value: float) -> pd.Series:
        """计算组合收益率"""
        weighted_returns = []
        common_index = None

        for code, position in positions.items():
            if code in returns_data:
                weight = position / portfolio_value
                ret = returns_data[code] * weight
                weighted_returns.append(ret)

                if common_index is None:
                    common_index = ret.index
                else:
                    common_index = common_index.intersection(ret.index)

        if common_index is not None and len(common_index) > 0:
            portfolio_returns = pd.Series(0, index=common_index)
            for wr in weighted_returns:
                portfolio_returns = portfolio_returns.add(wr, fill_value=0)
            return portfolio_returns

        return pd.Series()

    def _calculate_risk_contribution(self, positions: Dict[str, float],
                                     returns_data: Dict[str, pd.Series],
                                     correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """计算风险贡献"""
        if correlation_matrix.empty:
            return {}

        contributions = {}
        total_var = 0

        for code in positions:
            if code in correlation_matrix.index:
                # 简化计算：基于权重和相关性
                weight = positions[code] / sum(positions.values())
                if code in returns_data:
                    std = returns_data[code].std()
                    contributions[code] = weight * std * 100
                    total_var += contributions[code]

        # 归一化
        if total_var > 0:
            contributions = {k: v / total_var * 100 for k, v in contributions.items()}

        return contributions

    def _calculate_marginal_var(self, positions: Dict[str, float],
                                returns_data: Dict[str, pd.Series],
                                portfolio_value: float) -> Dict[str, float]:
        """计算边际VaR"""
        marginal_vars = {}

        for code, position in positions.items():
            if code in returns_data:
                # 边际VaR ≈ ∂VaR/∂w_i
                # 简化计算：使用VaR对权重的敏感度
                weight = position / portfolio_value
                std = returns_data[code].std()
                marginal_vars[code] = std * norm.ppf(self.confidence_level) * 100

        return marginal_vars

    def _stress_test(self, positions: Dict[str, float],
                     returns_data: Dict[str, pd.Series],
                     portfolio_value: float) -> Dict[str, float]:
        """压力测试"""
        scenarios = {
            "市场暴跌(-10%)": -0.10,
            "市场大跌(-5%)": -0.05,
            "市场震荡(-3%)": -0.03,
            "温和下跌(-2%)": -0.02,
            "极端情况(-20%)": -0.20,
            "流动性危机": -0.15,
        }

        results = {}

        # 基于历史波动率调整
        avg_volatility = np.mean([r.std() for r in returns_data.values()]) if returns_data else 0.02

        for scenario_name, base_shock in scenarios.items():
            # 考虑个股波动率差异
            scenario_loss = 0
            for code, position in positions.items():
                if code in returns_data:
                    individual_vol = returns_data[code].std()
                    beta = individual_vol / avg_volatility if avg_volatility > 0 else 1
                    scenario_loss += position * base_shock * beta
                else:
                    scenario_loss += position * base_shock

            results[scenario_name] = round(scenario_loss, 2)

        return results

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        excess_return = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        if volatility > 0:
            return round(excess_return / volatility, 4)
        return 0.0

    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        excess_return = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        if downside_std > 0:
            return round(excess_return / downside_std, 4)
        return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return round(drawdown.min() * 100, 4)

    def _calculate_calmar(self, returns: pd.Series) -> float:
        """计算卡玛比率"""
        annual_return = returns.mean() * 252
        max_dd = abs(self._calculate_max_drawdown(returns)) / 100
        if max_dd > 0:
            return round(annual_return / max_dd, 4)
        return 0.0

    def _calculate_overall_risk(self, report: PortfolioRiskReport) -> Tuple[float, str]:
        """计算综合风险评分 — 加权组合波动率/回撤/夏普/VaR"""
        # 各风险维度的评分（0-100，越高=风险越大）
        dimension_scores = {}

        # 1. VaR风险评分（权重0.25）
        if report.portfolio_var:
            dimension_scores['var'] = report.portfolio_var.risk_score

        # 2. 最大回撤评分（权重0.30）— 回撤越大风险越高
        if report.max_drawdown and report.max_drawdown != 0:
            dd_pct = abs(report.max_drawdown)
            if dd_pct < 3:
                dimension_scores['drawdown'] = 15
            elif dd_pct < 5:
                dimension_scores['drawdown'] = 30
            elif dd_pct < 8:
                dimension_scores['drawdown'] = 50
            elif dd_pct < 10:
                dimension_scores['drawdown'] = 65
            elif dd_pct < 15:
                dimension_scores['drawdown'] = 80
            elif dd_pct < 20:
                dimension_scores['drawdown'] = 90
            else:
                dimension_scores['drawdown'] = 95

        # 3. 夏普比率评分（权重0.25）— 夏普越低风险越高
        if report.sharpe_ratio != 0:
            sharpe = report.sharpe_ratio
            if sharpe > 2.0:
                dimension_scores['sharpe'] = 10
            elif sharpe > 1.0:
                dimension_scores['sharpe'] = 20
            elif sharpe > 0.5:
                dimension_scores['sharpe'] = 35
            elif sharpe > 0:
                dimension_scores['sharpe'] = 50
            elif sharpe > -0.5:
                dimension_scores['sharpe'] = 70
            elif sharpe > -1.0:
                dimension_scores['sharpe'] = 85
            else:
                dimension_scores['sharpe'] = 95

        # 4. 压力测试评分（权重0.20）
        if report.stress_test_results:
            extreme_loss = abs(report.stress_test_results.get("极端情况(-20%)", 0))
            dimension_scores['stress'] = min(100, extreme_loss / report.portfolio_value * 500)

        # 加权求和
        weights = {
            'var': 0.25,
            'drawdown': 0.30,
            'sharpe': 0.25,
            'stress': 0.20,
        }
        total_weight = sum(weights[k] for k in dimension_scores if k in weights)
        if total_weight > 0 and dimension_scores:
            overall = sum(dimension_scores[k] * weights.get(k, 0) for k in dimension_scores) / total_weight
        else:
            return 50.0, "中等风险"

        if overall <= 30:
            rating = "低风险"
        elif overall <= 50:
            rating = "中低风险"
        elif overall <= 70:
            rating = "中等风险"
        elif overall <= 85:
            rating = "中高风险"
        else:
            rating = "高风险"

        return round(overall, 2), rating


class RiskMonitor:
    """风险监控器"""

    def __init__(self, var_calculator: VaRCalculator = None):
        self.var_calculator = var_calculator or VaRCalculator()
        self.risk_history = []

    def check_risk_limits(self, var_result: VaRResult,
                          var_limit: float = 5.0,
                          cvar_limit: float = 7.0) -> Dict[str, Any]:
        """
        检查风险限额

        Args:
            var_result: VaR计算结果
            var_limit: VaR限额（百分比）
            cvar_limit: CVaR限额（百分比）

        Returns:
            检查结果
        """
        alerts = []

        if var_result.var_value > var_limit:
            alerts.append({
                'type': 'VAR_BREACH',
                'severity': 'HIGH' if var_result.var_value > var_limit * 1.5 else 'MEDIUM',
                'message': f"VaR ({var_result.var_value:.2f}%) 超过限额 ({var_limit:.2f}%)",
                'current_value': var_result.var_value,
                'limit': var_limit
            })

        if var_result.cvar > cvar_limit:
            alerts.append({
                'type': 'CVAR_BREACH',
                'severity': 'HIGH' if var_result.cvar > cvar_limit * 1.5 else 'MEDIUM',
                'message': f"CVaR ({var_result.cvar:.2f}%) 超过限额 ({cvar_limit:.2f}%)",
                'current_value': var_result.cvar,
                'limit': cvar_limit
            })

        return {
            'is_breach': len(alerts) > 0,
            'alerts': alerts,
            'var_status': 'BREACH' if var_result.var_value > var_limit else 'OK',
            'cvar_status': 'BREACH' if var_result.cvar > cvar_limit else 'OK'
        }

    def generate_risk_report(self, report: PortfolioRiskReport) -> str:
        """生成风险报告"""
        text = f"""
# 📊 风险度量报告

> 计算时间：{report.calculation_time.strftime('%Y-%m-%d %H:%M')}
> 组合价值：{report.portfolio_value:,.2f} 元
> 综合风险评级：{report.risk_rating} ({report.overall_risk_score:.1f}分)

---

## 一、VaR风险度量

"""

        if report.portfolio_var:
            var = report.portfolio_var
            text += f"""
### 组合VaR

| 指标 | 数值 |
|------|------|
| **VaR ({var.confidence_level:.0%})** | {var.var_value:.2f}% ({var.var_absolute:,.2f}元) |
| **CVaR (Expected Shortfall)** | {var.cvar:.2f}% |
| **置信区间** | [{var.var_range[0]:.2f}%, {var.var_range[1]:.2f}%] |
| **计算方法** | {var.method} |
| **时间跨度** | {var.time_horizon}天 |

### 收益分布特征

| 参数 | 数值 |
|------|------|
| 平均收益率 | {var.mean_return:.4f}% |
| 收益率标准差 | {var.std_return:.4f}% |
| 偏度 | {var.skewness:.4f} |
| 峰度 | {var.kurtosis:.4f} |

"""

        # 单资产风险
        if report.individual_risks:
            text += "\n### 单资产VaR\n\n"
            text += "| 代码 | VaR(%) | CVaR(%) | 风险等级 |\n"
            text += "|------|--------|---------|----------|\n"
            for code, var in report.individual_risks.items():
                text += f"| {code} | {var.var_value:.2f} | {var.cvar:.2f} | {var.risk_level} |\n"

        # 风险指标
        text += f"""
---

## 二、风险指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **夏普比率** | {report.sharpe_ratio:.4f} | 风险调整后收益 |
| **索提诺比率** | {report.sortino_ratio:.4f} | 下行风险调整收益 |
| **最大回撤** | {report.max_drawdown:.2f}% | 历史最大亏损 |
| **卡玛比率** | {report.calmar_ratio:.4f} | 收益/最大回撤 |

"""

        # 压力测试
        if report.stress_test_results:
            text += "\n---\n\n## 三、压力测试\n\n"
            text += "| 情景 | 预期损失 |\n"
            text += "|------|----------|\n"
            for scenario, loss in report.stress_test_results.items():
                text += f"| {scenario} | {loss:,.2f}元 ({loss/report.portfolio_value*100:.2f}%) |\n"

        # 风险贡献
        if report.risk_contribution:
            text += "\n---\n\n## 四、风险贡献\n\n"
            text += "| 资产 | 贡献度 |\n"
            text += "|------|--------|\n"
            for code, contrib in sorted(report.risk_contribution.items(),
                                        key=lambda x: x[1], reverse=True):
                text += f"| {code} | {contrib:.2f}% |\n"

        # 相关性矩阵
        if not report.correlation_matrix.empty:
            text += "\n---\n\n## 五、相关性矩阵\n\n"
            text += "```\n"
            text += report.correlation_matrix.round(2).to_string()
            text += "\n```\n"

        text += """

---

## ⚠️ 风险提示

1. VaR表示在给定置信水平下，投资组合在特定时间内的最大可能损失
2. CVaR（Expected Shortfall）表示超过VaR阈值时的平均损失
3. 历史模拟法基于历史数据，可能无法反映极端市场情况
4. 请结合其他风险指标综合评估投资风险
"""

        return text


def calculate_single_stock_var(price_data: pd.DataFrame,
                               confidence_level: float = 0.95,
                               position_value: float = 100000) -> VaRResult:
    """
    计算单只股票VaR的便捷函数

    Args:
        price_data: 价格数据DataFrame
        confidence_level: 置信水平
        position_value: 持仓金额

    Returns:
        VaRResult
    """
    calculator = VaRCalculator(confidence_level=confidence_level)
    returns = price_data['close'].pct_change().dropna()
    return calculator.calculate_var(returns, method=VaRMethod.HISTORICAL,
                                   position_value=position_value)


if __name__ == "__main__":
    # 测试VaR计算
    print("Testing VaR Calculator...")

    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')

    # 模拟股票价格（带趋势的随机游走）
    returns = np.random.normal(0.0005, 0.02, 252)
    prices = 100 * np.cumprod(1 + returns)

    test_df = pd.DataFrame({
        'date': dates,
        'close': prices
    })

    # 计算VaR
    calculator = VaRCalculator(confidence_level=0.95)
    returns_series = test_df['close'].pct_change().dropna()

    # 测试不同方法
    for method in VaRMethod:
        result = calculator.calculate_var(returns_series, method=method)
        print(f"\n{method.value}:")
        print(f"  VaR: {result.var_value:.2f}%")
        print(f"  CVaR: {result.cvar:.2f}%")
        print(f"  Risk Level: {result.risk_level}")

    # 测试组合风险分析
    print("\n\nTesting Portfolio Risk Analyzer...")

    positions = {
        'stock1': 50000,
        'stock2': 30000,
        'stock3': 20000
    }

    price_data = {}
    for i, code in enumerate(positions.keys()):
        np.random.seed(42 + i)
        ret = np.random.normal(0.0005, 0.02 * (1 + i * 0.2), 252)
        prices = 100 * np.cumprod(1 + ret)
        price_data[code] = pd.DataFrame({
            'date': dates,
            'close': prices
        })

    analyzer = PortfolioRiskAnalyzer()
    report = analyzer.analyze_portfolio(positions, price_data)

    print(f"\nPortfolio Risk Report:")
    print(f"  Total Value: {report.portfolio_value:,.2f}")
    print(f"  Risk Rating: {report.risk_rating}")
    print(f"  Sharpe Ratio: {report.sharpe_ratio:.4f}")
    print(f"  Max Drawdown: {report.max_drawdown:.2f}%")

    # 生成报告
    monitor = RiskMonitor()
    print(monitor.generate_risk_report(report))
