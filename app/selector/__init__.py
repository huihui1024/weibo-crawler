"""
选股模块

包含：
- StockSelector: 股票筛选器
- Predictor: 预测器
- MultiFactorModel: 多因子选股模型
- BacktestEngine: 回测引擎
- BacktestRunner: 回测运行器
- StyleDriftTracker: 风格漂移追踪器
- RebalanceSignalGenerator: 调仓信号生成器
- StyleDriftReporter: 风格漂移报告生成器
- MLPredictor: 机器学习预测器 (LSTM/Transformer) — 需要 sklearn
- VaRCalculator: VaR风险度量计算器 — 需要 sklearn
"""
from .stock_selector import StockSelector
from .predictor import Predictor
from .multi_factor_model import (
    MultiFactorModel,
    StockScore,
    FactorScore,
    TrendFactor,
    MomentumFactor,
    VolumeFactor,
    MoneyFlowFactor,
    VolatilityFactor,
    SentimentFactor,
    prepare_stock_data_for_scoring
)
from .backtest import (
    BacktestEngine,
    BacktestRunner,
    BacktestResult,
    BacktestTrade,
    BacktestPosition,
    BacktestStrategy,
    PredictionBasedStrategy,
    MultiFactorStrategy,
    generate_backtest_report,
    run_prediction_backtest
)
from .style_drift_tracker import (
    StyleDriftTracker,
    PositionSnapshot,
    StyleMetrics,
    StyleDriftEvent,
    PositionAdjustment,
    RebalanceSignal,
    RebalanceSignalGenerator,
    StyleDriftReporter,
    get_style_tracker
)

# Optional imports — require sklearn, fail gracefully if not installed
try:
    from .ml_predictor import (
        MLPredictor,
        MLPredictionResult,
        FeatureEngineer,
        generate_ml_prediction_report
    )
except ImportError:
    MLPredictor = None
    MLPredictionResult = None
    FeatureEngineer = None
    generate_ml_prediction_report = None

try:
    from .risk_metrics import (
        VaRCalculator,
        VaRResult,
        VaRMethod,
        PortfolioRiskAnalyzer,
        PortfolioRiskReport,
        RiskMonitor,
        calculate_single_stock_var
    )
except ImportError:
    VaRCalculator = None
    VaRResult = None
    VaRMethod = None
    PortfolioRiskAnalyzer = None
    PortfolioRiskReport = None
    RiskMonitor = None
    calculate_single_stock_var = None

__all__ = [
    'StockSelector',
    'Predictor',
    'MultiFactorModel',
    'StockScore',
    'FactorScore',
    'TrendFactor',
    'MomentumFactor',
    'VolumeFactor',
    'MoneyFlowFactor',
    'VolatilityFactor',
    'SentimentFactor',
    'prepare_stock_data_for_scoring',
    'BacktestEngine',
    'BacktestRunner',
    'BacktestResult',
    'BacktestTrade',
    'BacktestPosition',
    'BacktestStrategy',
    'PredictionBasedStrategy',
    'MultiFactorStrategy',
    'generate_backtest_report',
    'run_prediction_backtest',
    # 风格漂移追踪
    'StyleDriftTracker',
    'PositionSnapshot',
    'StyleMetrics',
    'StyleDriftEvent',
    'PositionAdjustment',
    'RebalanceSignal',
    'RebalanceSignalGenerator',
    'StyleDriftReporter',
    'get_style_tracker',
    # 机器学习预测（可选）
    'MLPredictor',
    'MLPredictionResult',
    'FeatureEngineer',
    'generate_ml_prediction_report',
    # 风险度量（可选）
    'VaRCalculator',
    'VaRResult',
    'VaRMethod',
    'PortfolioRiskAnalyzer',
    'PortfolioRiskReport',
    'RiskMonitor',
    'calculate_single_stock_var',
]