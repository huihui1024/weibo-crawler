"""
机器学习预测模型
基于 RandomForest 进行股票走势预测（分类+回归双模型）
特征工程：15个技术因子，从K线数据自动提取
训练数据：通过腾讯K线API采集，时间序列切分避免未来信息泄漏
"""
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)


@dataclass
class MLPredictionResult:
    """ML预测结果"""
    code: str
    name: str
    prediction_date: date
    prediction_time: datetime

    # 预测结果
    predicted_trend: str       # up / down / flat
    predicted_change_pct: float  # 预测涨跌幅（回归值）
    predicted_direction_prob: float  # 方向概率（分类置信度）
    confidence_interval: Tuple[float, float]  # 涨跌幅置信区间

    # 模型信息
    model_type: str = "random_forest"
    confidence: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)


class FeatureEngineer:
    """技术因子特征工程 — 从K线数据提取15个因子"""

    FEATURE_NAMES = [
        'return_1d',       # 当日涨跌幅
        'return_5d',       # 5日累计涨跌幅
        'return_10d',      # 10日累计涨跌幅
        'ma5_ratio',       # close/MA5 - 1
        'ma10_ratio',      # close/MA10 - 1
        'ma20_ratio',      # close/MA20 - 1
        'ma_cross',        # 均线多头排列编码 (0-3)
        'volatility_5d',   # 5日收益率标准差
        'volatility_20d',  # 20日收益率标准差
        'volume_ratio',    # 当日量/MA5量
        'rsi_14',          # 14日RSI
        'macd_signal',     # MACD信号方向 (-1, 0, 1)
        'amplitude',       # 当日振幅
        'volume_change',   # 量变化率
        'price_position',  # 价格在20日高低区间的位置
    ]

    def extract_from_klines(self, klines: List[Dict]) -> pd.DataFrame:
        """从K线数据提取特征DataFrame

        Args:
            klines: [{date, open, close, high, low, volume}, ...]

        Returns:
            DataFrame with FEATURE_NAMES columns + 'next_return' label column
        """
        if len(klines) < 25:
            return pd.DataFrame()

        df = pd.DataFrame(klines)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # 基础收益率
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)

        # 均线偏离
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma5_ratio'] = df['close'] / df['ma5'] - 1
        df['ma10_ratio'] = df['close'] / df['ma10'] - 1
        df['ma20_ratio'] = df['close'] / df['ma20'] - 1

        # 均线排列编码
        df['ma_cross'] = 0
        cond3 = (df['close'] > df['ma5']) & (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])
        cond2 = (df['close'] > df['ma5']) & (df['ma5'] > df['ma10'])
        cond1 = (df['close'] > df['ma5'])
        df.loc[cond3, 'ma_cross'] = 3
        df.loc[cond2 & ~cond3, 'ma_cross'] = 2
        df.loc[cond1 & ~cond2, 'ma_cross'] = 1

        # 波动率
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()

        # 量比
        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['vol_ma5'].replace(0, np.nan)
        df['volume_change'] = df['volume'].pct_change()

        # RSI
        df['rsi_14'] = self._calc_rsi(df['close'], 14)

        # MACD信号方向
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        df['macd_signal'] = 0
        df.loc[macd > signal, 'macd_signal'] = 1
        df.loc[macd < signal, 'macd_signal'] = -1

        # 振幅
        df['prev_close'] = df['close'].shift(1)
        df['amplitude'] = (df['high'] - df['low']) / df['prev_close'].replace(0, np.nan)

        # 价格位置
        df['low_20'] = df['low'].rolling(20).min()
        df['high_20'] = df['high'].rolling(20).max()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20']).replace(0, np.nan)

        # 标签：次日涨跌幅
        df['next_return'] = df['close'].pct_change(1).shift(-1)

        # 选取特征列
        feature_cols = self.FEATURE_NAMES + ['next_return']
        result = df[feature_cols].copy()
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.dropna()

        return result

    def extract_single_day(self, klines: List[Dict]) -> Optional[np.ndarray]:
        """提取最近一天的特征向量（用于实时预测）

        Args:
            klines: 至少25天的K线数据

        Returns:
            特征向量 (15,) 或 None
        """
        df = self.extract_from_klines(klines)
        if df.empty:
            return None
        return df[self.FEATURE_NAMES].iloc[-1].values

    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)


class StockMLModel:
    """RandomForest 双模型（分类+回归）"""

    def __init__(self):
        self.clf = None  # 分类模型: 涨/平/跌
        self.reg = None  # 回归模型: 涨跌幅精确值
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.train_metrics = {}

    def train(self, all_kline_data: Dict[str, List[Dict]]) -> Dict:
        """训练模型

        Args:
            all_kline_data: {code: [kline_dict, ...]} 历史K线数据

        Returns:
            训练指标
        """
        # 1. 构建特征/标签
        all_features = []
        all_labels_clf = []
        all_labels_reg = []

        for code, klines in all_kline_data.items():
            df = self.feature_engineer.extract_from_klines(klines)
            if df.empty or len(df) < 20:
                continue

            features = df[FeatureEngineer.FEATURE_NAMES].values
            next_returns = df['next_return'].values

            # 分类标签: 1=涨(>0.5%), 0=平, -1=跌(<-0.5%)
            clf_labels = np.zeros(len(next_returns), dtype=int)
            clf_labels[next_returns > 0.005] = 1
            clf_labels[next_returns < -0.005] = -1

            all_features.append(features)
            all_labels_clf.append(clf_labels)
            all_labels_reg.append(next_returns * 100)  # 转为百分比

        if not all_features:
            return {'status': 'error', 'message': 'No valid training data'}

        X = np.vstack(all_features)
        y_clf = np.concatenate(all_labels_clf)
        y_reg = np.concatenate(all_labels_reg)

        print(f"训练数据: {len(X)} 样本, {X.shape[1]} 特征")
        print(f"分类分布: 跌={sum(y_clf==-1)}, 平={sum(y_clf==0)}, 涨={sum(y_clf==1)}")

        # 2. 时间序列切分（前70%训练，后30%测试）
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_clf_train, y_clf_test = y_clf[:split_idx], y_clf[split_idx:]
        y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]

        # 3. 标准化
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # 4. 训练分类模型
        self.clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1, class_weight='balanced'
        )
        self.clf.fit(X_train_s, y_clf_train)

        y_clf_pred = self.clf.predict(X_test_s)
        clf_acc = accuracy_score(y_clf_test, y_clf_pred)
        clf_report = classification_report(y_clf_test, y_clf_pred, output_dict=True)

        # 5. 训练回归模型
        self.reg = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        )
        self.reg.fit(X_train_s, y_reg_train)

        y_reg_pred = self.reg.predict(X_test_s)
        reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
        reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
        reg_r2 = r2_score(y_reg_test, y_reg_pred)

        self.is_trained = True

        # 6. 评估报告
        self.train_metrics = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'clf_accuracy': round(clf_acc, 4),
            'clf_report': clf_report,
            'reg_mae': round(reg_mae, 4),
            'reg_rmse': round(reg_rmse, 4),
            'reg_r2': round(reg_r2, 4),
            'feature_importance': dict(zip(
                FeatureEngineer.FEATURE_NAMES,
                self.reg.feature_importances_
            ))
        }

        print(f"\n=== 训练结果 ===")
        print(f"分类准确率: {clf_acc:.2%}")
        print(f"回归 MAE: {reg_mae:.3f}%, RMSE: {reg_rmse:.3f}%, R²: {reg_r2:.3f}")
        print(f"\nTop5 重要特征:")
        sorted_imp = sorted(self.train_metrics['feature_importance'].items(), key=lambda x: -x[1])
        for name, imp in sorted_imp[:5]:
            print(f"  {name}: {imp:.4f}")

        return self.train_metrics

    def predict(self, klines: List[Dict], code: str = '', name: str = '') -> Optional[MLPredictionResult]:
        """预测次日走势

        Args:
            klines: 最近25+天的K线数据
            code: 股票代码
            name: 股票名称

        Returns:
            MLPredictionResult with confidence interval
        """
        if not self.is_trained:
            return None

        features = self.feature_engineer.extract_single_day(klines)
        if features is None:
            return None

        X = self.scaler.transform(features.reshape(1, -1))

        # 分类预测
        direction_proba = self.clf.predict_proba(X)[0]
        direction_classes = self.clf.classes_
        direction_map = {c: p for c, p in zip(direction_classes, direction_proba)}
        up_prob = direction_map.get(1, 0)
        down_prob = direction_map.get(-1, 0)

        if up_prob > down_prob and up_prob > 0.4:
            predicted_trend = 'up'
        elif down_prob > up_prob and down_prob > 0.4:
            predicted_trend = 'down'
        else:
            predicted_trend = 'flat'

        predicted_direction_prob = max(up_prob, down_prob)

        # 回归预测 + 置信区间
        tree_predictions = np.array([
            tree.predict(X)[0] for tree in self.reg.estimators_
        ])
        predicted_change = np.mean(tree_predictions)
        pred_std = np.std(tree_predictions)

        confidence_interval = (
            round(predicted_change - 1.96 * pred_std, 2),
            round(predicted_change + 1.96 * pred_std, 2)
        )

        # 综合置信度
        direction_confidence = max(direction_proba)
        spread_confidence = min(1.0, 0.5 + abs(predicted_change) / 5)
        confidence = round(direction_confidence * 0.6 + spread_confidence * 0.4, 3)

        return MLPredictionResult(
            code=code,
            name=name,
            prediction_date=date.today() + timedelta(days=1),
            prediction_time=datetime.now(),
            predicted_trend=predicted_trend,
            predicted_change_pct=round(predicted_change, 2),
            predicted_direction_prob=round(predicted_direction_prob, 3),
            confidence_interval=confidence_interval,
            model_type='random_forest',
            confidence=confidence,
            feature_importance=self.train_metrics.get('feature_importance', {})
        )

    def save(self, path: str = 'data/ml_model.pkl'):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        model_data = {
            'clf': self.clf,
            'reg': self.reg,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'train_metrics': self.train_metrics,
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存: {path}")

    def load(self, path: str = 'data/ml_model.pkl') -> bool:
        """加载模型"""
        if not os.path.exists(path):
            return False

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.clf = model_data.get('clf')
        self.reg = model_data.get('reg')
        self.scaler = model_data.get('scaler', StandardScaler())
        self.is_trained = model_data.get('is_trained', False)
        self.train_metrics = model_data.get('train_metrics', {})
        return True


def get_ml_model(path: str = 'data/ml_model.pkl') -> StockMLModel:
    """获取全局ML模型实例（自动加载已训练模型）"""
    model = StockMLModel()
    if model.load(path):
        print(f"ML模型已加载 (训练指标: 分类准确率={model.train_metrics.get('clf_accuracy', 'N/A')})")
    return model
