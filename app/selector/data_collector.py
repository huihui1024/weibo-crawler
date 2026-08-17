"""
历史数据采集与特征构建模块
通过腾讯财经K线API批量获取股票历史数据，构建训练特征/标签对
"""
import json
import os
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import sqlite3


def get_stock_pool_codes(db_path: str = None) -> List[Dict]:
    """从数据库获取股票池中所有活跃股票代码"""
    if db_path is None:
        from ..config import get_config
        db_path = str(Path(get_config().storage.database))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT stock_code, stock_name FROM stock_pool WHERE status = 'active'
    """)
    rows = cursor.fetchall()
    conn.close()

    stocks = []
    for code, name in rows:
        stocks.append({'code': code, 'name': name})

    # 补充沪深300核心标的（确保训练数据量足够）
    if len(stocks) < 30:
        core_codes = [
            ('600519', '贵州茅台'), ('000858', '五粮液'), ('601318', '中国平安'),
            ('600036', '招商银行'), ('000001', '平安银行'), ('600000', '浦发银行'),
            ('601398', '工商银行'), ('600276', '恒瑞医药'), ('000333', '美的集团'),
            ('002415', '海康威视'), ('300750', '宁德时代'), ('601012', '隆基绿能'),
            ('000725', '京东方A'), ('002230', '科大讯飞'), ('300059', '东方财富'),
            ('600900', '长江电力'), ('601888', '中国中免'), ('000568', '泸州老窖'),
            ('002475', '立讯精密'), ('600809', '山西汾酒'), ('300308', '中际旭创'),
            ('002371', '北方华创'), ('688981', '中芯国际'), ('601899', '紫金矿业'),
            ('600030', '中信证券'), ('000651', '格力电器'), ('002594', '比亚迪'),
            ('601728', '中国电信'), ('600887', '伊利股份'), ('603259', '药明康德'),
        ]
        existing_codes = {s['code'] for s in stocks}
        for code, name in core_codes:
            if code not in existing_codes:
                stocks.append({'code': code, 'name': name})

    return stocks


def code_to_tencent(code: str) -> str:
    """转换股票代码为腾讯格式: 600xxx -> sh600xxx, 000/300xxx -> sz000xxx"""
    pure = code.replace('sh', '').replace('sz', '').replace('SH', '').replace('SZ', '')
    if pure.startswith('6') or pure.startswith('9'):
        return f'sh{pure}'
    return f'sz{pure}'


def fetch_kline_data(code: str, days: int = 250) -> List[Dict]:
    """通过腾讯K线API获取单只股票历史数据

    Args:
        code: 股票代码 (如 600519)
        days: 获取天数

    Returns:
        List of dicts: [{date, open, close, high, low, volume}, ...]
    """
    tencent_code = code_to_tencent(code)
    start = (date.today() - timedelta(days=days + 60)).strftime('%Y-%m-%d')
    end = date.today().strftime('%Y-%m-%d')
    url = (
        f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
        f'?param={tencent_code},day,{start},{end},{days},qfq'
    )

    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        stock_data = data.get('data', {}).get(tencent_code, {})
        raw = stock_data.get('qfqday', [])
        if not raw:
            raw = stock_data.get('day', [])

        result = []
        for k in raw:
            result.append({
                'date': k[0],
                'open': float(k[1]),
                'close': float(k[2]),
                'high': float(k[3]),
                'low': float(k[4]),
                'volume': float(k[5]) if len(k) > 5 else 0,
            })
        return result
    except Exception as e:
        print(f"  获取 {code} K线失败: {e}")
        return []


def collect_training_data(
    stock_codes: Optional[List[str]] = None,
    days: int = 250,
    db_path: str = None,
) -> Dict[str, List[Dict]]:
    """批量采集训练数据

    Args:
        stock_codes: 股票代码列表，None则用股票池+核心标的
        days: 每只股票获取天数

    Returns:
        Dict[code, List[kline_dict]]
    """
    if stock_codes is None:
        stocks = get_stock_pool_codes(db_path)
        stock_codes = [s['code'] for s in stocks]

    print(f"开始采集 {len(stock_codes)} 只股票的历史数据...")

    all_data = {}
    success = 0
    for i, code in enumerate(stock_codes):
        klines = fetch_kline_data(code, days)
        if len(klines) >= 60:
            all_data[code] = klines
            success += 1
        if (i + 1) % 10 == 0:
            print(f"  已采集 {i + 1}/{len(stock_codes)}, 成功 {success}")

    print(f"采集完成: {success}/{len(stock_codes)} 只股票有足够数据")
    return all_data


def save_raw_data(data: Dict[str, List[Dict]], path: str = 'data/kline_raw.pkl'):
    """保存原始K线数据"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"原始数据已保存: {path} ({len(data)} 只股票)")


def load_raw_data(path: str = 'data/kline_raw.pkl') -> Dict[str, List[Dict]]:
    """加载原始K线数据"""
    with open(path, 'rb') as f:
        return pickle.load(f)
