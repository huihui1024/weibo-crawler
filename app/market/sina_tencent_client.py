"""
新浪/腾讯直连行情数据客户端
不依赖 akshare/efinance，直接调用新浪/腾讯 HTTP API
在此服务器网络环境下，东方财富 API 被阻断，新浪/腾讯 API 正常可用
"""
import re
import time
import json
import requests
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .base_client import MarketClientBase, MarketType


class SinaTencentClient(MarketClientBase):
    """
    基于新浪+腾讯直连 API 的行情客户端
    - 新浪 hq.sinajs.cn：实时报价（速度极快，单次可查多只股票）
    - 腾讯 web.ifzq.gtimg.cn：K线历史数据
    - 腾讯 qt.gtimg.cn：实时报价（备用）
    """
    market_type = MarketType.A_SHARE

    # A股指数代码映射
    INDEX_MAP = {
        '000001': 'sh000001',  # 上证指数
        '399001': 'sz399001',  # 深证成指
        '399006': 'sz399006',  # 创业板指
    }

    def __init__(self):
        super().__init__()
        self._request_interval = 0.5
        self._session = requests.Session()
        self._session.headers.update({
            'Referer': 'http://finance.sina.com.cn',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })
        self._realtime_cache = None
        self._cache_time = None

    def _code_to_sina(self, code: str) -> str:
        """将股票代码转为新浪格式 (sh600519 / sz000001)"""
        code = str(code).strip()
        if code in self.INDEX_MAP:
            return self.INDEX_MAP[code]
        # 6/9开头为沪市主板/科创板，0开头为深市主板，3开头为创业板，8/4开头为北交所
        if code.startswith(('6', '9')):
            return f'sh{code}'
        if code.startswith(('0', '3', '8', '4')):
            return f'sz{code}'
        # 默认按沪市处理
        return f'sh{code}'

    def _code_to_tencent(self, code: str) -> str:
        """将股票代码转为腾讯格式 (sh600519 / sz000001)"""
        return self._code_to_sina(code)

    def _parse_sina_quote(self, raw: str) -> Dict[str, Any]:
        """解析新浪行情数据"""
        # 格式: var hq_str_sh600519="贵州茅台,开盘价,昨收,当前价,..."
        result = {}
        for line in raw.strip().split('\n'):
            if not line.strip():
                continue
            match = re.match(r'var hq_str_(s[hz]\d+)="(.+)"', line)
            if not match:
                continue
            full_code = match.group(1)
            fields = match.group(2).split(',')
            if len(fields) < 32:
                continue
            code = full_code[2:]  # 去掉 sh/sz 前缀
            try:
                result[code] = {
                    'code': code,
                    'name': fields[0],
                    'open': self._safe_float(fields[1]),
                    'prev_close': self._safe_float(fields[2]),
                    'price': self._safe_float(fields[3]),
                    'high': self._safe_float(fields[4]),
                    'low': self._safe_float(fields[5]),
                    'volume': self._safe_float(fields[8]),
                    'amount': self._safe_float(fields[9]),
                    'change': self._safe_float(fields[3]) - self._safe_float(fields[2]),
                    'change_pct': round(
                        (self._safe_float(fields[3]) - self._safe_float(fields[2]))
                        / self._safe_float(fields[2]) * 100, 2
                    ) if self._safe_float(fields[2]) > 0 else 0,
                    'bid1_vol': self._safe_float(fields[10]),
                    'bid1': self._safe_float(fields[11]),
                    'bid2_vol': self._safe_float(fields[12]),
                    'bid2': self._safe_float(fields[13]),
                    'bid3_vol': self._safe_float(fields[14]),
                    'bid3': self._safe_float(fields[15]),
                    'bid4_vol': self._safe_float(fields[16]),
                    'bid4': self._safe_float(fields[17]),
                    'bid5_vol': self._safe_float(fields[18]),
                    'bid5': self._safe_float(fields[19]),
                    'ask1_vol': self._safe_float(fields[20]),
                    'ask1': self._safe_float(fields[21]),
                    'ask2_vol': self._safe_float(fields[22]),
                    'ask2': self._safe_float(fields[23]),
                    'ask3_vol': self._safe_float(fields[24]),
                    'ask3': self._safe_float(fields[25]),
                    'ask4_vol': self._safe_float(fields[26]),
                    'ask4': self._safe_float(fields[27]),
                    'ask5_vol': self._safe_float(fields[28]),
                    'ask5': self._safe_float(fields[29]),
                    'date': fields[30],
                    'time': fields[31],
                    'market': 'A_SHARE',
                    'source': 'sina',
                    'timestamp': datetime.now().isoformat(),
                }
            except (IndexError, ValueError):
                continue
        return result

    def _parse_tencent_quote(self, raw: str) -> Dict[str, Any]:
        """解析腾讯行情数据"""
        result = {}
        for line in raw.strip().split(';'):
            line = line.strip()
            if not line:
                continue
            match = re.match(r'v_(s[hz]\d+)="(.+)"', line)
            if not match:
                continue
            full_code = match.group(1)
            fields = match.group(2).split('~')
            if len(fields) < 45:
                continue
            code = full_code[2:]
            try:
                result[code] = {
                    'code': code,
                    'name': fields[1],
                    'price': self._safe_float(fields[3]),
                    'prev_close': self._safe_float(fields[4]),
                    'open': self._safe_float(fields[5]),
                    'volume': self._safe_float(fields[6]),
                    'high': self._safe_float(fields[33]) if len(fields) > 33 else 0,
                    'low': self._safe_float(fields[34]) if len(fields) > 34 else 0,
                    'amount': self._safe_float(fields[37]) if len(fields) > 37 else 0,
                    'change': self._safe_float(fields[31]) if len(fields) > 31 else 0,
                    'change_pct': self._safe_float(fields[32]) if len(fields) > 32 else 0,
                    'turnover_rate': self._safe_float(fields[38]) if len(fields) > 38 else 0,
                    'pe_ratio': self._safe_float(fields[39]) if len(fields) > 39 else 0,
                    'market_cap': self._safe_float(fields[45]) if len(fields) > 45 else 0,
                    'market': 'A_SHARE',
                    'source': 'tencent',
                    'timestamp': datetime.now().isoformat(),
                }
            except (IndexError, ValueError):
                continue
        return result

    @staticmethod
    def _safe_float(val) -> float:
        if val is None:
            return 0.0
        try:
            if isinstance(val, str):
                val = val.strip().replace(',', '')
                if val in ('-', '', 'N/A', '--'):
                    return 0.0
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def get_stock_price(self, code: str) -> Dict[str, Any]:
        """获取单只股票实时行情"""
        sina_code = self._code_to_sina(code)
        try:
            resp = self._session.get(
                f'http://hq.sinajs.cn/list={sina_code}',
                timeout=5
            )
            resp.encoding = 'gbk'
            data = self._parse_sina_quote(resp.text)
            if code in data:
                return data[code]
        except Exception as e:
            print(f"新浪行情失败 {code}: {e}")

        # 备用：腾讯
        try:
            tencent_code = self._code_to_tencent(code)
            resp = self._session.get(
                f'http://qt.gtimg.cn/q={tencent_code}',
                timeout=5
            )
            resp.encoding = 'gbk'
            data = self._parse_tencent_quote(resp.text)
            if code in data:
                return data[code]
        except Exception as e:
            print(f"腾讯行情失败 {code}: {e}")

        return {'code': code, 'price': 0, 'change_pct': 0, 'source': 'none'}

    def get_realtime_quotes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取实时行情列表（基于新浪，按成交额排序取前N只）"""
        # 新浪不支持全市场快照，使用腾讯市场排行接口
        try:
            # 腾讯市场排行：沪深A股涨幅榜
            codes = self._get_top_stocks_by_tencent(limit)
            if codes:
                sina_codes = [self._code_to_sina(c) for c in codes]
                # 分批请求（新浪限制每批约800只）
                batch_size = 200
                all_data = {}
                for i in range(0, len(sina_codes), batch_size):
                    batch = sina_codes[i:i + batch_size]
                    try:
                        resp = self._session.get(
                            f'http://hq.sinajs.cn/list={",".join(batch)}',
                            timeout=10
                        )
                        resp.encoding = 'gbk'
                        all_data.update(self._parse_sina_quote(resp.text))
                    except Exception:
                        continue
                return sorted(all_data.values(), key=lambda x: x.get('amount', 0), reverse=True)[:limit]
        except Exception as e:
            print(f"实时行情获取失败: {e}")

        return []

    def _get_top_stocks_by_tencent(self, limit: int) -> List[str]:
        """通过腾讯接口获取热门股票代码列表"""
        try:
            # 使用腾讯股票排行接口
            resp = self._session.get(
                f'http://qt.gtimg.cn/q=fl_sh,fl_sz',
                timeout=5
            )
            resp.encoding = 'gbk'
            # 解析获取代码列表
            codes = re.findall(r's[hz]\d{6}', resp.text)
            return [c[2:] for c in codes[:limit]]
        except Exception:
            return []

    def get_stock_history(self, code: str, days: int = 30, adjust: str = "qfq") -> pd.DataFrame:
        """获取股票历史K线数据（腾讯接口）"""
        def _fetch():
            tencent_code = self._code_to_tencent(code)
            fqt = 1 if adjust == "qfq" else 2 if adjust == "hfq" else 0
            url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tencent_code},day,,,{days},{fqt}'
            resp = self._session.get(url, timeout=10)
            data = resp.json()

            # 解析腾讯K线数据
            stock_data = data.get('data', {}).get(tencent_code, {})
            day_data = stock_data.get('day') or stock_data.get('qfqday') or stock_data.get('hfqday') or []

            if not day_data:
                return pd.DataFrame()

            df = pd.DataFrame(day_data, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['amount'] = 0
            df['change_pct'] = df['close'].pct_change() * 100
            df['turnover_rate'] = 0
            return df.tail(days)

        return self._retry_request(_fetch)

    def get_market_overview(self) -> Dict[str, Any]:
        """获取市场概览（基于主要指数）"""
        try:
            # 获取上证、深证、创业板三大指数
            codes = ['sh000001', 'sz399001', 'sz399006']
            resp = self._session.get(
                f'http://hq.sinajs.cn/list={",".join(codes)}',
                timeout=5
            )
            resp.encoding = 'gbk'
            data = self._parse_sina_quote(resp.text)

            indices = {}
            for name, code in [('上证指数', '000001'), ('深证成指', '399001'), ('创业板指', '399006')]:
                if code in data:
                    indices[name] = data[code]

            # 判断市场强弱
            avg_change = sum(v.get('change_pct', 0) for v in indices.values()) / len(indices) if indices else 0

            return {
                'indices': {k: {'price': v['price'], 'change_pct': v['change_pct'], 'name': v['name']}
                           for k, v in indices.items()},
                'avg_change': round(avg_change, 2),
                'market_strength': 'strong' if avg_change > 1 else 'weak' if avg_change < -1 else 'neutral',
                'source': 'sina',
            }
        except Exception as e:
            print(f"市场概览获取失败: {e}")
            return {}

    def detect_market_regime(self) -> str:
        """
        检测市场状态: bull / bear / oscillation

        基于: 沪深300近20日均线趋势 + 成交量变化
        - MA20趋势向上 + 量能放大 → bull
        - MA20趋势向下 + 量能萎缩 → bear
        - 其他 → oscillation
        """
        try:
            # 获取沪深300近30日K线
            df = self.get_stock_history('000300', days=30)
            if df.empty or len(df) < 20:
                return 'oscillation'

            # 计算20日均线
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma5'] = df['close'].rolling(window=5).mean()

            # 成交量变化率（近5日 vs 前5日）
            recent_vol = df['volume'].tail(5).mean()
            prev_vol = df['volume'].iloc[-10:-5].mean() if len(df) >= 10 else recent_vol
            vol_change = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0

            # MA趋势判断
            last_ma5 = df['ma5'].iloc[-1]
            last_ma20 = df['ma20'].iloc[-1]
            prev_ma20 = df['ma20'].iloc[-2] if len(df) > 1 else last_ma20

            # MA5 > MA20 且 MA20上升 → 上升趋势
            ma5_above_ma20 = last_ma5 > last_ma20 if not (pd.isna(last_ma5) or pd.isna(last_ma20)) else False
            ma20_rising = last_ma20 > prev_ma20 if not (pd.isna(last_ma20) or pd.isna(prev_ma20)) else False

            # 综合判断
            if ma5_above_ma20 and ma20_rising and vol_change > 0.1:
                return 'bull'
            elif not ma5_above_ma20 and not ma20_rising and vol_change < -0.1:
                return 'bear'
            else:
                return 'oscillation'

        except Exception as e:
            print(f"市场状态检测失败: {e}")
            return 'oscillation'

    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索股票（基于新浪实时数据，需先获取股票列表）"""
        # 新浪无搜索接口，使用腾讯搜索
        try:
            resp = self._session.get(
                f'http://smartbox.gtimg.cn/s3/?q={keyword}&t=all',
                timeout=5
            )
            resp.encoding = 'gbk'
            results = []
            for line in resp.text.strip().split(';'):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'v_(s[hz]\d+)="(.+)"', line)
                if not match:
                    continue
                full_code = match.group(1)
                fields = match.group(2).split('~')
                if len(fields) >= 2:
                    code = full_code[2:]
                    results.append({
                        'code': code,
                        'name': fields[1] if len(fields) > 1 else '',
                        'market': 'A_SHARE',
                    })
                    if len(results) >= 20:
                        break
            return results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def get_stock_technical_indicators(self, code: str) -> Dict[str, Any]:
        """获取技术指标"""
        hist = self.get_stock_history(code, days=60)
        if hist.empty or len(hist) < 20:
            return {}
        close = hist['close'].astype(float).values
        ma5 = sum(close[-5:]) / 5 if len(close) >= 5 else 0
        ma10 = sum(close[-10:]) / 10 if len(close) >= 10 else 0
        ma20 = sum(close[-20:]) / 20 if len(close) >= 20 else 0
        current_price = close[-1] if len(close) > 0 else 0
        trend = "上涨" if ma5 > ma10 > ma20 else "下跌" if ma5 < ma10 < ma20 else "震荡"
        return {
            'current_price': current_price, 'ma5': round(ma5, 2), 'ma10': round(ma10, 2),
            'ma20': round(ma20, 2), 'trend': trend,
            'is_above_ma5': current_price > ma5, 'is_above_ma20': current_price > ma20,
        }

    def get_index_quote(self, code: str) -> Dict[str, Any]:
        """获取指数行情"""
        return self.get_stock_price(code)

    def get_batch_prices(self, codes: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量获取股票行情（新浪支持批量查询，高效）"""
        sina_codes = [self._code_to_sina(c) for c in codes]
        result = {}
        batch_size = 200
        for i in range(0, len(sina_codes), batch_size):
            batch = sina_codes[i:i + batch_size]
            try:
                resp = self._session.get(
                    f'http://hq.sinajs.cn/list={",".join(batch)}',
                    timeout=10
                )
                resp.encoding = 'gbk'
                data = self._parse_sina_quote(resp.text)
                result.update(data)
            except Exception as e:
                print(f"批量行情失败: {e}")
        return result

    def _get_realtime_quotes_sina(self) -> Dict[str, Dict]:
        """获取全市场实时行情（多源容错）

        优先级:
        1. 新浪Market Center全市场数据（可能触发456限制）
        2. hq.sinajs.cn批量获取（稳定，需预知代码列表）
        3. 本地缓存
        """
        if self._realtime_cache and self._cache_time:
            if (datetime.now() - self._cache_time).total_seconds() < 300:
                return self._realtime_cache

        all_data = {}

        # 方案1: 新浪Market Center（有频率限制，增加延迟）
        try:
            headers = {
                'Referer': 'https://finance.sina.com.cn',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            for page in range(1, 70):
                url = (
                    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                    "Market_Center.getHQNodeData?page={}&num=80&sort=symbol&asc=1&node=hs_a"
                    "&symbol=&_s_r_a=auto".format(page)
                )
                resp = self._session.get(url, headers=headers, timeout=10)
                if resp.status_code == 456:
                    # IP被限流，切换到备用方案
                    print("新浪Market Center限流(456)，切换到hq批量接口")
                    break
                if not resp.text.strip() or resp.text == '[]':
                    break
                items = json.loads(resp.text)
                if not items:
                    break
                for item in items:
                    code = item.get('code', '')
                    price = self._safe_float(item.get('trade', 0))
                    if code and price > 0:
                        all_data[code] = {
                            'name': item.get('name', ''),
                            'price': price,
                            'change_pct': self._safe_float(item.get('changepercent', 0)),
                            'prev_close': self._safe_float(item.get('settlement', 0)),
                            'open': self._safe_float(item.get('open', 0)),
                            'high': self._safe_float(item.get('high', 0)),
                            'low': self._safe_float(item.get('low', 0)),
                            'volume': self._safe_float(item.get('volume', 0)),
                            'amount': self._safe_float(item.get('amount', 0)),
                            'pe_ratio': self._safe_float(item.get('per', 0)),
                            'pb_ratio': self._safe_float(item.get('pb', 0)),
                            'market_cap': self._safe_float(item.get('mktcap', 0)),
                            'turnover_ratio': self._safe_float(item.get('turnoverratio', 0)),
                        }
                if len(items) < 80:
                    break
                import time
                time.sleep(0.5)  # 避免触发限流

            if len(all_data) > 100:
                # 缓存成功获取的数据到本地文件
                self._save_stock_codes_cache(all_data)
                self._realtime_cache = all_data
                self._cache_time = datetime.now()
                return all_data

        except Exception as e:
            print(f"新浪Market Center获取失败: {e}")

        # 方案2: hq.sinajs.cn批量获取（使用缓存的代码列表）
        if not all_data:
            all_data = self._fetch_quotes_by_code_list()

        if all_data:
            self._realtime_cache = all_data
            self._cache_time = datetime.now()
            return all_data

        # 方案3: 返回本地缓存文件
        return self._load_stock_codes_cache()

    def _fetch_quotes_by_code_list(self) -> Dict[str, Dict]:
        """通过hq.sinajs.cn批量获取行情（使用已知代码列表）"""
        all_data = {}

        # 优先使用上次缓存的代码列表
        code_list = self._load_code_list_from_cache()

        if not code_list:
            # 备用：本地预设代码文件
            code_list = self._load_preset_code_list()

        if not code_list:
            return {}

        # 转换为新浪代码格式并分批请求
        sina_codes = []
        for code in code_list:
            if code.startswith('6') or code.startswith('9'):
                sina_codes.append(f'sh{code}')
            else:
                sina_codes.append(f'sz{code}')

        batch_size = 50
        for i in range(0, len(sina_codes), batch_size):
            batch = sina_codes[i:i + batch_size]
            try:
                resp = self._session.get(
                    f'http://hq.sinajs.cn/list={",".join(batch)}',
                    headers={'Referer': 'https://finance.sina.com.cn'},
                    timeout=10
                )
                resp.encoding = 'gbk'
                if resp.status_code == 200:
                    parsed = self._parse_sina_quote(resp.text)
                    # 提取纯数字代码作为key
                    for code, q in parsed.items():
                        pure_code = code[2:] if code.startswith(('sh', 'sz')) else code
                        all_data[pure_code] = q
            except Exception:
                continue

        return all_data

    def _load_preset_code_list(self) -> List[str]:
        """加载预设的股票代码列表"""
        import os
        preset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stock_codes.json')
        if os.path.exists(preset_path):
            try:
                with open(preset_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_stock_codes_cache(self, data: Dict[str, Dict]):
        """将行情数据中的代码列表保存到本地缓存"""
        import os
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(cache_dir, exist_ok=True)
        try:
            code_list = list(data.keys())
            path = os.path.join(cache_dir, 'stock_codes_cache.json')
            with open(path, 'w') as f:
                json.dump(code_list, f)
            # 同时保存完整行情数据
            path2 = os.path.join(cache_dir, 'quotes_cache.json')
            with open(path2, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_stock_codes_cache(self) -> Dict[str, Dict]:
        """从本地缓存文件加载行情数据"""
        import os
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        path = os.path.join(cache_dir, 'quotes_cache.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                if data:
                    self._realtime_cache = data
                    self._cache_time = datetime.now()
                    return data
            except Exception:
                pass
        return {}

    def _load_code_list_from_cache(self) -> List[str]:
        """从本地缓存文件加载代码列表"""
        import os
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        path = os.path.join(cache_dir, 'stock_codes_cache.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    # 行业板块到新浪板块名称的映射
    SINA_SECTOR_MAP = {
        '煤炭': ['煤炭开采', '煤炭采选', '煤炭'],
        '商业航天': ['航天军工', '航天', '军工'],
        '科技芯片': ['半导体', '集成电路', '电子'],
        '新能源': ['新能源', '光伏', '锂电池'],
        '消费白酒': ['白酒', '食品饮料'],
        '金融券商': ['证券', '银行', '金融'],
        '半导体': ['半导体', '集成电路'],
        '集成电路': ['集成电路', '半导体'],
        '国产芯片': ['半导体', '集成电路', '国产芯片'],
        '锂电池': ['锂电池', '新能源'],
        '光伏': ['光伏', '新能源'],
        '储能': ['储能', '新能源'],
        '新能源车': ['新能源车', '汽车'],
        '军工': ['军工', '航天军工'],
        '航天': ['航天', '航天军工'],
        '证券': ['证券', '金融'],
        '券商': ['证券', '券商'],
        '银行': ['银行', '金融'],
    }

    # 板块关键词匹配（用于从全市场数据中筛选）
    SECTOR_KEYWORDS = {
        '煤炭': ['煤', '神华', '焦炭', '煤业'],
        '商业航天': ['航天', '卫星', '火箭', '导航', '宇航', '航空动力', '航空电子', '北方导航'],
        '科技芯片': ['芯片', '半导体', '集成电路', '微电子', '光刻', '电科芯片', '华微'],
        '新能源': ['锂电', '光伏', '储能', '新能源', '电池'],
        '消费白酒': ['茅台', '五粮液', '白酒', '酒'],
        '金融券商': ['证券', '券商', '银行', '金融', '信托', '保险'],
    }

    def get_hot_concepts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取热门概念板块（新浪接口）"""
        try:
            # 新浪概念板块行情（按涨幅排序）
            headers = {
                'Referer': 'https://finance.sina.com.cn',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            concepts = []

            # 尝试新浪行业板块接口
            for node_type in ['new_fl']:
                url = (
                    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                    "Market_Center.getHQNodeData?page=1&num={}&sort=changepercent&asc=0"
                    "&node={}&symbol=&_s_r_a=auto".format(limit, node_type)
                )
                try:
                    resp = self._session.get(url, headers=headers, timeout=10)
                    if resp.text.strip() and resp.text != '[]':
                        items = json.loads(resp.text)
                        for item in items:
                            concepts.append({
                                'name': item.get('name', ''),
                                'code': item.get('code', ''),
                                'change_pct': self._safe_float(item.get('changepercent', 0)),
                            })
                        break
                except Exception:
                    continue

            if concepts:
                concepts.sort(key=lambda x: x.get('change_pct', 0), reverse=True)
                return concepts[:limit]

            # 备用：从全市场数据中按行业聚合
            quotes = self._get_realtime_quotes_sina()
            if quotes:
                sector_changes = {}
                for code, q in quotes.items():
                    name = q.get('name', '')
                    for sector, kws in self.SECTOR_KEYWORDS.items():
                        if any(kw in name for kw in kws):
                            if sector not in sector_changes:
                                sector_changes[sector] = []
                            sector_changes[sector].append(q.get('change_pct', 0))
                            break
                for sector, changes in sector_changes.items():
                    avg_chg = sum(changes) / len(changes) if changes else 0
                    concepts.append({
                        'name': sector,
                        'code': '',
                        'change_pct': round(avg_chg, 2),
                    })
                concepts.sort(key=lambda x: x.get('change_pct', 0), reverse=True)
                return concepts[:limit]

        except Exception as e:
            print(f"获取热门概念失败: {e}")

        return []

    def get_sector_stocks(self, sector_name: str) -> List[Dict[str, Any]]:
        """获取板块成分股（新浪接口+全市场关键词匹配）"""
        stocks = []
        seen_codes = set()

        # 1. 先尝试新浪板块接口
        headers = {
            'Referer': 'https://finance.sina.com.cn',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        sector_names = self.SINA_SECTOR_MAP.get(sector_name, [sector_name])
        for sn in sector_names:
            try:
                url = (
                    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                    "Market_Center.getHQNodeData?page=1&num=50&sort=changepercent&asc=0"
                    "&node=hangye_{}&symbol=&_s_r_a=auto".format(sn)
                )
                resp = self._session.get(url, headers=headers, timeout=10)
                if resp.text.strip() and resp.text != '[]':
                    items = json.loads(resp.text)
                    for item in items:
                        code = item.get('code', '')
                        if code and code not in seen_codes:
                            seen_codes.add(code)
                            stocks.append({
                                'code': code,
                                'name': item.get('name', ''),
                                'price': self._safe_float(item.get('trade', 0)),
                                'change_pct': self._safe_float(item.get('changepercent', 0)),
                            })
            except Exception:
                continue

        if stocks:
            return stocks

        # 2. 备用：从全市场数据中按关键词匹配
        quotes = self._get_realtime_quotes_sina()
        if quotes:
            kws = self.SECTOR_KEYWORDS.get(sector_name, [sector_name])
            for code, q in quotes.items():
                name = q.get('name', '')
                if any(kw in name for kw in kws):
                    if code not in seen_codes:
                        seen_codes.add(code)
                        stocks.append({
                            'code': code,
                            'name': name,
                            'price': q.get('price', 0),
                            'change_pct': q.get('change_pct', 0),
                        })

        return stocks

    def get_concept_stocks(self, concept_name: str) -> List[Dict[str, Any]]:
        """获取概念板块成分股（复用get_sector_stocks）"""
        return self.get_sector_stocks(concept_name)

    def get_money_flow(self, code: str) -> Dict[str, Any]:
        """获取个股资金流向（新浪接口）"""
        try:
            sina_code = self._code_to_sina(code)
            # 新浪资金流向
            resp = self._session.get(
                f'http://vip.stock.finance.sina.com.cn/quotes/go.php/vFinanceAnalyze/kind/mainindex/index.phtml?symbol={sina_code}',
                timeout=5
            )
            # 新浪资金流向API较复杂，尝试腾讯备用
        except Exception:
            pass

        try:
            # 腾讯资金流向
            tencent_code = self._code_to_tencent(code)
            resp = self._session.get(
                f'http://qt.gtimg.cn/q=ff_{tencent_code}',
                timeout=5
            )
            resp.encoding = 'gbk'
            match = re.search(r'v_ff_s[hz]\d+="(.+)"', resp.text)
            if match:
                fields = match.group(1).split('~')
                if len(fields) >= 10:
                    return {
                        'main_net_inflow': self._safe_float(fields[1]),
                        'main_net_inflow_pct': self._safe_float(fields[3]),
                        'big_order_inflow_pct': self._safe_float(fields[5]) if len(fields) > 5 else 0,
                        'retail_net_inflow': self._safe_float(fields[7]) if len(fields) > 7 else 0,
                        'retail_net_inflow_pct': self._safe_float(fields[9]) if len(fields) > 9 else 0,
                    }
        except Exception as e:
            print(f"获取资金流向失败 {code}: {e}")

        return {}

    def get_top_gainers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取涨幅榜"""
        quotes = self._get_realtime_quotes_sina()
        if not quotes:
            quotes_list = self.get_realtime_quotes(limit=500)
        else:
            quotes_list = [{'code': code, **q} for code, q in quotes.items()]

        sorted_quotes = sorted(quotes_list, key=lambda x: x.get('change_pct', 0), reverse=True)
        return sorted_quotes[:limit]

    def get_top_losers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取跌幅榜"""
        quotes = self._get_realtime_quotes_sina()
        if not quotes:
            quotes_list = self.get_realtime_quotes(limit=500)
        else:
            quotes_list = [{'code': code, **q} for code, q in quotes.items()]

        sorted_quotes = sorted(quotes_list, key=lambda x: x.get('change_pct', 0))
        return sorted_quotes[:limit]

    def get_stock_info(self, code: str) -> Optional[Dict[str, Any]]:
        """获取股票基本信息（腾讯接口）"""
        try:
            tencent_code = self._code_to_tencent(code)
            resp = self._session.get(
                f'http://qt.gtimg.cn/q={tencent_code}',
                timeout=5
            )
            resp.encoding = 'gbk'
            for line in resp.text.strip().split(';'):
                match = re.match(r'v_(s[hz]\d+)="(.+)"', line.strip())
                if not match:
                    continue
                fields = match.group(2).split('~')
                if len(fields) >= 45:
                    return {
                        'code': code,
                        'name': fields[1],
                        'price': self._safe_float(fields[3]),
                        'prev_close': self._safe_float(fields[4]),
                        'change_pct': self._safe_float(fields[32]),
                        'pe_ratio': self._safe_float(fields[39]),
                        'market_cap': self._safe_float(fields[45]),
                        'total_shares': self._safe_float(fields[44]) if len(fields) > 44 else 0,
                    }
        except Exception as e:
            print(f"获取股票信息失败 {code}: {e}")
        return None


class SinaTencentHKClient(MarketClientBase):
    """港股行情客户端（新浪/腾讯直连）"""
    market_type = MarketType.HK

    def __init__(self):
        super().__init__()
        self._session = requests.Session()
        self._session.headers.update({
            'Referer': 'http://finance.sina.com.cn',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })

    def _code_to_sina(self, code: str) -> str:
        code = str(code).strip().zfill(5)
        return f'hk{code}'

    def get_stock_price(self, code: str) -> Dict[str, Any]:
        sina_code = self._code_to_sina(code)
        try:
            resp = self._session.get(f'http://hq.sinajs.cn/list={sina_code}', timeout=5)
            resp.encoding = 'gbk'
            match = re.search(r'var hq_str_hk\d+="(.+)"', resp.text)
            if match:
                fields = match.group(1).split(',')
                if len(fields) >= 13:
                    price = self._safe_float(fields[6])
                    prev_close = self._safe_float(fields[3])
                    return {
                        'code': code, 'name': fields[1],
                        'open': self._safe_float(fields[2]),
                        'prev_close': prev_close,
                        'price': price,
                        'high': self._safe_float(fields[4]),
                        'low': self._safe_float(fields[5]),
                        'change': price - prev_close,
                        'change_pct': round((price - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0,
                        'volume': self._safe_float(fields[12]),
                        'market': 'HK', 'source': 'sina',
                        'timestamp': datetime.now().isoformat(),
                    }
        except Exception as e:
            print(f"港股行情失败 {code}: {e}")
        return {'code': code, 'price': 0, 'change_pct': 0}

    def get_realtime_quotes(self, limit: int = 100) -> List[Dict[str, Any]]:
        return []

    def get_stock_history(self, code: str, days: int = 30, adjust: str = "qfq") -> pd.DataFrame:
        def _fetch():
            tencent_code = f'r_hk{str(code).zfill(5)}'
            url = f'https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?param={tencent_code},day,,,{days},1'
            resp = self._session.get(url, timeout=10)
            data = resp.json()
            kline_data = data.get('data', {}).get(tencent_code, {}).get('day') or \
                         data.get('data', {}).get(tencent_code, {}).get('qfqday') or []
            if not kline_data:
                return pd.DataFrame()
            df = pd.DataFrame(kline_data, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.tail(days)
        return self._retry_request(_fetch)

    def get_market_overview(self) -> Dict[str, Any]:
        return {'market': 'HK', 'note': 'use get_stock_price for individual stocks'}

    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        return []

    def get_stock_technical_indicators(self, code: str) -> Dict[str, Any]:
        hist = self.get_stock_history(code, days=60)
        if hist.empty or len(hist) < 20:
            return {}
        close = hist['close'].astype(float)
        return {
            'ma5': float(close.rolling(5).mean().iloc[-1]),
            'ma20': float(close.rolling(20).mean().iloc[-1]),
        }

    def get_index_quote(self, code: str = "HSI") -> Dict[str, Any]:
        try:
            resp = self._session.get('http://hq.sinajs.cn/list=rt_hkHSI', timeout=5)
            resp.encoding = 'gbk'
            match = re.search(r'var hq_str_rt_hkHSI="(.+)"', resp.text)
            if match:
                fields = match.group(1).split(',')
                if len(fields) >= 7:
                    return {'name': '恒生指数', 'price': self._safe_float(fields[6]),
                            'change_pct': self._safe_float(fields[7])}
        except Exception:
            pass
        return {}

    @staticmethod
    def _safe_float(val) -> float:
        if val is None:
            return 0.0
        try:
            if isinstance(val, str):
                val = val.strip().replace(',', '')
                if val in ('-', '', 'N/A', '--'):
                    return 0.0
            return float(val)
        except (ValueError, TypeError):
            return 0.0


class SinaTencentUSClient(MarketClientBase):
    """美股行情客户端（新浪/腾讯直连）"""
    market_type = MarketType.US

    def __init__(self):
        super().__init__()
        self._session = requests.Session()
        self._session.headers.update({
            'Referer': 'http://finance.sina.com.cn',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })

    def _code_to_sina(self, code: str) -> str:
        code = str(code).strip().upper()
        # 新浪美股格式: gb_.aapl 或 gb_aapl
        return f'gb_{code.lower()}'

    def get_stock_price(self, code: str) -> Dict[str, Any]:
        sina_code = self._code_to_sina(code)
        try:
            resp = self._session.get(f'http://hq.sinajs.cn/list={sina_code}', timeout=5)
            resp.encoding = 'gbk'
            match = re.search(r'var hq_str_gb_\w+="(.+)"', resp.text)
            if match:
                fields = match.group(1).split(',')
                if len(fields) >= 10:
                    price = self._safe_float(fields[1])
                    prev_close = self._safe_float(fields[3])
                    change_pct = self._safe_float(fields[2])
                    # 新浪美股 fields[2] 可能是涨跌额而非百分比，需自行计算
                    if prev_close > 0 and abs(change_pct) > 100:
                        change_pct = round((price - prev_close) / prev_close * 100, 2)
                    return {
                        'code': code.upper(), 'name': fields[0],
                        'open': self._safe_float(fields[7]),
                        'prev_close': prev_close,
                        'price': price,
                        'high': self._safe_float(fields[8]),
                        'low': self._safe_float(fields[9]),
                        'change': price - prev_close,
                        'change_pct': change_pct,
                        'volume': self._safe_float(fields[10]) if len(fields) > 10 else 0,
                        'market': 'US', 'source': 'sina',
                        'timestamp': datetime.now().isoformat(),
                    }
        except Exception as e:
            print(f"美股行情失败 {code}: {e}")
        return {'code': code, 'price': 0, 'change_pct': 0}

    def get_realtime_quotes(self, limit: int = 100) -> List[Dict[str, Any]]:
        return []

    def get_stock_history(self, code: str, days: int = 30, adjust: str = "qfq") -> pd.DataFrame:
        def _fetch():
            sina_code = self._code_to_sina(code)
            url = f'https://finance.sina.com.cn/stock/usstock/api/jsonp.php/IO.XSRV2.CallbackList/xru6t5/US_MinKline.getUSMinKline?symbol={sina_code}&scale=240&datalen={days}'
            resp = self._session.get(url, timeout=10)
            text = resp.text
            # 解析 JSONP
            json_match = re.search(r'\((.+)\)', text, re.DOTALL)
            if not json_match:
                return pd.DataFrame()
            data = json.loads(json_match.group(1))
            klines = data.get('data', [])
            if not klines:
                return pd.DataFrame()
            df = pd.DataFrame(klines)
            if 'd' in df.columns:
                df = df.rename(columns={'d': 'date', 'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 'v': 'volume'})
            for col in ['open', 'close', 'high', 'low', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.tail(days)
        return self._retry_request(_fetch)

    def get_market_overview(self) -> Dict[str, Any]:
        return {'market': 'US', 'note': 'use get_stock_price for individual stocks'}

    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        return []

    def get_stock_technical_indicators(self, code: str) -> Dict[str, Any]:
        hist = self.get_stock_history(code, days=60)
        if hist.empty or len(hist) < 20:
            return {}
        close = hist['close'].astype(float)
        return {
            'ma5': float(close.rolling(5).mean().iloc[-1]),
            'ma20': float(close.rolling(20).mean().iloc[-1]),
        }

    def get_index_quote(self, code: str = ".INX") -> Dict[str, Any]:
        try:
            # 新浪美股指数
            resp = self._session.get('http://hq.sinajs.cn/list=gb_dji,gb_ixic,gb_inx', timeout=5)
            resp.encoding = 'gbk'
            result = {}
            for line in resp.text.strip().split('\n'):
                match = re.search(r'var hq_str_gb_(\w+)="(.+)"', line)
                if match:
                    fields = match.group(2).split(',')
                    if len(fields) >= 3:
                        name = {'dji': '道琼斯', 'ixic': '纳斯达克', 'inx': '标普500'}.get(match.group(1), match.group(1))
                        result[name] = {'price': self._safe_float(fields[1]), 'change_pct': self._safe_float(fields[2])}
            return result
        except Exception:
            return {}

    @staticmethod
    def _safe_float(val) -> float:
        if val is None:
            return 0.0
        try:
            if isinstance(val, str):
                val = val.strip().replace(',', '')
                if val in ('-', '', 'N/A', '--'):
                    return 0.0
            return float(val)
        except (ValueError, TypeError):
            return 0.0