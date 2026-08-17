"""
股市博主量化分析引擎 - 实用版
目标：让用户能直接抄作业、做复盘
改进版：加入语义理解、时间衰减、信号优先级、基于复盘的评分
"""
import re
import json
import os
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict

from .llm_extractor import LLMExtractor, ExtractionResult
from ..data.stock_sector_map import STOCK_SECTOR_MAP, get_stock_sector


@dataclass
class StockSignal:
    """股票信号记录"""
    stock_name: str
    signal_type: str  # 'buy', 'sell', 'hold', 'watch', 'no_position'
    confidence: float  # 置信度 0-1
    publish_time: datetime
    content: str  # 原文片段
    negation_detected: bool  # 是否检测到否定词


@dataclass
class AnalysisResult:
    """分析结果"""
    blogger_name: str
    analysis_time: datetime

    # 基础数据
    current_positions: List[Dict] = field(default_factory=list)
    watch_positions: List[Dict] = field(default_factory=list)  # 新增：关注但未买入
    closed_positions: List[Dict] = field(default_factory=list)  # 新增：已清仓
    likely_positions: List[Dict] = field(default_factory=list)  # 看好/可能持仓
    bearish_positions: List[Dict] = field(default_factory=list)  # 看空/已清仓
    mentioned_positions: List[Dict] = field(default_factory=list)  # 仅讨论/提及  # 新增：可能持仓个股（基于微博推断）
    industry_distribution: Dict[str, int] = field(default_factory=dict)
    sector_distribution: Dict[str, int] = field(default_factory=dict)  # 新增：板块分布

    # 新增：仓位配置（核心改进）
    position_allocation: Dict[str, int] = field(default_factory=dict)  # 板块仓位比例 {"煤炭": 35, "商业航天": 50}
    attack_sector: str = ""  # 进攻板块
    defense_sector: str = ""  # 防守板块
    no_position_sectors: List[str] = field(default_factory=list)  # 明确不持有的板块

    # 新增：实用分析结果
    market_type: str = ""  # A股/港股/美股
    position_style: str = ""  # 激进/稳健/保守
    hot_industries: List[str] = field(default_factory=list)  # 重仓板块
    avoid_industries: List[str] = field(default_factory=list)  # 不碰板块

    # 选股体系
    selection_factors: Dict[str, int] = field(default_factory=dict)
    selection_formula: str = ""  # 选股口诀
    kline_patterns: List[str] = field(default_factory=list)
    avoid_stocks: List[str] = field(default_factory=list)

    # 交易体系
    trading_cycle: str = ""
    trading_cycle_detail: str = ""  # 具体持有天数
    stop_profit: str = ""  # 止盈点位
    stop_loss: str = ""  # 止损点位
    buy_sell_flow: str = ""  # 完整买卖流程

    # 交易心态
    discipline_score: int = 0
    common_mistakes: List[str] = field(default_factory=list)  # 最常犯的3个错误

    # 原始分析文本
    analysis_text: str = ""


class StockAnalyzer:
    """股市博主分析器 - 改进版，支持语义理解和时间衰减"""

    # ==================== 问题1修复：否定词和上下文感知 ====================
    # 否定词列表（用于检测"不看好"、"没买"等否定语义）
    NEGATION_WORDS = [
        '不', '没', '无', '未', '别', '莫', '非', '无', '并非', '不是',
        '没有', '不再', '不会', '不要', '不可', '无法', '难以'
    ]

    # 否定短语模式（这些短语本身表示否定，优先匹配）
    NEGATION_PHRASES = [
        '不看好', '不持有', '不碰', '不参与', '不建议', '不推荐',
        '没买', '没有买', '没持有', '没有持有',
        '已清仓', '已卖出', '已离场', '已止损',
        '清仓了', '卖出了', '离场了',
        '卖飞了', '踏空了', '错过了',
        '我不会买', '我不买', '我不看好',
        '不是股东', '不是我家股东',
    ]

    # 买入关键词（带权重）
    BUY_KEYWORDS = {
        # 高置信度（明确买入行为）
        '买了': 3, '买入': 3, '建仓': 3, '加仓': 3, '补仓': 3,
        '上车': 3, '入手': 3, '已买': 3, '低吸': 2, '抄底': 2,
        # 中置信度（买入意向）
        '潜伏': 2, '进场': 2, '埋伏': 2,
        # 低置信度（关注/看好，非明确买入）
        '关注': 1, '看好': 1, '推荐': 1, '目标': 1,
    }

    # 卖出关键词（带权重）
    SELL_KEYWORDS = {
        # 高置信度（明确卖出行为）
        '卖出': 3, '清仓': 3, '止盈': 3, '止损': 3,
        '离场': 3, '下车': 3, '卖了': 3, '已卖': 3,
        # 中置信度
        '减仓': 2, '高抛': 2, '兑现': 2, '落袋': 2,
        # 特殊短语
        '不是股东': 3, '不是我家股东': 3, '卖飞': 2,
    }

    # 持有关键词
    HOLD_KEYWORDS = {
        '持有': 3, '持仓': 3, '锁仓': 3, '继续持有': 3,
        '躺平': 2, '持有中': 3, '我有': 2,
    }

    # 不持有/已清仓关键词
    NO_POSITION_KEYWORDS = {
        '我没有': 3, '没买': 3, '不持有': 3, '已清': 3,
        '空仓': 3, '不碰': 2, '不参与': 2, '离场了': 3,
    }

    # 仓位比例识别模式
    POSITION_RATIO_PATTERNS = [
        r'(\w+)[大概约]?(\d+)[%％]的?仓位',
        r'(\w+)[大概约]?(\d+)[%％]',
        r'(\w+)占[比]?(\d+)[%％]',
        r'(\d+)[%％].*?(\w+)',
        r'仓位[大概约]?(\d+)[%％]',
    ]

    # 行业关键词
    INDUSTRY_KEYWORDS = {
        '煤炭': ['煤炭', '煤化工', '动力煤', '焦煤', '焦炭'],
        '商业航天': ['商业航天', '航天', '卫星', '火箭', '太空', '发射'],
        '锂电': ['锂电', '锂矿', '锂电池', '锂盐', '碳酸锂', '储能'],
        '化工': ['化工', '化纤', '聚氨酯', 'MDI', 'TDI'],
        '科技芯片': ['科技', '芯片', '半导体', '电子', '软件', '计算机', 'AI', '人工智能'],
        '新能源': ['新能源', '光伏', '风电', '充电桩', '电池', '硅料'],
        '医药医疗': ['医药', '生物', '医疗', '中药', '创新药', '医疗器械', 'CXO', '疫苗'],
        '消费白酒': ['消费', '食品', '饮料', '白酒', '啤酒', '调味品', '家电'],
        '金融券商': ['银行', '证券', '保险', '券商', '信托', '基金'],
        '地产基建': ['地产', '房地产', '物业', '基建', '建筑'],
        '有色': ['有色', '铜', '铝', '黄金', '白银', '稀土'],
        '汽车': ['汽车', '整车', '零部件', '新能源车'],
    }

    # 板块到上级板块的映射（LLM提取的细分板块→粗分板块，含STOCK_SECTOR_MAP的细分行业）
    SECTOR_ROLLUP = {
        # 新能源
        '光伏': '新能源', '风电': '新能源', '锂电': '新能源', '储能': '新能源',
        '充电桩': '新能源', '氢能': '新能源', '电池': '新能源', '硅料': '新能源',
        '光伏组件': '新能源', '光伏电池': '新能源', '光伏胶膜': '新能源',
        '光伏玻璃': '新能源', '光伏支架': '新能源', '光伏设备': '新能源',
        '光伏逆变器': '新能源', '微型逆变器': '新能源', '硅片': '新能源',
        '锂电池': '新能源', '锂矿': '新能源', '储能电池': '新能源',
        '锂电设备': '新能源',
        # 半导体/芯片
        '芯片': '半导体', '集成电路': '半导体', '封测': '半导体', '光刻': '半导体',
        '刻蚀': '半导体', '存储': '半导体', 'hbm': '半导体', 'eda': '半导体',
        '半导体': '半导体', '半导体设备': '半导体', '半导体检测': '半导体',
        '半导体代工': '半导体', '芯片设计': '半导体', '存储芯片': '半导体',
        '模拟芯片': '半导体', '功率半导体': '半导体', '射频芯片': '半导体',
        'CIS': '半导体', 'GPU': '人工智能', 'CPU': '半导体', 'FPGA': '半导体',
        'IGBT': '半导体', 'SoC芯片': '半导体', '物联网芯片': '半导体',
        '通信芯片': '半导体', '电源管理芯片': '半导体', '驱动芯片': '半导体',
        'AI芯片': '人工智能', '光刻胶': '半导体', 'CMP抛光液': '半导体',
        'CMP材料': '半导体', '靶材': '半导体',
        # AI/算力
        'ai': '人工智能', '大模型': '人工智能', '算力': '人工智能',
        'gpu': '人工智能', '光模块': '人工智能', '服务器': '人工智能',
        '数据中心': '人工智能', 'cpo': '人工智能', 'AI服务器': '人工智能',
        '光器件': '人工智能', '光连接器': '人工智能',
        # 医药
        '创新药': '医药', '中药': '医药', '医疗器械': '医药', 'cxo': '医药',
        'CXO': '医药', '疫苗': '医药', '仿制药': '医药', '血制品': '医药',
        '医药': '医药', '医药流通': '医药', '医疗服务': '医药',
        '生物制品': '医药', '医疗IT': '医药',
        # 消费
        '白酒': '消费', '食品': '消费', '零售': '消费', '家电': '消费',
        '啤酒': '消费', '调味品': '消费', '乳制品': '消费', '化妆品': '消费',
        '医美': '消费', '免税': '消费', '小家电': '消费', '厨电': '消费',
        '扫地机器人': '消费', '投影仪': '消费', '超市': '消费',
        # 金融
        '银行': '金融', '券商': '金融', '保险': '金融', '金融IT': '金融',
        # 地产/基建
        '房地产': '地产', '物业': '地产', '建材': '地产链',
        '地产': '地产', '基建': '基建',
        # 军工/航天
        '航天': '军工', '航空': '军工', '商业航天': '军工', '卫星': '军工',
        '导弹': '军工', '兵器': '军工', '舰船': '军工',
        '军工': '军工', '军工信息化': '军工', '军工电子': '军工',
        '军工电容': '军工', '军工连接器': '军工', '军工通信': '军工',
        '北斗导航': '军工', '航空发动机': '军工', '高温合金': '军工',
        '精密铸造': '军工', '超导/钛合金': '军工', '石英纤维': '军工',
        '碳纤维': '军工', '红外': '军工', '无人机': '军工',
        # 汽车
        '新能源车': '汽车', '智能驾驶': '汽车', '整车': '汽车', '零部件': '汽车',
        '汽车零部件': '汽车', '汽车电子': '汽车', '汽车内饰': '汽车',
        '汽车制动': '汽车', '汽车玻璃': '汽车', '铝压铸': '汽车',
        '热管理': '汽车', '车路协同': '汽车', 'ETC/V2X': '汽车',
        # 互联网/传媒
        '电商': '互联网', '游戏': '互联网', '短视频': '互联网', '直播': '互联网',
        '互联网': '互联网', '广告': '传媒', '视频': '传媒',
        # 有色/矿产
        '铜': '有色', '铝': '有色', '锂': '有色', '稀土': '有色', '黄金': '有色',
        '白银': '有色', '钴': '有色', '镍': '有色',
        '黄金/铜': '有色', '铜加工': '有色', '钼/钴': '有色', '锌/锗': '有色',
        '铅锌': '有色', '锡': '有色', '钨': '有色', '镁': '有色',
        '镍钴': '有色',
        # 煤炭
        '煤炭': '煤炭', '焦煤': '煤炭', '焦炭': '煤炭', '动力煤': '煤炭',
        '煤化工': '煤炭',
        # 石油/化工
        '石油': '石油', '油气': '石油', '炼化': '石油',
        '化工': '化工', '新材料': '化工', '碳纤维': '化工', '有机硅': '化工',
        '氟化工': '化工', '氯碱': '化工', '磷化工': '化工', '精细化工': '化工',
        '钛白粉': '化工',
        # 钢铁
        '钢铁': '钢铁', '铁矿石': '钢铁', '特钢': '钢铁',
        '不锈钢': '钢铁', '钢管': '钢铁',
        # 电力
        '电力': '电力', '火电': '电力', '水电': '电力', '核电': '电力', '绿电': '电力',
        # 农业
        '农业': '农业', '种业': '农业', '化肥': '农业', '养殖': '农业', '饲料': '农业',
        '种业/饲料': '农业', '饲料/养殖': '农业',
        # 机器人
        '机器人': '机器人', '减速器': '机器人', '伺服': '机器人', '人形机器人': '机器人',
        '伺服电机': '机器人', '伺服系统': '机器人', '工业机器人': '机器人',
        '机器人本体': '机器人',
        # 电子
        'pcb': '电子', 'PCB': '电子', '面板': '电子', 'led': '电子', 'LED': '电子',
        '连接器': '电子', '覆铜板': '电子', '消费电子': '电子',
        '声学': '电子', '声学/VR': '电子', '天线': '电子',
        '手机': '电子', '玻璃盖板': '电子', '指纹芯片': '电子',
        'ODM': '电子',
        '激光器': '电子', '激光显示': '电子', '激光设备': '电子',
        # 计算机/软件
        '软件': '计算机', '信创': '计算机', '网络安全': '计算机', '云计算': '计算机',
        '办公软件': '计算机', '企业软件': '计算机', '建筑软件': '计算机',
        '酒店IT': '计算机',
        # 通信
        '光纤': '通信', '光缆': '通信', '5g': '通信', '基站': '通信',
        '光纤光缆': '通信', '通信设备': '通信', '基站天线': '通信',
        '射频器件': '通信', '滤波器': '通信', '卫星通信': '通信',
        # 交运
        '快递': '交运', '物流': '交运', '航运': '交运', '港口': '交运',
        '铁路': '交运', '航空': '交运',
        # 机械
        '机械': '机械', '工程机械': '机械', '机床': '机械', '工业母机': '机械',
        '液压': '机械', '高空作业平台': '机械', '叉车': '机械',
        '破碎锤': '机械', '注塑机': '机械', '缝纫机': '机械',
        # 纺织服装
        '纺织': '纺织服装', '纺织辅料': '纺织服装', '印染': '纺织服装',
        '袜业': '纺织服装', '鞋履': '纺织服装', '服装': '纺织服装',
        '运动服装': '纺织服装', '商用车监控': '汽车',
        # 环保
        '环保': '环保', '固废': '环保', '环卫': '环保',
        # 美股
        '美股半导体': '美股', '美股新能源车': '美股', '美股消费电子': '美股',
        '美股互联网': '美股', '美股软件': '美股', '美股电商': '美股',
        '美股半导体设备': '美股', '美股存储': '美股',
        # 隔膜/正极/负极/电解液
        '隔膜': '新能源', '正极': '新能源', '负极': '新能源',
        '电解液': '新能源', '电解液溶剂': '新能源', '电池回收': '新能源',
        # 其他
        '光学': '电子', '钛': '有色', '钛合金': '有色',
    }

    # K线形态关键词
    KLINE_PATTERNS = {
        '突破形态': ['突破', '新高', '放量突破', '突破前高'],
        '回调形态': ['回调', '回踩', '企稳', '支撑位', '均线支撑'],
        '底部形态': ['底部', '筑底', 'W底', '双底', '头肩底'],
        '顶部形态': ['顶部', '双顶', '头肩顶', 'M头'],
        '趋势形态': ['上升趋势', '下降趋势', '箱体震荡', '横盘'],
    }

    # 周期关键词
    CYCLE_KEYWORDS = {
        '超短线（1-3天）': ['隔日', '次日', '今天买', '明天卖', '打板', '连板', '涨停', '日内'],
        '短线（3-10天）': ['短线', '一周内', '快进快出', '几天'],
        '波段（1-4周）': ['波段', '中线', '月线', '趋势', '持有待涨', '几周'],
        '中长线（1-6月）': ['中长线', '季度', '半年', '趋势持有'],
        '长线（6月以上）': ['长线', '价值投资', '长期持有', '年化', '复利'],
    }

    # 止盈止损关键词
    STOP_KEYWORDS = {
        '止盈': ['止盈', '落袋', '获利了结', '目标价', '预期收益'],
        '止损': ['止损', '破位', '跌破', '割肉', '认错'],
    }

    # 常见交易错误关键词
    MISTAKE_KEYWORDS = {
        '不止损死扛': ['死扛', '扛单', '不止损', '亏损加仓', '越跌越买'],
        '拿不住盈利': ['拿不住', '卖飞', '早卖', '卖早了', '踏空'],
        '追涨杀跌': ['追涨杀跌', '追高', '高位接盘', '杀跌'],
        '频繁交易': ['频繁', '乱操作', '手痒', '管不住手'],
        '情绪化操作': ['情绪', '冲动', '后悔', '贪心', '恐惧'],
        '满仓梭哈': ['满仓', '梭哈', '全仓', '一把'],
    }

    # 市场类型关键词
    MARKET_KEYWORDS = {
        'A股': ['A股', '主板', '创业板', '科创板', '北交所', '沪深'],
        '港股': ['港股', '港服', '恒生', '港股通'],
        '美股': ['美股', '纳指', '道指', '标普'],
    }

    # 美股到A股概念股/供应链企业映射（类级别属性）
    US_TO_A_STOCK_MAPPING = {
        'NVIDIA': {
            'name_cn': '英伟达',
            'market': '美股（纳斯达克）',
            'business': 'GPU芯片、AI计算、数据中心',
            'a_stock_concepts': [
                {'code': '300308', 'name': '中际旭创', 'relation': '光模块龙头，为NVIDIA提供800G/400G光模块'},
                {'code': '300394', 'name': '天孚通信', 'relation': '光器件核心供应商，配套NVIDIA光模块'},
                {'code': '300502', 'name': '新易盛', 'relation': '光模块供应商，LPO/CPO技术'},
                {'code': '601138', 'name': '工业富联', 'relation': 'AI服务器代工，为NVIDIA代工H100/H800'},
                {'code': '002475', 'name': '立讯精密', 'relation': '连接器供应商，配套AI服务器'},
                {'code': '300661', 'name': '圣邦股份', 'relation': '模拟芯片，电源管理配套'},
            ],
            'a_stock_sector': '光模块/AI算力',
        },
        'ASML': {
            'name_cn': '阿斯麦',
            'market': '美股/荷兰（纳斯达克）',
            'business': '光刻机设备，芯片制造核心设备',
            'a_stock_concepts': [
                {'code': '002371', 'name': '北方华创', 'relation': '国产刻蚀/薄膜设备龙头，对标ASML生态'},
                {'code': '688012', 'name': '中微公司', 'relation': '刻蚀设备龙头，国产替代'},
                {'code': '002222', 'name': '福晶科技', 'relation': '光学晶体，曾传为ASML供货'},
                {'code': '688256', 'name': '芯源微', 'relation': '涂胶显影设备'},
                {'code': '688037', 'name': '芯碁微装', 'relation': '直写光刻设备'},
                {'code': '688120', 'name': '华海清科', 'relation': 'CMP抛光设备'},
            ],
            'a_stock_sector': '半导体设备',
        },
        'AMD': {
            'name_cn': '超威半导体',
            'market': '美股（纳斯达克）',
            'business': 'CPU、GPU芯片，AI加速器',
            'a_stock_concepts': [
                {'code': '688041', 'name': '海光信息', 'relation': '国产CPU，AMD技术授权背景'},
                {'code': '300308', 'name': '中际旭创', 'relation': '光模块，配套AMD AI芯片'},
                {'code': '002049', 'name': '紫光国微', 'relation': '国产芯片，FPGA/安全芯片'},
                {'code': '688981', 'name': '中芯国际', 'relation': '晶圆代工，潜在代工AMD产品'},
            ],
            'a_stock_sector': '国产芯片',
        },
        'Intel': {
            'name_cn': '英特尔',
            'market': '美股（纳斯达克）',
            'business': 'CPU芯片、半导体制造',
            'a_stock_concepts': [
                {'code': '688981', 'name': '中芯国际', 'relation': '国产晶圆代工龙头'},
                {'code': '002371', 'name': '北方华创', 'relation': '半导体设备，国产替代'},
                {'code': '603501', 'name': '韦尔股份', 'relation': '芯片设计，CIS传感器'},
                {'code': '688008', 'name': '澜起科技', 'relation': '内存接口芯片，Intel生态'},
            ],
            'a_stock_sector': '半导体/国产替代',
        },
        'Micron': {
            'name_cn': '美光科技',
            'market': '美股（纳斯达克）',
            'business': '内存芯片（DRAM、NAND闪存）',
            'a_stock_concepts': [
                {'code': '688981', 'name': '中芯国际', 'relation': '国产晶圆代工'},
                {'code': '002371', 'name': '北方华创', 'relation': '存储设备供应商'},
                {'code': '688047', 'name': '龙芯中科', 'relation': '国产CPU'},
                {'code': '688008', 'name': '澜起科技', 'relation': '内存接口芯片'},
                {'code': '300661', 'name': '圣邦股份', 'relation': '模拟芯片，存储配套'},
            ],
            'a_stock_sector': '存储芯片',
        },
        'Corning': {
            'name_cn': '康宁公司',
            'market': '美股（纽交所）',
            'business': '特种玻璃、光纤、显示面板玻璃',
            'a_stock_concepts': [
                {'code': '601869', 'name': '长飞光纤', 'relation': '光纤光缆龙头'},
                {'code': '600487', 'name': '亨通光电', 'relation': '光纤光缆'},
                {'code': '600522', 'name': '中天科技', 'relation': '光纤光缆'},
                {'code': '002281', 'name': '光迅科技', 'relation': '光器件/光模块'},
                {'code': '300308', 'name': '中际旭创', 'relation': '光模块，使用光纤器件'},
            ],
            'a_stock_sector': '光纤光缆',
        },
        'Tesla': {
            'name_cn': '特斯拉',
            'market': '美股（纳斯达克）',
            'business': '电动汽车、自动驾驶、储能',
            'a_stock_concepts': [
                {'code': '300750', 'name': '宁德时代', 'relation': '动力电池龙头，特斯拉供应商'},
                {'code': '002594', 'name': '比亚迪', 'relation': '新能源车龙头'},
                {'code': '002920', 'name': '德赛西威', 'relation': '智能驾驶域控制器'},
                {'code': '300124', 'name': '汇川技术', 'relation': '电机控制器'},
                {'code': '603378', 'name': '旭升集团', 'relation': '特斯拉铝压铸件供应商'},
                {'code': '002050', 'name': '三花智控', 'relation': '热管理系统供应商'},
            ],
            'a_stock_sector': '新能源车/智能驾驶',
        },
    }

    def __init__(self, enable_llm: bool = True):
        self.stock_mentions: Dict[str, Counter] = {}
        self.all_text: str = ""
        # 实例属性引用类级别映射，方便方法调用
        self.us_to_a_stock_mapping = self.US_TO_A_STOCK_MAPPING
        # 时间衰减参数
        self.time_decay_days = 30  # 30天前的观点权重衰减到50%
        self.recent_weight = 1.0   # 最近观点权重
        self.old_weight = 0.15     # 旧观点权重（降低，更突出近期信号）
        self.time_decay_half_life = 7  # 指数衰减半衰期（天）：7天后权重衰减到50%

        # 时序聚合参数
        self.signal_density_bonus = 1.3   # 同一窗口内多次出现的信号额外加成
        self.signal_consistency_bonus = 1.2  # 近期信号立场一致的额外加成
        self.temporal_windows = [1, 3, 7, 14, 30]  # 时间窗口（天）：1天/3天/1周/2周/1月

        # LLM实体提取器
        self._llm_extractor = LLMExtractor()
        llm_available = self._llm_extractor.available

        # ==================== P0修复：LLM语义理解 ====================
        self.enable_llm = enable_llm and (os.environ.get('ENABLE_LLM_ANALYSIS', 'true').lower() == 'true' or llm_available)
        self._llm_analyzer = None  # 延迟加载，避免启动时加载失败
        self._llm_analyzer_failed = False
        self._llm_cache: Dict[str, Tuple[str, float, bool, str]] = {}  # LLM结果缓存

        # ==================== P1优化：增强语义分析 ====================
        self._enhanced_analyzer = None  # 延迟加载EnhancedSemanticAnalyzer
        self._use_enhanced_semantic = False  # 已废弃

    # ==================== 问题1修复：语义感知的信号识别 ====================
    def _detect_negation(self, content: str, keyword: str) -> Tuple[bool, str]:
        """
        检测关键词是否被否定

        Args:
            content: 微博内容
            keyword: 关键词

        Returns:
            (is_negated, reason): 是否被否定，否定原因
        """
        # 1. 先检查是否包含否定短语（优先匹配）
        for neg_phrase in self.NEGATION_PHRASES:
            if neg_phrase in content:
                # 检查否定短语和关键词是否在同一句话中
                # 找到关键词位置
                kw_pos = content.find(keyword)
                neg_pos = content.find(neg_phrase)
                if kw_pos != -1 and neg_pos != -1:
                    # 检查是否在同一句（以句号、感叹号、问号分隔）
                    sentence_start = max(content.rfind('。', 0, kw_pos), content.rfind('！', 0, kw_pos), content.rfind('？', 0, kw_pos), -1)
                    sentence_end = min(
                        content.find('。', kw_pos) if content.find('。', kw_pos) != -1 else len(content),
                        content.find('！', kw_pos) if content.find('！', kw_pos) != -1 else len(content),
                        content.find('？', kw_pos) if content.find('？', kw_pos) != -1 else len(content)
                    )
                    if sentence_start < neg_pos < sentence_end:
                        return True, f"检测到否定短语: {neg_phrase}"

        # 2. 检查关键词前是否有否定词
        kw_pos = content.find(keyword)
        if kw_pos == -1:
            return False, ""

        # 检查关键词前5个字符内是否有否定词
        check_start = max(0, kw_pos - 10)
        prefix = content[check_start:kw_pos]

        for neg_word in self.NEGATION_WORDS:
            if neg_word in prefix:
                # 检查否定词和关键词之间是否有其他词（避免误判）
                neg_pos = prefix.rfind(neg_word)
                between = prefix[neg_pos + len(neg_word):]
                # 如果中间有其他实质性词汇，可能不是否定
                if len(between.strip()) <= 2:  # 中间只有1-2个字符（如"的"、"了"）
                    return True, f"检测到否定词: {neg_word}"

        return False, ""

    # ==================== P0修复：LLM语义理解增强 ====================
    def _get_llm_analyzer(self):
        """获取LLM分析器 — 静默降级，仅首次失败时打印"""
        if self._llm_analyzer is None and self.enable_llm and not self._llm_analyzer_failed:
            try:
                from .llm_semantic import HybridAnalyzer
                self._llm_analyzer = HybridAnalyzer()
            except Exception:
                self._llm_analyzer_failed = True
                self._llm_analyzer = None
        return self._llm_analyzer

    # ==================== P1优化：增强语义分析 ====================
    def _get_enhanced_analyzer(self):
        """获取增强版语义分析器 — 已废弃，统一返回None"""
        return None

    def _analyze_with_enhanced_semantic(self, content: str, stock_name: str,
                                         blogger_id: str = None) -> Tuple[str, float, bool, str]:
        """使用增强版语义分析器进行深度分析

        Args:
            content: 微博内容
            stock_name: 股票名称
            blogger_id: 博主ID（用于上下文管理）

        Returns:
            (signal_type, confidence, is_negated, reason): 分析结果
        """
        cache_key = f"enhanced:{content[:50]}:{stock_name}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        analyzer = self._get_enhanced_analyzer()
        if analyzer is None:
            return self._analyze_with_llm(content, stock_name)

        try:
            result = analyzer.analyze(content, stock_name, blogger_id)

            signal_type = result.action_type
            confidence = result.action_confidence
            reason = result.reasoning

            # 场景类型调整（与_analyze_with_llm保持一致）
            if result.scenario_type == "simulation":
                confidence *= 0.5
                reason = f"[模拟盘] {reason}"
            elif result.scenario_type == "example":
                signal_type = "watch"
                confidence = 0.3
                reason = f"[举例说明] {reason}"
            elif result.scenario_type == "discussion":
                confidence *= 0.7
                reason = f"[市场讨论] {reason}"
            elif not result.is_formal_opinion:
                confidence *= 0.5
                reason = f"[闲聊] {reason}"
            elif result.true_sentiment == "sarcastic":
                if signal_type == "buy":
                    signal_type = "no_position"
                elif signal_type == "sell":
                    signal_type = "hold"
                reason = f"[反讽] {reason}"
            else:
                reason = f"[增强语义] {reason}"

            # 模糊匹配股票名称变体
            if result.extracted_info.stock_names:
                try:
                    from .enhanced_semantic import FuzzyMatcher
                    for raw_name in result.extracted_info.stock_names:
                        matched = FuzzyMatcher.fuzzy_match_stock(raw_name)
                        if matched and matched != raw_name:
                            reason += f" (模糊匹配: {raw_name}->{matched})"
                except Exception:
                    pass

            # 低质量分析日志
            if result.quality.overall_score < 0.5:
                print(f"低质量语义分析 (score={result.quality.overall_score:.2f}): {result.quality.issues}")

            signal_tuple = (signal_type, confidence, False, reason)
            self._llm_cache[cache_key] = signal_tuple
            return signal_tuple

        except Exception:
            return self._analyze_with_llm(content, stock_name)

    def _should_use_llm_analysis(self, content: str, keyword_result: Tuple) -> bool:
        """
        判断是否需要使用LLM进行深度分析

        Args:
            content: 微博内容
            keyword_result: 关键词分析结果

        Returns:
            bool: 是否需要LLM分析
        """
        signal_type, confidence, is_negated, reason = keyword_result

        # 1. 置信度低于阈值时使用LLM
        if confidence < 0.6:
            return True

        # 2. 检测到否定时使用LLM（可能是复杂语义）
        if is_negated:
            return True

        # 3. 内容包含需要深度理解的模式
        complex_patterns = [
            r'如果.*就',  # 假设性讨论
            r'假设',
            r'要是',
            r'早知道',  # 后悔情绪
            r'可惜',
            r'本来',
            r'其实',  # 转折
            r'开玩笑',  # 玩笑/反讽
            r'呵呵',  # 可能反讽
            r'笑死',
            r'我就看看',  # 围观心态
            r'吃瓜',
            r'模拟',  # 模拟盘
            r'虚拟',
            r'小号',
            r'只是',  # 强调非真实
            r'举例',  # 举例说明
            r'比如',  # 举例说明
            r'不是说',  # 否定/澄清
            r'别误会',  # 澄清
        ]

        for pattern in complex_patterns:
            if re.search(pattern, content):
                return True

        # 4. 同时存在买入和卖出关键词（冲突）
        has_buy = any(kw in content for kw in self.BUY_KEYWORDS.keys())
        has_sell = any(kw in content for kw in self.SELL_KEYWORDS.keys())
        if has_buy and has_sell:
            return True

        return False

    def _analyze_with_llm(self, content: str, stock_name: str) -> Tuple[str, float, bool, str]:
        """
        使用LLM进行深度语义分析

        Args:
            content: 微博内容
            stock_name: 股票名称

        Returns:
            (signal_type, confidence, is_negated, reason): 分析结果
        """
        # 检查缓存
        cache_key = f"{content[:50]}:{stock_name}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        llm_analyzer = self._get_llm_analyzer()
        if llm_analyzer is None:
            return ('watch', 0.5, False, "LLM分析器不可用")

        try:
            # 使用混合分析器
            result, method = llm_analyzer.analyze_with_fallback(content, stock_name)

            # 将LLM结果转换为信号格式
            signal_type = result.action_type
            confidence = result.action_confidence

            # 根据场景类型调整置信度
            if result.scenario_type == "simulation":
                # 模拟盘信号降低置信度
                confidence *= 0.5
                reason = f"[模拟盘] {result.reasoning}"
            elif result.scenario_type == "example":
                # 举例说明不作为真实信号
                signal_type = "watch"
                confidence = 0.3
                reason = f"[举例说明] {result.reasoning}"
            elif result.scenario_type == "discussion":
                # 市场讨论降低置信度
                confidence *= 0.7
                reason = f"[市场讨论] {result.reasoning}"
            elif not result.is_formal_opinion:
                # 非正式观点降低置信度
                confidence *= 0.5
                reason = f"[闲聊] {result.reasoning}"
            elif result.true_sentiment == "sarcastic":
                # 反讽内容反转信号
                if signal_type == "buy":
                    signal_type = "no_position"
                elif signal_type == "sell":
                    signal_type = "hold"
                reason = f"[反讽] {result.reasoning}"
            else:
                reason = f"[LLM分析] {result.reasoning}"

            # 缓存结果
            self._llm_cache[cache_key] = (signal_type, confidence, False, reason)

            return (signal_type, confidence, False, reason)

        except Exception as e:
            return ('watch', 0.5, False, f"LLM分析失败: {str(e)}")

    def _analyze_signal_with_context(self, content: str, stock_name: str) -> Tuple[str, float, bool, str]:
        """
        结合上下文分析股票信号

        Args:
            content: 微博内容
            stock_name: 股票名称

        Returns:
            (signal_type, confidence, is_negated, reason): 信号类型、置信度、是否被否定、原因
        """
        # 提取包含股票的句子
        sentences = re.split(r'[。！？\n]', content)
        stock_sentences = [s for s in sentences if stock_name in s]
        if not stock_sentences:
            stock_sentences = [content]  # 如果找不到，用全文

        relevant_text = ' '.join(stock_sentences)

        # 检测各种信号
        signals_found = []  # [(signal_type, weight, is_negated, reason)]

        # 检查卖出信号（优先级最高）
        for kw, weight in self.SELL_KEYWORDS.items():
            if kw in relevant_text:
                is_negated, reason = self._detect_negation(relevant_text, kw)
                if not is_negated:
                    signals_found.append(('sell', weight, False, f"关键词: {kw}"))
                else:
                    signals_found.append(('no_position', weight * 0.5, True, reason))

        # 检查不持有信号
        for kw, weight in self.NO_POSITION_KEYWORDS.items():
            if kw in relevant_text:
                is_negated, reason = self._detect_negation(relevant_text, kw)
                if not is_negated:
                    signals_found.append(('no_position', weight, False, f"关键词: {kw}"))

        # 检查持有信号
        for kw, weight in self.HOLD_KEYWORDS.items():
            if kw in relevant_text:
                is_negated, reason = self._detect_negation(relevant_text, kw)
                if not is_negated:
                    signals_found.append(('hold', weight, False, f"关键词: {kw}"))
                else:
                    signals_found.append(('no_position', weight * 0.5, True, reason))

        # 检查买入信号
        for kw, weight in self.BUY_KEYWORDS.items():
            if kw in relevant_text:
                is_negated, reason = self._detect_negation(relevant_text, kw)
                if not is_negated:
                    signals_found.append(('buy', weight, False, f"关键词: {kw}"))
                else:
                    # 被否定的买入 = 不持有
                    signals_found.append(('no_position', weight * 0.5, True, reason))

        if not signals_found:
            return ('watch', 0.5, False, "仅提及，无明确信号")

        # ==================== 问题3修复：信号优先级和冲突消解 ====================
        # 按优先级排序: sell > no_position > hold > buy > watch
        priority_order = {'sell': 5, 'no_position': 4, 'hold': 3, 'buy': 2, 'watch': 1}

        # 按优先级和权重排序
        signals_found.sort(key=lambda x: (priority_order.get(x[0], 0), x[1]), reverse=True)

        # 取最高优先级的信号
        best_signal = signals_found[0]

        # 归一化置信度
        confidence = min(best_signal[1] / 3.0, 1.0)

        keyword_result = (best_signal[0], confidence, best_signal[2], best_signal[3])

        # ==================== P0/P1修复：语义增强分析 ====================
        # 如果关键词匹配置信度低或存在复杂语义，使用语义分析进行深度分析
        if self.enable_llm and self._should_use_llm_analysis(content, keyword_result):
            if self._use_enhanced_semantic:
                llm_result = self._analyze_with_enhanced_semantic(content, stock_name)
            else:
                llm_result = self._analyze_with_llm(content, stock_name)
            # 如果LLM分析给出了更高置信度的结果，使用LLM结果
            if llm_result[1] > confidence * 0.8:
                return llm_result
            # 否则保留关键词结果，但记录LLM分析作为补充
            return keyword_result

        return keyword_result

    # ==================== 问题3修复：时间衰减机制 ====================
    def _calculate_time_weight(self, publish_time: datetime, now: datetime = None) -> float:
        """
        计算时间衰减权重 - 指数衰减，近期信号权重显著更高

        使用半衰期指数衰减: weight = recent_weight * 2^(-days_diff / half_life)
        - 0天(今天): 1.0
        - 7天: ~0.5
        - 14天: ~0.25
        - 30天: ~0.12
        - 60天: ~0.03

        Args:
            publish_time: 发布时间
            now: 当前时间

        Returns:
            权重值 (0.03 - 1.0)
        """
        if now is None:
            now = datetime.now()

        if not publish_time:
            return 0.3  # 无时间信息，低权重（避免无时间信号干扰近期判断）

        try:
            if isinstance(publish_time, str):
                publish_time = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
            days_diff = (now - publish_time).days
        except:
            return 0.3

        if days_diff <= 0:
            return self.recent_weight

        # 指数衰减: weight = recent_weight * 2^(-days / half_life)
        weight = self.recent_weight * (2 ** (-days_diff / self.time_decay_half_life))
        return max(weight, self.old_weight)

    def analyze_posts(self, posts: List[Dict], blogger_name: str = "博主") -> AnalysisResult:
        """分析微博内容"""
        result = AnalysisResult(
            blogger_name=blogger_name,
            analysis_time=datetime.now()
        )

        if not posts:
            result.analysis_text = "暂无微博数据，无法分析。请先爬取博主微博内容。"
            return result

        # 提取所有文本
        self.all_text = "\n".join([p.get('content', '') for p in posts if p.get('content')])

        # 0. 分析仓位配置（新增核心功能）
        self._analyze_position_allocation(posts, result)

        # 1. 分析持仓画像
        self._analyze_positions(posts, result)

        # 2. 分析选股体系
        self._analyze_selection(result)

        # 3. 分析交易体系
        self._analyze_trading_system(result)

        # 4. 分析交易心态（传入博主名称以获取复盘数据）
        self._analyze_psychology(result, blogger_name)

        # 5. 生成实用报告
        result.analysis_text = self._generate_practical_report(result)

        return result

    def _analyze_position_allocation(self, posts: List[Dict], result: AnalysisResult):
        """分析仓位配置 - 核心改进：识别博主自报的仓位比例"""
        position_allocation = {}  # {板块: 比例}
        no_position_sectors = []  # 明确不持有的板块
        attack_sector = ""
        defense_sector = ""

        # 【改进】优先搜索包含仓位关键词的微博
        position_keywords = ['仓位', '持仓', '%', '我有', '我没有', '加仓', '清仓', '攻击', '防守']
        position_posts = []
        for post in posts:
            content = post.get('content', '')
            if content and any(kw in content for kw in position_keywords):
                position_posts.append(post)

        # 如果找到包含仓位信息的微博，优先分析这些
        analyze_posts = position_posts if position_posts else posts
        print(f"分析仓位配置: 共{len(analyze_posts)}条相关微博")

        # 遍历微博，识别仓位配置
        for post in analyze_posts:
            content = post.get('content', '')
            if not content:
                continue

            # 1. 识别仓位比例（多种格式）
            for industry, keywords in self.INDUSTRY_KEYWORDS.items():
                for kw in keywords:
                    # 匹配各种格式 - 使用非捕获组(?:...)匹配"大概"或"约"
                    patterns = [
                        # "板块 + (大概/约) + 数字%"
                        rf'{kw}(?:大概|约)?(\d+)[%％]',
                        # "板块 + (大概/约) + 数字% + 仓位"
                        rf'{kw}(?:大概|约)?(\d+)[%％]\s*仓位',
                        # "板块 + 我有 + (大概/约) + 数字%" - 关键格式！
                        rf'{kw}我有[，,]?\s*(?:大概|约)?(\d+)[%％]',
                        # "板块 + 我有 + (大概/约) + 数字% + 仓位" - 关键格式！
                        rf'{kw}我有[，,]?\s*(?:大概|约)?(\d+)[%％]\s*仓位',
                        # "板块 + 占比 + 数字%"
                        rf'{kw}占[比]?(\d+)[%％]',
                        # "加了 + 板块 + (大概/约) + 数字%"
                        rf'加了{kw}(?:大概|约)?(\d+)[%％]',
                        # "加了 + 板块 + (大概/约) + 数字% + 仓位"
                        rf'加了{kw}(?:大概|约)?(\d+)[%％]\s*仓位',
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, content)
                        if match:
                            ratio = int(match.group(1))
                            if ratio > 0 and ratio <= 100:
                                position_allocation[industry] = ratio
                                print(f"识别到仓位: {industry} {ratio}%")
                                break

            # 2. 识别"不持有/已清仓"的板块 - 更精确的匹配
            for industry, keywords in self.INDUSTRY_KEYWORDS.items():
                for kw in keywords:
                    # 只匹配明确的否定模式
                    no_pos_patterns = [
                        # "板块 + 我没有"（板块在前）
                        rf'{kw}[，,。\s]+我没有',
                        # "我没有 + 板块"（板块在后）
                        rf'我没有[，,。\s]+{kw}',
                        # "板块 + 不碰"
                        rf'不碰.*{kw}',
                        # "板块 + 不参与"
                        rf'不参与.*{kw}',
                        # "板块 + 已清仓"
                        rf'{kw}[已清清]仓',
                        # "清仓 + 板块"
                        rf'清仓.*{kw}',
                    ]
                    for pattern in no_pos_patterns:
                        if re.search(pattern, content):
                            # 检查是否同时有仓位比例（如果有仓位，则不是不持有）
                            if industry not in position_allocation:
                                if industry not in no_position_sectors:
                                    no_position_sectors.append(industry)
                                    print(f"识别到不持有: {industry}")
                            break

            # 3. 识别"攻击/防守"板块配置
            if '攻击' in content and '商业航天' in content:
                attack_sector = '商业航天'
                print(f"识别到攻击板块: 商业航天")
            if '防守' in content and '煤炭' in content:
                defense_sector = '煤炭'
                print(f"识别到防守板块: 煤炭")
            # 更通用的匹配
            attack_match = re.search(r'攻击.*?板块?\s*(\S+)', content)
            if attack_match and not attack_sector:
                attack_sector = attack_match.group(1)
            defense_match = re.search(r'防守.*?板块?\s*(\S+)', content)
            if defense_match and not defense_sector:
                defense_sector = defense_match.group(1)

        # 保存结果
        result.position_allocation = position_allocation
        result.no_position_sectors = no_position_sectors
        result.attack_sector = attack_sector
        result.defense_sector = defense_sector

    # 卖出关键词 - 扩展版
    SELL_KEYWORDS_EXTENDED = [
        '卖出', '清仓', '止盈', '减仓', '离场', '止损', '割肉', '高抛',
        '不是股东', '不是你家股东', '卖了', '已卖', '卖掉了', '出掉了',
        '落袋', '兑现', '出货', '抛售', '调出', '卖点', '下车', '走了',
    ]

    # 买入关键词 - 扩展版
    BUY_KEYWORDS_EXTENDED = [
        '买入', '建仓', '低吸', '抄底', '加仓', '补仓', '进场',
        '上车', '入手', '买了', '已买', '潜伏',
    ]

    def _analyze_positions(self, posts: List[Dict], result: AnalysisResult):
        """分析持仓画像 - 改进版v7，时序聚合+指数衰减"""
        stock_counter = Counter()
        sector_counter = Counter()
        industry_counter = Counter()
        industry_signal_times = defaultdict(list)  # {industry: [datetime, ...]} 用于时序密度加成
        stock_signals = defaultdict(lambda: defaultdict(list))

        now = datetime.now()

        # ========== LLM提取优先 ==========
        llm_extraction = None
        if self.enable_llm:
            print("尝试LLM实体提取...")
            llm_extraction = self._llm_extractor.extract(posts, max_posts=30)
            if llm_extraction.method == "llm" and (llm_extraction.stocks or llm_extraction.sectors):
                print(f"LLM提取成功: {len(llm_extraction.stocks)}只股票, {len(llm_extraction.sectors)}个板块")
            else:
                print(f"LLM提取未获得结果(method={llm_extraction.method})，使用正则回退")

        if llm_extraction and llm_extraction.method == "llm" and (llm_extraction.stocks or llm_extraction.sectors):
            # ===== LLM结果路径 =====
            for s in llm_extraction.stocks:
                stock_counter[s.name] += 1
                sentiment_score = 0
                if s.sentiment == "bullish":
                    sentiment_score = 3 * s.confidence
                elif s.sentiment == "bearish":
                    sentiment_score = -2 * s.confidence
                # 有明确交易信号加权
                signal_bonus = 1.5 if s.signal in ("买入", "加仓", "建仓") else 1.0
                if s.signal in ("卖出", "清仓", "减仓"):
                    signal_bonus = 0.3
                final_score = sentiment_score * signal_bonus

                if final_score > 0:
                    stock_signals[s.name]['buy' if s.sentiment == 'bullish' else 'hold'].append(
                        (final_score, '', f"LLM: sentiment={s.sentiment}, signal={s.signal}, conf={s.confidence:.1f}")
                    )
                elif final_score < 0:
                    stock_signals[s.name]['sell'].append(
                        (abs(final_score), '', f"LLM: sentiment={s.sentiment}, signal={s.signal}")
                    )
                else:
                    stock_signals[s.name]['watch'].append(
                        (0.5, '', f"LLM: neutral")
                    )

            for sec in llm_extraction.sectors:
                rollup = self.SECTOR_ROLLUP.get(sec.name, sec.name)
                if sec.sentiment == "bullish":
                    industry_counter[rollup] += sec.confidence * 3
                elif sec.sentiment == "bearish":
                    industry_counter[rollup] -= sec.confidence * 1
                else:
                    industry_counter[rollup] += sec.confidence * 0.5
                industry_signal_times[rollup].append(now)

            # 股票→板块映射：LLM提取的股票自动归类到行业
            llm_sector_names = {self.SECTOR_ROLLUP.get(sec.name, sec.name) for sec in llm_extraction.sectors}
            for s in llm_extraction.stocks:
                stock_sector = get_stock_sector(s.name)
                if stock_sector:
                    rollup = self.SECTOR_ROLLUP.get(stock_sector, stock_sector)
                    # LLM已明确提取的板块不重复加
                    if rollup not in llm_sector_names:
                        if s.sentiment == "bullish":
                            industry_counter[rollup] += s.confidence * 2
                        elif s.sentiment == "bearish":
                            industry_counter[rollup] -= s.confidence * 0.5
                        else:
                            industry_counter[rollup] += s.confidence * 0.5
                        industry_signal_times[rollup].append(now)

            # 更新仓位配置（如果LLM提取到了sentiment信息）
            self._update_position_from_llm(llm_extraction, result)

        else:
            # ===== 正则回退路径（原有逻辑） =====
            image_stocks = {}
            ocr_instance = None
            ocr_available = False
            enable_ocr = os.environ.get('ENABLE_IMAGE_OCR', 'false').lower() == 'true'
            has_images = any(post.get('image_paths') for post in posts)

            if has_images and enable_ocr:
                try:
                    from .image_ocr import ImageOCR
                    ocr_instance = ImageOCR(use_gpu=False)
                    ocr_available = True
                    print("图片OCR已初始化")
                except Exception as e:
                    print(f"图片OCR模块加载失败: {e}")

            for post in posts:
                content = post.get('content', '')
                publish_time = post.get('publish_time', '')
                image_paths = post.get('image_paths', [])

                try:
                    if isinstance(publish_time, str):
                        post_time = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
                    else:
                        post_time = publish_time
                except:
                    post_time = None

                time_weight = self._calculate_time_weight(post_time, now)

                # 图片OCR识别
                image_ocr_stocks = []
                if ocr_available and ocr_instance and image_paths:
                    try:
                        for img_path in image_paths:
                            if img_path and os.path.exists(img_path):
                                img_result = ocr_instance.analyze_image(img_path)
                                if img_result.stock_names:
                                    image_ocr_stocks.extend(img_result.stock_names)
                    except Exception as e:
                        print(f"图片OCR处理失败: {e}")

                # 提取股票
                stocks = self._extract_stocks(content)
                all_stocks = list(set(stocks + image_ocr_stocks))

                for stock in all_stocks:
                    if stock.startswith('[板块]'):
                        sector_name = stock.replace('[板块]', '')
                        sector_counter[sector_name] += 1
                        continue

                    stock_counter[stock] += 1
                    is_us_market_discussion = any(kw in content for kw in ['美股', '纳斯达克', '费城半导体', '纳指', '道指'])
                    if is_us_market_discussion and stock in ['NVIDIA', 'Tesla', 'AMD', 'Intel', 'Micron', 'ASML', 'Corning']:
                        stock_signals[stock]['watch'].append((time_weight * 0.5, publish_time, content[:100]))
                        continue

                    signal_type, confidence, is_negated, reason = self._analyze_signal_with_context(content, stock)
                    final_weight = confidence * time_weight
                    context = f"[{publish_time}] {content[:80]}... ({reason})"
                    stock_signals[stock][signal_type].append((final_weight, publish_time, context))

                for industry, keywords in self.INDUSTRY_KEYWORDS.items():
                    if any(kw in content for kw in keywords):
                        industry_counter[industry] += time_weight
                        industry_signal_times[industry].append(post_time or now)

            # 股票→板块映射：正则提取的股票自动归类到行业
            for stock_name, count in stock_counter.items():
                stock_sector = get_stock_sector(stock_name)
                if stock_sector:
                    rollup = self.SECTOR_ROLLUP.get(stock_sector, stock_sector)
                    industry_counter[rollup] += count * 0.8
                    # 每次提及记录一个时间信号（用now近似，因为正则路径已包含time_weight）
                    for _ in range(min(count, 5)):
                        industry_signal_times[rollup].append(now)

        # ===== 行业时序密度加成 =====
        for ind_name, times in industry_signal_times.items():
            if len(times) < 2:
                continue
            # 检查短窗口内的信号密度
            for window in self.temporal_windows:
                recent_count = sum(1 for t in times if t and (now - t).days <= window)
                if recent_count >= 3:
                    # 短窗口内多次提及同一板块 → 密度加成
                    bonus = self.signal_density_bonus ** min(recent_count - 2, 2)
                    window_factor = self.temporal_windows[-1] / window
                    density_bonus = bonus * (1 + 0.05 * window_factor)
                    industry_counter[ind_name] *= density_bonus
                    break
                elif recent_count >= 2:
                    bonus = self.signal_density_bonus ** 0.3
                    window_factor = self.temporal_windows[-1] / window
                    density_bonus = bonus * (1 + 0.02 * window_factor)
                    industry_counter[ind_name] *= density_bonus
                    break

        # 汇总结果
        buy_stocks = {}
        sell_stocks = {}
        hold_stocks = {}
        watch_stocks = {}
        no_position_stocks = {}

        for stock, signals in stock_signals.items():
            # 计算各类信号的总权重
            sell_weight = sum(s[0] for s in signals.get('sell', []))
            no_pos_weight = sum(s[0] for s in signals.get('no_position', []))
            hold_weight = sum(s[0] for s in signals.get('hold', []))
            buy_weight = sum(s[0] for s in signals.get('buy', []))
            watch_weight = sum(s[0] for s in signals.get('watch', []))

            # 获取最近的上下文
            all_contexts = []
            for sig_type in ['sell', 'no_position', 'hold', 'buy', 'watch']:
                for weight, time, ctx in signals.get(sig_type, []):
                    all_contexts.append((weight, time, ctx, sig_type))
            all_contexts.sort(key=lambda x: x[1] if x[1] else '', reverse=True)
            top_contexts = [c[2] for c in all_contexts[:3]]

            # 信号优先级决策：sell > no_position > hold > buy > watch
            # 但需要考虑权重，如果买入权重远大于卖出权重，可能是先卖后买
            if sell_weight > 0 and sell_weight >= buy_weight * 0.5:
                sell_stocks[stock] = top_contexts
            elif no_pos_weight > 0 and no_pos_weight >= buy_weight * 0.5:
                no_position_stocks[stock] = top_contexts
            elif hold_weight > 0:
                hold_stocks[stock] = top_contexts
            elif buy_weight > 0:
                buy_stocks[stock] = top_contexts
            elif watch_weight > 0:
                watch_stocks[stock] = top_contexts

        # 构建结果
        current_positions = []
        for stock in set(list(buy_stocks.keys()) + list(hold_stocks.keys())):
            if stock not in sell_stocks and stock not in no_position_stocks:
                mention_count = stock_counter.get(stock, 0)
                context_list = buy_stocks.get(stock, []) + hold_stocks.get(stock, [])
                current_positions.append({
                    'name': stock,
                    'mention_count': mention_count,
                    'action': '持有' if stock in hold_stocks else '买入',
                    'contexts': context_list[:3]
                })

        watch_positions = []
        for stock in watch_stocks:
            if stock not in buy_stocks and stock not in hold_stocks and stock not in sell_stocks and stock not in no_position_stocks:
                watch_positions.append({
                    'name': stock,
                    'mention_count': stock_counter.get(stock, 0),
                    'action': '关注',
                    'contexts': watch_stocks[stock][:3]
                })

        closed_positions = []
        for stock in sell_stocks:
            closed_positions.append({
                'name': stock,
                'mention_count': stock_counter.get(stock, 0),
                'action': '已清仓',
                'contexts': sell_stocks[stock][:3]
            })

        result.current_positions = sorted(current_positions, key=lambda x: x['mention_count'], reverse=True)[:10]
        result.watch_positions = sorted(watch_positions, key=lambda x: x['mention_count'], reverse=True)[:10]
        result.closed_positions = sorted(closed_positions, key=lambda x: x['mention_count'], reverse=True)[:5]
        result.sector_distribution = dict(sector_counter.most_common(10))

        # 归一化 industry_counter 到 [0, 10]
        if industry_counter:
            max_weight = max(industry_counter.values())
            if max_weight > 0:
                industry_counter = Counter({
                    k: (v / max_weight) * 10
                    for k, v in industry_counter.items()
                })

        result.industry_distribution = dict(industry_counter.most_common(10))

        if industry_counter:
            # 只取正权重的板块，附带sentiment标签
            bullish_industries = [(k, v) for k, v in industry_counter.most_common(20) if v > 0]
            hot_list = []
            for ind, weight in sorted(bullish_industries, key=lambda x: x[1], reverse=True)[:5]:
                # 判断行业情感：优先看LLM提取结果，其次看信号权重
                sentiment = 'neutral'
                if llm_extraction and llm_extraction.method == "llm":
                    bullish_w = sum(s.confidence for s in llm_extraction.sectors
                                    if self.SECTOR_ROLLUP.get(s.name, s.name) == ind and s.sentiment == 'bullish')
                    bearish_w = sum(s.confidence for s in llm_extraction.sectors
                                    if self.SECTOR_ROLLUP.get(s.name, s.name) == ind and s.sentiment == 'bearish')
                    if bullish_w > bearish_w:
                        sentiment = 'bullish'
                    elif bearish_w > bullish_w:
                        sentiment = 'bearish'
                else:
                    # 正则路径：归一化后权重>5标记bullish（对应原>3，归一化后50%位）
                    if weight > 5:
                        sentiment = 'bullish'
                hot_list.append({'name': ind, 'sentiment': sentiment, 'weight': round(weight, 2)})
            result.hot_industries = hot_list

        # 仓位风格判断（基于实际持仓数量）
        position_count = len(current_positions)
        if position_count > 8:
            result.position_style = "激进（持仓分散，多只股票）"
        elif position_count > 4:
            result.position_style = "稳健（持仓适中）"
        else:
            result.position_style = "保守（持仓集中）"

        for market, keywords in self.MARKET_KEYWORDS.items():
            if any(kw in self.all_text for kw in keywords):
                result.market_type = market
                break
        if not result.market_type:
            result.market_type = "A股（默认）"

        position_results = self._analyze_likely_positions(posts, stock_counter, buy_stocks, hold_stocks, sell_stocks, llm_extraction)
        result.likely_positions = position_results.get('likely_positions', [])
        result.bearish_positions = position_results.get('bearish_positions', [])
        result.mentioned_positions = position_results.get('mentioned_positions', [])

    def _update_position_from_llm(self, extraction: ExtractionResult, result: AnalysisResult):
        """从LLM提取结果更新仓位配置"""
        bullish_sectors = [s for s in extraction.sectors if s.sentiment == "bullish" and s.confidence >= 0.5]
        bearish_sectors = [s for s in extraction.sectors if s.sentiment == "bearish" and s.confidence >= 0.5]

        # 如果LLM识别到了看好的板块，补充到position_allocation
        if bullish_sectors and not result.position_allocation:
            for sec in bullish_sectors:
                rollup = self.SECTOR_ROLLUP.get(sec.name, sec.name)
                if rollup not in result.position_allocation:
                    result.position_allocation[rollup] = int(sec.confidence * 30)

        # 看空的板块补充到no_position_sectors
        for sec in bearish_sectors:
            rollup = self.SECTOR_ROLLUP.get(sec.name, sec.name)
            if rollup not in result.no_position_sectors:
                result.no_position_sectors.append(rollup)

    def _temporal_aggregate(self, stock_signals: Dict, now: datetime = None) -> Dict:
        """
        时序聚合：按时间窗口聚合信号，近期信号权重更高

        对每个股票/板块的信号列表：
        1. 计算每条信号的指数衰减时间权重
        2. 按时间窗口分组，检测信号密度（同一窗口内多次提及=加成）
        3. 检测近期立场一致性（同向信号加成，矛盾信号减权）
        4. 综合输出加权分数

        Args:
            stock_signals: {stock_name: [{"time": datetime, "signal": str, "score": float, "sentiment": str}, ...]}
            now: 当前时间

        Returns:
            {stock_name: {"score": float, "count": int, "sentiment": str, "density_bonus": float, "consistency_bonus": float, "time_weighted_count": float}}
        """
        if now is None:
            now = datetime.now()

        results = {}

        for stock_name, signals in stock_signals.items():
            if not signals:
                continue

            # Step 1: 计算每条信号的时间权重
            weighted_signals = []
            for sig in signals:
                tw = self._calculate_time_weight(sig.get("time"), now)
                weighted_signals.append({
                    "time": sig.get("time"),
                    "signal": sig.get("signal", ""),
                    "score": sig.get("score", 0.5),
                    "sentiment": sig.get("sentiment", "neutral"),
                    "time_weight": tw
                })

            # Step 2: 按时间窗口分组，计算信号密度
            window_counts = {w: [] for w in self.temporal_windows}
            for ws in weighted_signals:
                t = ws.get("time")
                if not t:
                    continue
                try:
                    if isinstance(t, str):
                        t = datetime.fromisoformat(t.replace('Z', '+00:00'))
                    days_diff = (now - t).days
                except:
                    continue

                for window in self.temporal_windows:
                    if days_diff <= window:
                        window_counts[window].append(ws)

            # 密度加成：在短窗口内多次出现
            density_bonus = 1.0
            for window in self.temporal_windows:
                count_in_window = len(window_counts[window])
                if count_in_window >= 3:
                    # 短窗口内3次以上提及，大幅加成
                    bonus = self.signal_density_bonus ** min(count_in_window - 2, 3)
                    # 短窗口加成更大
                    window_factor = self.temporal_windows[-1] / window
                    density_bonus = max(density_bonus, bonus * (1 + 0.1 * window_factor))
                    break  # 取最短有效窗口的加成
                elif count_in_window >= 2:
                    bonus = self.signal_density_bonus ** 0.5
                    window_factor = self.temporal_windows[-1] / window
                    density_bonus = max(density_bonus, bonus * (1 + 0.05 * window_factor))
                    break

            # Step 3: 立场一致性检测（仅看近30天信号）
            recent_signals = [ws for ws in weighted_signals if ws["time_weight"] > 0.1]
            consistency_bonus = 1.0
            if recent_signals:
                sentiments = [ws["sentiment"] for ws in recent_signals]
                bullish_count = sentiments.count("bullish")
                bearish_count = sentiments.count("bearish")
                total = len(sentiments)

                if total >= 2:
                    # 立场一致度 = max方向占比
                    dominant_ratio = max(bullish_count, bearish_count) / total
                    if dominant_ratio >= 0.7:
                        # 70%以上同向 → 一致性加成
                        consistency_bonus = self.signal_consistency_bonus * dominant_ratio
                    elif dominant_ratio < 0.5:
                        # 立场矛盾 → 减权
                        consistency_bonus = 0.7

            # Step 4: 综合加权分数
            time_weighted_score = sum(
                ws["score"] * ws["time_weight"] for ws in weighted_signals
            )
            time_weighted_count = sum(ws["time_weight"] for ws in weighted_signals)

            # 最终分数 = 时间加权分数 × 密度加成 × 一致性加成
            final_score = time_weighted_score * density_bonus * consistency_bonus

            # 主导立场
            recent_sentiments = [ws["sentiment"] for ws in weighted_signals if ws["time_weight"] > 0.2]
            if recent_sentiments:
                from collections import Counter
                sentiment_counter = Counter(recent_sentiments)
                dominant_sentiment = sentiment_counter.most_common(1)[0][0]
            else:
                dominant_sentiment = "neutral"

            results[stock_name] = {
                "score": round(final_score, 4),
                "count": len(signals),
                "sentiment": dominant_sentiment,
                "density_bonus": round(density_bonus, 3),
                "consistency_bonus": round(consistency_bonus, 3),
                "time_weighted_count": round(time_weighted_count, 3)
            }

        return results

    def _analyze_likely_positions(self, posts: List[Dict], stock_counter: Counter,
                                   buy_stocks: Dict, hold_stocks: Dict, sell_stocks: Dict,
                                   llm_extraction: Optional[ExtractionResult] = None) -> Dict:
        """分析个股持仓推断 - 区分看好/看空/仅讨论，含时序聚合"""
        now = datetime.now()
        # 收集每只股票的所有信号（带时间戳），用于时序聚合
        stock_signal_list = defaultdict(list)  # {stock_name: [{"time", "signal", "score", "sentiment"}, ...]}

        # ===== LLM路径 =====
        if llm_extraction and llm_extraction.method == "llm" and llm_extraction.stocks:
            for s in llm_extraction.stocks:
                # LLM单次提取没有每条微博的时间戳，使用当前时间近似
                score = s.confidence * 2.0
                if s.sentiment == "bullish" and s.signal in ("买入", "加仓", "建仓"):
                    score *= 1.5
                stock_signal_list[s.name].append({
                    "time": now,
                    "signal": s.signal or "",
                    "score": score,
                    "sentiment": s.sentiment,
                })

        # ===== 正则路径（作为LLM补充或回退） =====
        for post in posts:
            content = post.get('content', '')
            publish_time = post.get('publish_time', '')
            if not content:
                continue

            try:
                if isinstance(publish_time, str):
                    post_time = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
                else:
                    post_time = publish_time
            except:
                post_time = None

            stocks = self._extract_stocks(content)
            for stock in stocks:
                if stock.startswith('[板块]'):
                    continue

                is_us_market_discussion = any(kw in content for kw in ['美股', '纳斯达克', '费城半导体', '纳指', '道指'])
                if is_us_market_discussion and stock in ['NVIDIA', 'Tesla', 'AMD', 'Intel', 'Micron', 'ASML', 'Corning']:
                    continue

                signal_type, confidence, is_negated, reason = self._analyze_signal_with_context(content, stock)

                if signal_type in ('buy', 'hold') and not is_negated:
                    score = confidence * 3
                    sentiment = 'bullish'
                elif signal_type in ('sell', 'no_position'):
                    score = confidence * 2
                    sentiment = 'bearish'
                else:
                    score = 0.3
                    sentiment = 'neutral'

                stock_signal_list[stock].append({
                    "time": post_time or now,
                    "signal": reason,
                    "score": score,
                    "sentiment": sentiment,
                })

        # ===== 时序聚合 =====
        temporal_results = self._temporal_aggregate(stock_signal_list, now)

        # ===== 按聚合结果分类到看好/看空/仅讨论 =====
        bullish_positions = {}
        bearish_positions = {}
        neutral_positions = {}

        for stock_name, agg in temporal_results.items():
            dominant_sentiment = agg["sentiment"]
            entry = {
                'score': agg["score"],
                'density_bonus': agg["density_bonus"],
                'consistency_bonus': agg["consistency_bonus"],
                'time_weighted_count': agg["time_weighted_count"],
                'reasons': [],
                'contexts': [],
                'latest_time': '',
                'sector': get_stock_sector(stock_name) or '',
                'sentiment': dominant_sentiment,
                'signal': '',
                'confidence': agg["score"] / max(agg["time_weighted_count"], 0.01),
            }

            # 收集该股票的原始信号信息
            sig_list = stock_signal_list[stock_name]
            # 按时间排序，取最近的
            sorted_sigs = sorted(sig_list, key=lambda x: str(x.get("time", "")), reverse=True)
            for sig in sorted_sigs[:3]:
                entry['reasons'].append(sig.get('signal', '') or (sig.get('sentiment', '')))
                entry['signal'] = sig.get('signal', '') or entry['signal']

            # 取最近的时间作为latest_time
            latest_time_str = str(sorted_sigs[0].get("time", "")) if sorted_sigs else ""
            entry['latest_time'] = latest_time_str[:19] if latest_time_str else ""

            if dominant_sentiment == "bullish":
                bullish_positions[stock_name] = entry
            elif dominant_sentiment == "bearish":
                bearish_positions[stock_name] = entry
            else:
                neutral_positions[stock_name] = entry

        # ===== 构建结果 =====
        def _build_list(positions_dict, top_n=8, min_score=0.5):
            result_list = []
            for name, info in sorted(positions_dict.items(), key=lambda x: x[1]['score'], reverse=True):
                if info['score'] < min_score:
                    continue
                # 排除冲突：看多的股票如果在卖出列表且不在买入列表
                if info['sentiment'] == 'bullish' and name in sell_stocks and name not in buy_stocks and name not in hold_stocks:
                    continue
                result_list.append({
                    'name': name,
                    'score': round(info['score'], 2),
                    'reasons': info['reasons'][:3],
                    'contexts': info['contexts'][:2],
                    'mention_count': stock_counter.get(name, 0),
                    'latest_time': info['latest_time'],
                    'sector': info.get('sector', ''),
                    'sentiment': info.get('sentiment', 'neutral'),
                    'signal': info.get('signal', ''),
                    'confidence': round(info.get('confidence', 0.5), 2),
                    'density_bonus': info.get('density_bonus', 1.0),
                    'consistency_bonus': info.get('consistency_bonus', 1.0),
                })
                if len(result_list) >= top_n:
                    break
            return result_list

        return {
            'likely_positions': _build_list(bullish_positions, min_score=1.0),
            'bearish_positions': _build_list(bearish_positions, min_score=0.5),
            'mentioned_positions': _build_list(neutral_positions, min_score=0.3),
        }

    def _extract_stocks(self, text: str) -> List[str]:
        """提取股票名称 - v5: stock_aliases + STOCK_SECTOR_MAP全名匹配"""
        stocks = []
        matched_names = set()  # 已匹配的股票全名，避免重复

        # 1. 常见A股股票简称/别名映射
        stock_aliases = {
            # 光模块/光通信
            '中际旭创': '中际旭创', '中际': '中际旭创',
            '天孚通信': '天孚通信', '天孚': '天孚通信',
            '新易盛': '新易盛',
            '光迅科技': '光迅科技', '光迅': '光迅科技',
            '长飞光纤': '长飞光纤', '长飞': '长飞光纤',
            '亨通光电': '亨通光电', '亨通': '亨通光电',
            '中天科技': '中天科技', '中天': '中天科技',
            '太辰光': '太辰光',
            '联特科技': '联特科技',
            '剑桥科技': '剑桥科技',

            # 新能源/电池/锂电
            '宁德时代': '宁德时代', '宁德': '宁德时代',
            '比亚迪': '比亚迪',
            '隆基绿能': '隆基绿能', '隆基': '隆基绿能',
            '通威股份': '通威股份', '通威': '通威股份',
            '阳光电源': '阳光电源',
            '国轩高科': '国轩高科', '国轩': '国轩高科',
            '亿纬锂能': '亿纬锂能', '亿纬': '亿纬锂能',
            '容百科技': '容百科技',
            '融捷股份': '融捷股份', '融捷': '融捷股份',
            '天齐锂业': '天齐锂业', '天齐': '天齐锂业',
            '赣锋锂业': '赣锋锂业', '赣锋': '赣锋锂业',
            '华友钴业': '华友钴业',
            '恩捷股份': '恩捷股份', '恩捷': '恩捷股份',
            '璞泰来': '璞泰来',
            '昊华能源': '昊华能源',
            '蔚蓝锂芯': '蔚蓝锂芯',
            '海科新源': '海科新源',

            # 半导体/芯片
            '寒武纪': '寒武纪',
            '海光信息': '海光信息', '海光': '海光信息',
            '中芯国际': '中芯国际', '中芯': '中芯国际',
            '北方华创': '北方华创',
            '韦尔股份': '韦尔股份', '韦尔': '韦尔股份',
            '紫光国微': '紫光国微',
            '圣邦股份': '圣邦股份', '圣邦': '圣邦股份',
            '兆易创新': '兆易创新',
            '澜起科技': '澜起科技',
            '龙芯中科': '龙芯中科',
            '复旦微电': '复旦微电',
            '乐鑫信息': '乐鑫信息',
            '瑞芯微': '瑞芯微',
            '全志科技': '全志科技',
            '晶晨股份': '晶晨股份',

            # 消费/白酒
            '贵州茅台': '贵州茅台', '茅台': '贵州茅台',
            '五粮液': '五粮液',
            '泸州老窖': '泸州老窖',
            '山西汾酒': '山西汾酒', '汾酒': '山西汾酒',
            '洋河股份': '洋河股份', '洋河': '洋河股份',
            '古井贡酒': '古井贡酒',
            '今世缘': '今世缘',
            '迎驾贡酒': '迎驾贡酒',
            '舍得酒业': '舍得酒业', '舍得': '舍得酒业',
            '酒鬼酒': '酒鬼酒',

            # 金融
            '东方财富': '东方财富', '东财': '东方财富',
            '中信证券': '中信证券',
            '招商银行': '招商银行', '招行': '招商银行',
            '宁波银行': '宁波银行',
            '平安银行': '平安银行',
            '兴业银行': '兴业银行',
            '常熟银行': '常熟银行',
            '成都银行': '成都银行',
            '江苏银行': '江苏银行',
            '杭州银行': '杭州银行',

            # 医药
            '药明康德': '药明康德',
            '恒瑞医药': '恒瑞医药', '恒瑞': '恒瑞医药',
            '迈瑞医疗': '迈瑞医疗', '迈瑞': '迈瑞医疗',
            '长春高新': '长春高新',
            '片仔癀': '片仔癀',
            '云南白药': '云南白药',
            '同仁堂': '同仁堂',
            '东阿阿胶': '东阿阿胶',
            '华润三九': '华润三九',
            '以岭药业': '以岭药业',

            # 算力/AI相关
            '网宿科技': '网宿科技', '网宿': '网宿科技',
            '工业富联': '工业富联',
            '立讯精密': '立讯精密', '立讯': '立讯精密',
            '歌尔股份': '歌尔股份', '歌尔': '歌尔股份',
            '京东方A': '京东方A', '京东方': '京东方A',
            'TCL科技': 'TCL科技',
            '浪潮信息': '浪潮信息', '浪潮': '浪潮信息',
            '中科曙光': '中科曙光', '曙光': '中科曙光',
            '紫光股份': '紫光股份',
            '拓维信息': '拓维信息',

            # 机械/制造
            '三一重工': '三一重工', '三一': '三一重工',
            '恒立液压': '恒立液压',
            '汇川技术': '汇川技术', '汇川': '汇川技术',
            '浙江鼎力': '浙江鼎力',
            '艾迪精密': '艾迪精密',

            # 商业航天
            '航天电子': '航天电子',
            '中国卫星': '中国卫星',
            '航天电器': '航天电器',
            '中航光电': '中航光电',
            '振华科技': '振华科技',
            '鸿远电子': '鸿远电子',
            '火炬电子': '火炬电子',
            '宏达电子': '宏达电子',
            '中航沈飞': '中航沈飞', '沈飞': '中航沈飞',
            '航发动力': '航发动力', '航发': '航发动力',

            # 有色/资源/煤炭
            '紫金矿业': '紫金矿业',
            '中国神华': '中国神华', '神华': '中国神华',
            '陕西煤业': '陕西煤业', '陕煤': '陕西煤业',
            '兖矿能源': '兖矿能源', '兖矿': '兖矿能源',
            '潞安环能': '潞安环能',
            '中煤能源': '中煤能源',
            '淮北矿业': '淮北矿业',
            '平煤股份': '平煤股份',

            # 汽车
            '赛力斯': '赛力斯',
            '长城汽车': '长城汽车',
            '长安汽车': '长安汽车',
            '上汽集团': '上汽集团',
            '广汽集团': '广汽集团',
            '江淮汽车': '江淮汽车',

            # 纺织服装
            '百隆东方': '百隆东方',
            '华孚时尚': '华孚时尚', '华孚': '华孚时尚',
            '鲁泰A': '鲁泰A',
            '伟星股份': '伟星股份',
            '航民股份': '航民股份',
            '华利集团': '华利集团',
            '健盛集团': '健盛集团',

            # 地产/基建
            '保利发展': '保利发展', '保利': '保利发展',
            '万科A': '万科A', '万科': '万科A',
            '招商蛇口': '招商蛇口',
            '金地集团': '金地集团',
            '中国建筑': '中国建筑',

            # 其他热门
            '沐曦股份': '沐曦股份',
            '摩尔线程': '摩尔线程',
            '德明利': '德明利',
            '石头科技': '石头科技',
            '爱美客': '爱美客',
            '禾迈股份': '禾迈股份',
            '昱能科技': '昱能科技',
            '固德威': '固德威',
            '锦浪科技': '锦浪科技',
            '德业股份': '德业股份',

            # 美股
            '英伟达': 'NVIDIA', 'NVIDIA': 'NVIDIA',
            '特斯拉': 'Tesla', 'Tesla': 'Tesla',
            'AMD': 'AMD',
            '英特尔': 'Intel', 'Intel': 'Intel',
            '美光': 'Micron', 'Micron': 'Micron',
            '阿斯麦': 'ASML', 'ASML': 'ASML',
            '康宁': 'Corning', 'Corning': 'Corning',
        }

        # Step 1: stock_aliases 匹配
        for alias, full_name in stock_aliases.items():
            if full_name in matched_names:
                continue
            if len(alias) <= 1:
                continue
            if len(alias) == 2:
                # 2字别名：要求前有分隔符或文本开头，后有分隔符/助词/量词
                if re.search(rf'(^|[，,、；;。\s!！?？])({alias})([，,、；;。\s!！?？的了着过是])', text):
                    matched_names.add(full_name)
                    stocks.append(full_name)
                elif text.startswith(alias) and len(text) > len(alias) and text[len(alias)] in '，,、；;。的了着过是':
                    matched_names.add(full_name)
                    stocks.append(full_name)
            elif len(alias) == 3:
                # 3字别名：要求不是更长词的子串（后面不能紧跟汉字）
                if alias in text:
                    for m in re.finditer(re.escape(alias), text):
                        end = m.end()
                        if end >= len(text) or not text[end].isalpha() or ord(text[end]) > 0x9fff:
                            matched_names.add(full_name)
                            stocks.append(full_name)
                            break
            else:
                # 4字+别名：直接子串匹配
                if alias in text:
                    matched_names.add(full_name)
                    stocks.append(full_name)

        # Step 2: STOCK_SECTOR_MAP 全名匹配（按长度降序，长名优先）
        sorted_stock_names = sorted(STOCK_SECTOR_MAP.keys(), key=len, reverse=True)
        for name in sorted_stock_names:
            if name in matched_names:
                continue
            if len(name) < 2:
                continue
            if name in text:
                matched_names.add(name)
                stocks.append(name)

        # Step 3: 匹配 名称(6位代码) 格式
        code_pattern = r'([一-龥]{2,4})[（(](\d{6})[）)]'
        code_matches = re.findall(code_pattern, text)
        for name, code in code_matches:
            if name not in matched_names:
                matched_names.add(name)
                stocks.append(name)

        # Step 4: 噪声过滤
        noise_patterns = ['一个', '卖了一个', '买了一个', '这个', '那个', '另一个',
                          '愿意参与', '想参与', '准备参与', '正在参与',
                          '第一证券', '一个证券', '某个证券', '什么证券']
        noise_suffixes = {'科技', '能源', '证券', '银行', '保险', '集团', '股份', '板块', '基金', '指数'}
        filtered = []
        for s in stocks:
            if any(noise in s for noise in noise_patterns):
                continue
            if s in noise_suffixes:
                continue
            if len(s) <= 1:
                continue
            filtered.append(s)
        return filtered

    def _analyze_selection(self, result: AnalysisResult):
        """分析选股体系"""
        # 选股因子
        factor_keywords = {
            '看K线形态': ['K线', '形态', '均线', '支撑', '压力', '突破', '回踩', '箱体', 'W底', '头肩底', '趋势'],
            '看业绩数据': ['业绩', '盈利', '增长', '净利润', '营收', 'ROE', '财报', '业绩预告', '年报'],
            '看估值水平': ['估值', 'PE', 'PB', '市盈率', '市净率', '低估', '便宜', '价值'],
            '看量能资金': ['量能', '成交量', '放量', '缩量', '换手', '资金流入', '主力', '北向'],
            '追政策热点': ['政策', '利好', '热点', '概念', '题材', '风口', '主线', '龙头'],
            '看龙虎榜': ['龙虎榜', '机构', '游资', '北向', '外资', '席位'],
        }

        factor_scores = {}
        for factor, keywords in factor_keywords.items():
            score = sum(self.all_text.count(kw) for kw in keywords)
            if score > 0:
                factor_scores[factor] = score

        result.selection_factors = dict(sorted(factor_scores.items(), key=lambda x: x[1], reverse=True))

        # K线形态
        kline_patterns = []
        for pattern, keywords in self.KLINE_PATTERNS.items():
            if any(kw in self.all_text for kw in keywords):
                kline_patterns.append(pattern)
        result.kline_patterns = kline_patterns

        # 不选的股票
        avoid_keywords = {
            'ST股、退市风险股': ['ST', 'st', '退市', '*ST'],
            '高位追涨股': ['高位', '追高', '涨幅过大', '翻倍'],
            '业绩亏损股': ['亏损', '业绩下滑', '暴雷', '亏损股'],
            '解禁减持股': ['解禁', '减持', '套现'],
            '妖股、纯炒作': ['妖股', '纯炒作', '无业绩'],
        }
        avoid_stocks = []
        for avoid_type, keywords in avoid_keywords.items():
            if any(kw in self.all_text for kw in keywords):
                avoid_stocks.append(avoid_type)
        result.avoid_stocks = avoid_stocks

        # 生成选股口诀
        top_factors = list(result.selection_factors.keys())[:3] if result.selection_factors else []
        if top_factors:
            result.selection_formula = f"先看{top_factors[0].replace('看', '')}，再看{top_factors[1].replace('看', '') if len(top_factors) > 1 else '仓位'}，最后看{top_factors[2].replace('看', '') if len(top_factors) > 2 else '时机'}"
        else:
            result.selection_formula = "数据不足，无法总结选股口诀"

    def _analyze_trading_system(self, result: AnalysisResult):
        """分析交易体系"""
        # 交易周期
        cycle_scores = {}
        for cycle, keywords in self.CYCLE_KEYWORDS.items():
            score = sum(self.all_text.count(kw) for kw in keywords)
            if score > 0:
                cycle_scores[cycle] = score

        if cycle_scores:
            result.trading_cycle = max(cycle_scores, key=cycle_scores.get)
            result.trading_cycle_detail = result.trading_cycle

        # 止盈点位
        stop_profit_match = re.search(r'止盈[点位的]?(\d+)[%点]', self.all_text)
        if stop_profit_match:
            result.stop_profit = f"{stop_profit_match.group(1)}%"
        elif '止盈' in self.all_text or '落袋' in self.all_text:
            result.stop_profit = "有提及止盈，但未明确点位"

        # 止损点位
        stop_loss_match = re.search(r'止损[点位的]?(\d+)[%点]', self.all_text)
        if stop_loss_match:
            result.stop_loss = f"{stop_loss_match.group(1)}%"
        elif '止损' in self.all_text or '破位' in self.all_text:
            result.stop_loss = "有提及止损，但未明确点位"

        # 买卖流程
        buy_signals = []
        sell_signals = []

        if '低吸' in self.all_text or '回调' in self.all_text:
            buy_signals.append("回调低吸")
        if '突破' in self.all_text:
            buy_signals.append("突破买入")
        if '企稳' in self.all_text or '支撑' in self.all_text:
            buy_signals.append("企稳进场")
        if '放量' in self.all_text:
            buy_signals.append("放量确认")

        if '止盈' in self.all_text or '落袋' in self.all_text:
            sell_signals.append("达到止盈目标")
        if '止损' in self.all_text or '破位' in self.all_text:
            sell_signals.append("跌破止损位")
        if '冲高' in self.all_text or '减仓' in self.all_text:
            sell_signals.append("冲高减仓")
        if '逻辑兑现' in self.all_text:
            sell_signals.append("逻辑兑现离场")

        if buy_signals and sell_signals:
            result.buy_sell_flow = f"买入信号：{' → '.join(buy_signals[:3])}\n卖出信号：{' → '.join(sell_signals[:3])}"
        else:
            result.buy_sell_flow = "数据不足，无法总结完整买卖流程"

    def _analyze_psychology(self, result: AnalysisResult, blogger_name: str = None):
        """分析交易心态 - 改进版：基于复盘准确率计算纪律评分"""
        # 统计常见错误（保留原有逻辑）
        mistake_scores = {}
        for mistake, keywords in self.MISTAKE_KEYWORDS.items():
            score = sum(self.all_text.count(kw) for kw in keywords)
            if score > 0:
                mistake_scores[mistake] = score

        sorted_mistakes = sorted(mistake_scores.items(), key=lambda x: x[1], reverse=True)
        result.common_mistakes = [m[0] for m in sorted_mistakes[:3]]

        # ==================== 问题4修复：基于复盘准确率计算纪律评分 ====================
        # 尝试从数据库获取该博主的复盘准确率
        try:
            from app.database import get_sync_db
            conn = get_sync_db()
            cursor = conn.cursor()

            # 查询该博主的预测准确率（最近30天）
            if blogger_name:
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN trend_match = 1 THEN 1 ELSE 0 END) as correct,
                           AVG(prediction_score) as avg_score
                    FROM review_results r
                    JOIN predictions p ON r.stock_code = p.stock_code AND r.review_date = p.prediction_date
                    WHERE p.blogger_name = ?
                    AND r.review_date >= DATE('now', '-30 days')
                """, (blogger_name,))
            else:
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN trend_match = 1 THEN 1 ELSE 0 END) as correct,
                           AVG(prediction_score) as avg_score
                    FROM review_results
                    WHERE review_date >= DATE('now', '-30 days')
                """)

            row = cursor.fetchone()
            conn.close()

            if row and row[0] > 0:
                total_predictions = row[0]
                correct_predictions = row[1] or 0
                avg_score = row[2] or 0

                # 基于实际准确率计算纪律评分
                # 准确率权重 60%，平均评分权重 40%
                accuracy_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
                accuracy_score = accuracy_rate * 60  # 最高60分
                avg_score_part = (avg_score / 100) * 40  # 最高40分

                result.discipline_score = int(accuracy_score + avg_score_part)
                print(f"纪律评分基于复盘数据: 准确率{accuracy_rate:.1%}, 平均评分{avg_score:.1f}, 最终{result.discipline_score}分")
            else:
                # 无复盘数据时，使用保守评分
                result.discipline_score = 50  # 中等分数，表示数据不足
                print("无复盘数据，使用默认纪律评分")

        except Exception as e:
            print(f"获取复盘数据失败: {e}")
            # 失败时使用原有逻辑（但降低权重）
            positive_count = sum(self.all_text.count(kw) for kw in ['纪律', '规则', '严格执行', '计划'])
            negative_count = sum(self.all_text.count(kw) for kw in ['后悔', '冲动', '情绪', '贪心', '恐惧', '犹豫', '死扛'])

            if positive_count + negative_count > 0:
                # 文字分析评分只占50%，因为不可靠
                text_score = int(positive_count / (positive_count + negative_count) * 50)
                result.discipline_score = 50 + text_score  # 基础50分 + 文字分析最多50分
            else:
                result.discipline_score = 50

    def _generate_practical_report(self, result: AnalysisResult) -> str:
        """生成实用分析报告 - 改进版"""
        report = f"""
# 📊 {result.blogger_name} 交易分析报告（抄作业版）

> 分析时间：{result.analysis_time.strftime('%Y-%m-%d %H:%M')}
> 数据来源：博主公开微博内容

---

## 一、仓位配置（核心！博主自报比例）

### 🔥 板块仓位比例
"""
        # 显示仓位配置（核心改进）
        if result.position_allocation:
            total = sum(result.position_allocation.values())
            for sector, ratio in sorted(result.position_allocation.items(), key=lambda x: x[1], reverse=True):
                report += f"\n- **{sector}**: {ratio}%"
            if total < 100:
                report += f"\n- **其他/现金**: 约{100-total}%"
        else:
            report += "\n暂未识别到明确的仓位比例数据"

        # 显示攻防配置
        if result.attack_sector or result.defense_sector:
            report += f"""

### ⚔️ 攻防配置
- **进攻板块**: {result.attack_sector or '未明确'}
- **防守板块**: {result.defense_sector or '未明确'}
"""

        # 显示明确不持有的板块
        if result.no_position_sectors:
            report += f"""

### ❌ 明确不持有的板块
"""
            for sector in result.no_position_sectors:
                report += f"\n- **{sector}**: 明确表示不持有/已清仓"

        report += f"""

---

## 二、持仓画像（她做什么股票？）

### 1. 主要市场
**{result.market_type}**

### 2. 仓位风格
**{result.position_style}**

### 3. 🔥 当前持仓（买入/持有中，未卖出）
"""
        if result.current_positions:
            for i, pos in enumerate(result.current_positions, 1):
                stock_name = pos['name']
                report += f"\n{i}. **{stock_name}** - 提及{pos['mention_count']}次 [{pos['action']}]"
                if pos.get('contexts'):
                    report += f"\n   > 最新动态: {pos['contexts'][0][:80]}..."
                # 检查是否是美股，如果是则显示A股概念股
                if stock_name in self.us_to_a_stock_mapping:
                    us_info = self.us_to_a_stock_mapping[stock_name]
                    report += f"\n   > 🌍 **美股**: {us_info['market']} | 业务: {us_info['business']}"
                    report += f"\n   > 📈 **A股相关概念股/供应链企业**:"
                    for a_stock in us_info['a_stock_concepts'][:3]:  # 显示前3个
                        report += f"\n   >   - {a_stock['code']} {a_stock['name']}: {a_stock['relation']}"
        else:
            report += "\n暂无明确持仓数据"

        report += f"""

### 4. 👀 关注但未买入的股票
"""
        if result.watch_positions:
            for pos in result.watch_positions:
                stock_name = pos['name']
                report += f"\n- **{stock_name}** - 提及{pos['mention_count']}次 [关注中]"
                # 检查是否是美股，如果是则显示A股概念股
                if stock_name in self.us_to_a_stock_mapping:
                    us_info = self.us_to_a_stock_mapping[stock_name]
                    report += f"\n  > 🌍 **美股**: {us_info['market']} | 业务: {us_info['business']}"
                    report += f"\n  > 📈 **A股相关概念股/供应链企业**:"
                    for a_stock in us_info['a_stock_concepts'][:4]:  # 显示前4个
                        report += f"\n  >   - {a_stock['code']} {a_stock['name']}: {a_stock['relation']}"
        else:
            report += "\n暂无明确关注股票"

        report += f"""

### 5. ✅ 已清仓的股票
"""
        if result.closed_positions:
            for pos in result.closed_positions:
                stock_name = pos['name']
                report += f"\n- **{stock_name}** - 提及{pos['mention_count']}次 [已清仓]"
                # 检查是否是美股，如果是则显示A股概念股
                if stock_name in self.us_to_a_stock_mapping:
                    us_info = self.us_to_a_stock_mapping[stock_name]
                    report += f"\n  > 🌍 **美股**: {us_info['market']} | 业务: {us_info['business']}"
                    report += f"\n  > 📈 **A股相关概念股/供应链企业**:"
                    for a_stock in us_info['a_stock_concepts'][:3]:  # 显示前3个
                        report += f"\n  >   - {a_stock['code']} {a_stock['name']}: {a_stock['relation']}"
        else:
            report += "\n暂无明确清仓记录"

        report += f"""

### 6. 重仓板块（经常提到的）
"""
        if result.hot_industries:
            for i, industry in enumerate(result.hot_industries, 1):
                if isinstance(industry, dict):
                    ind_name = industry.get('name', '')
                    ind_sentiment = industry.get('sentiment', '')
                    sentiment_label = {'bullish': '看好', 'bearish': '看空', 'neutral': '中性'}.get(ind_sentiment, '')
                    count = result.industry_distribution.get(ind_name, 0)
                    report += f"\n{i}. **{ind_name}** - 提及{count}次 ({sentiment_label})"
                else:
                    count = result.industry_distribution.get(industry, 0)
                    report += f"\n{i}. **{industry}** - 提及{count}次"
        else:
            report += "\n暂无明确重仓板块数据"

        report += f"""

### 7. 不碰的板块
"""
        if result.avoid_stocks:
            for avoid in result.avoid_stocks:
                report += f"\n- ❌ {avoid}"
        else:
            report += "\n暂未明确提及不碰的板块"

        report += f"""

---

## 二、选股体系（她怎么选股票？）

### 1. 选股看什么？（按重要性排序）
"""
        if result.selection_factors:
            total = sum(result.selection_factors.values())
            for i, (factor, score) in enumerate(list(result.selection_factors.items())[:5], 1):
                pct = int(score / total * 100)
                report += f"\n{i}. **{factor}** - 提及{score}次，占比{pct}%"
        else:
            report += "\n暂无明确选股因子数据"

        report += f"""

### 2. 具体看什么K线形态？
"""
        if result.kline_patterns:
            for pattern in result.kline_patterns:
                report += f"\n- {pattern}"
        else:
            report += "\n暂无明确K线形态数据"

        report += f"""

### 3. 什么股票绝对不选？
"""
        if result.avoid_stocks:
            for avoid in result.avoid_stocks:
                report += f"\n- ❌ {avoid}"
        else:
            report += "\n暂未明确提及"

        report += f"""

### 4. 📌 选股口诀（可直接照抄）
> **{result.selection_formula}**

---

## 三、交易体系（她怎么买卖？）

### 1. 交易周期
**{result.trading_cycle_detail if result.trading_cycle_detail else '数据不足，无法判断'}**

### 2. 止盈点位
**{result.stop_profit if result.stop_profit else '未明确提及'}**

### 3. 止损点位
**{result.stop_loss if result.stop_loss else '未明确提及'}**

### 4. 完整买卖流程
```
{result.buy_sell_flow}
```

---

## 四、交易心态&纪律（她的得分：{result.discipline_score}/100分）

### ⚠️ 她最常犯的3个交易错误（抄作业必须避开）
"""
        if result.common_mistakes:
            for i, mistake in enumerate(result.common_mistakes, 1):
                report += f"\n{i}. **{mistake}** - 这是她自己也承认的问题，你要特别注意避免"
        else:
            report += "\n暂未发现明显的交易错误记录"

        report += f"""

---

## 五、个人交易复盘对照表

| 对比项 | 博主做法 | 我的做法 | 差距分析 |
|--------|----------|----------|----------|
| 持仓风格 | {result.position_style} | （请填写） | |
| 选股逻辑 | {result.selection_formula[:20] if result.selection_formula else '数据不足'}... | （请填写） | |
| 交易周期 | {result.trading_cycle_detail if result.trading_cycle_detail else '数据不足'} | （请填写） | |
| 止盈止损 | 止盈{result.stop_profit if result.stop_profit else '未设'} / 止损{result.stop_loss if result.stop_loss else '未设'} | （请填写） | |
| 交易纪律 | {result.discipline_score}/100分 | （请填写） | |

---

## 六、抄作业建议

### ✅ 可以学的
"""
        if result.selection_factors:
            top_factor = list(result.selection_factors.keys())[0]
            report += f"\n- 选股思路：{top_factor}"
        if result.trading_cycle_detail:
            report += f"\n- 交易周期：{result.trading_cycle_detail}"
        if result.hot_industries:
            ind_names = [i.get('name', i) if isinstance(i, dict) else i for i in result.hot_industries[:3]]
            report += f"\n- 关注板块：{', '.join(ind_names)}"

        report += f"""

### ❌ 必须避开的坑
"""
        if result.common_mistakes:
            for mistake in result.common_mistakes:
                report += f"\n- {mistake}"
        else:
            report += "\n- 暂无明显问题"

        report += """

---

## 📝 分析说明

本报告基于博主公开微博内容分析，仅供学习参考：
- 博主的实盘操作可能与微博表述有差异
- 请结合自身情况调整，不要盲目跟投
- 投资有风险，决策需谨慎
"""
        return report

    def get_us_stock_a_concepts(self, us_stock_name: str) -> Optional[Dict]:
        """
        查询美股对应的A股概念股/供应链企业

        Args:
            us_stock_name: 美股名称（英文或中文）

        Returns:
            Dict: 包含美股信息和A股概念股列表，如果不存在返回None
        """
        # 支持中文和英文查询
        # 先检查英文名称
        if us_stock_name in self.us_to_a_stock_mapping:
            return self.us_to_a_stock_mapping[us_stock_name]

        # 再检查中文名称映射
        cn_to_en = {
            '英伟达': 'NVIDIA',
            '阿斯麦': 'ASML',
            '超威半导体': 'AMD',
            '英特尔': 'Intel',
            '美光科技': 'Micron',
            '康宁': 'Corning',
            '特斯拉': 'Tesla',
        }
        if us_stock_name in cn_to_en:
            return self.us_to_a_stock_mapping[cn_to_en[us_stock_name]]

        return None

    def list_all_us_stocks(self) -> List[Dict]:
        """
        列出所有支持的美股及其A股概念股

        Returns:
            List[Dict]: 所有美股信息列表
        """
        result = []
        for us_name, info in self.us_to_a_stock_mapping.items():
            result.append({
                'us_stock': us_name,
                'cn_name': info['name_cn'],
                'market': info['market'],
                'business': info['business'],
                'a_stock_sector': info['a_stock_sector'],
                'a_stock_count': len(info['a_stock_concepts']),
            })
        return result

    def generate_us_stock_report(self, us_stock_name: str) -> str:
        """
        生成单只美股的A股概念股报告

        Args:
            us_stock_name: 美股名称

        Returns:
            str: 格式化的报告文本
        """
        info = self.get_us_stock_a_concepts(us_stock_name)
        if not info:
            return f"未找到美股 '{us_stock_name}' 的相关信息。支持的美股: NVIDIA, ASML, AMD, Intel, Micron, Corning, Tesla"

        report = f"""
# 📊 {us_stock_name} ({info['name_cn']}) A股概念股分析

> 市场: {info['market']}
> 业务: {info['business']}
> A股相关板块: {info['a_stock_sector']}

---

## 📈 A股相关概念股/供应链企业

"""
        for i, a_stock in enumerate(info['a_stock_concepts'], 1):
            report += f"{i}. **{a_stock['code']} {a_stock['name']}**\n"
            report += f"   > 关系: {a_stock['relation']}\n\n"

        report += """
---

## 📝 说明

- 以上A股股票与该美股存在业务关联或供应链关系
- 关系描述基于公开信息，仅供参考
- 投资有风险，决策需谨慎
"""
        return report
