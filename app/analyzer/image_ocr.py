"""
图片OCR识别模块
用于从微博图片中提取股票名称和交易信息
"""
import os
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# 全局OCR实例，避免重复初始化
_ocr_instance = None


@dataclass
class ImageAnalysisResult:
    """图片分析结果"""
    image_path: str
    stock_names: List[str]  # 识别出的股票名称
    stock_codes: List[str]  # 识别出的股票代码
    buy_signals: int = 0    # B点/买入信号数量
    sell_signals: int = 0   # S点/卖出信号数量
    raw_text: str = ""      # OCR原始文本


class ImageOCR:
    """图片OCR识别器"""

    # 常见A股股票名称（用于匹配）
    STOCK_NAMES = [
        # 纺织服装
        '百隆东方', '华孚时尚', '鲁泰A', '伟星股份', '航民股份', '华利集团', '健盛集团',
        # 光模块/通信
        '中际旭创', '天孚通信', '新易盛', '光迅科技', '亨通光电',
        '中天科技', '长飞光纤', '剑桥科技', '华工科技', '太辰光', '联特科技',
        # 半导体/芯片
        '寒武纪', '海光信息', '中芯国际', '北方华创', '韦尔股份',
        '紫光国微', '圣邦股份', '兆易创新', '澜起科技', '龙芯中科',
        '复旦微电', '乐鑫信息', '瑞芯微', '全志科技', '晶晨股份',
        # 新能源/锂电
        '宁德时代', '比亚迪', '隆基绿能', '通威股份', '阳光电源',
        '亿纬锂能', '国轩高科', '容百科技', '融捷股份', '天齐锂业',
        '赣锋锂业', '华友钴业', '中伟股份', '恩捷股份', '璞泰来',
        # 消费/白酒
        '贵州茅台', '五粮液', '泸州老窖', '山西汾酒', '洋河股份',
        '古井贡酒', '今世缘', '迎驾贡酒', '舍得酒业', '酒鬼酒',
        # 金融
        '东方财富', '中信证券', '招商银行', '宁波银行', '平安银行',
        '兴业银行', '常熟银行', '成都银行', '江苏银行', '杭州银行',
        # 医药
        '药明康德', '恒瑞医药', '迈瑞医疗', '长春高新', '片仔癀',
        '云南白药', '同仁堂', '东阿阿胶', '华润三九', '以岭药业',
        # 算力/AI
        '工业富联', '立讯精密', '歌尔股份', '京东方A', 'TCL科技',
        '网宿科技', '浪潮信息', '中科曙光', '紫光股份', '拓维信息',
        # 机械/制造
        '三一重工', '恒立液压', '汇川技术', '浙江鼎力', '艾迪精密',
        # 资源/有色
        '紫金矿业', '中国神华', '陕西煤业', '兖矿能源', '潞安环能',
        '昊华能源', '蔚蓝锂芯', '海科新源',
        # 军工/航天
        '航天电子', '中国卫星', '航天电器', '中航光电', '振华科技',
        '鸿远电子', '火炬电子', '宏达电子',
        # 汽车
        '赛力斯', '长城汽车', '长安汽车', '上汽集团', '广汽集团',
        '江淮汽车', '小鹏汽车', '理想汽车', '蔚来',
        # 地产/基建
        '保利发展', '万科A', '招商蛇口', '金地集团', '中国建筑',
        # 其他热门
        '沐曦股份', '摩尔线程', '德明利', '石头科技', '爱美客',
        '禾迈股份', '昱能科技', '固德威', '锦浪科技', '德业股份',
    ]

    # 股票代码正则模式
    CODE_PATTERNS = [
        r'[036]\d{5}',      # A股代码：0/3/6开头的6位数字
        r'688\d{3}',        # 科创板
        r'300\d{3}',        # 创业板
        r'002\d{3}',        # 中小板
        r'60[013]\d{3}',    # 主板
        r'00\d{4}',         # 深市
    ]

    def __init__(self, use_gpu: bool = False):
        """
        初始化OCR识别器

        Args:
            use_gpu: 是否使用GPU加速
        """
        self.reader = None
        self.use_gpu = use_gpu
        self._init_reader()

    def _init_reader(self):
        """延迟初始化EasyOCR"""
        global _ocr_instance

        # 如果全局实例已存在，直接使用
        if _ocr_instance is not None:
            self.reader = _ocr_instance
            return

        try:
            import easyocr
            # 支持中英文，禁用gpu如果不可用
            self.reader = easyocr.Reader(
                ['ch_sim', 'en'],
                gpu=self.use_gpu,
                verbose=False
            )
            # 保存到全局实例
            _ocr_instance = self.reader
            print("EasyOCR初始化成功")
        except Exception as e:
            print(f"EasyOCR初始化失败: {e}")
            self.reader = None

    def ocr_image(self, image_path: str) -> List[Tuple]:
        """
        对单张图片进行OCR识别

        Args:
            image_path: 图片路径

        Returns:
            List[Tuple]: OCR结果列表，每个元素为 (bbox, text, confidence)
        """
        if not self.reader:
            return []

        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return []

        try:
            results = self.reader.readtext(image_path)
            return results
        except Exception as e:
            print(f"OCR识别失败 {image_path}: {e}")
            return []

    def extract_stocks_from_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        从文本中提取股票名称和代码

        Args:
            text: OCR识别的文本

        Returns:
            Tuple[List[str], List[str]]: (股票名称列表, 股票代码列表)
        """
        stock_names = []
        stock_codes = []

        # 匹配股票名称
        for name in self.STOCK_NAMES:
            if name in text:
                stock_names.append(name)

        # 匹配股票代码
        for pattern in self.CODE_PATTERNS:
            codes = re.findall(pattern, text)
            stock_codes.extend(codes)

        # 去重
        stock_names = list(set(stock_names))
        stock_codes = list(set(stock_codes))

        return stock_names, stock_codes

    def detect_signals(self, text: str) -> Tuple[int, int]:
        """
        检测买入/卖出信号

        Args:
            text: OCR识别的文本

        Returns:
            Tuple[int, int]: (买入信号数, 卖出信号数)
        """
        buy_signals = 0
        sell_signals = 0

        # B点/买入信号
        buy_keywords = ['B', '买', '买入', '建仓', '低吸']
        for kw in buy_keywords:
            buy_signals += text.upper().count(kw.upper())

        # S点/卖出信号
        sell_keywords = ['S', '卖', '卖出', '清仓', '止盈']
        for kw in sell_keywords:
            sell_signals += text.upper().count(kw.upper())

        return buy_signals, sell_signals

    def analyze_image(self, image_path: str) -> ImageAnalysisResult:
        """
        分析单张图片

        Args:
            image_path: 图片路径

        Returns:
            ImageAnalysisResult: 分析结果
        """
        ocr_results = self.ocr_image(image_path)

        # 合并所有文本
        all_text = ' '.join([r[1] for r in ocr_results if len(r) > 1])

        # 提取股票信息
        stock_names, stock_codes = self.extract_stocks_from_text(all_text)

        # 检测信号
        buy_signals, sell_signals = self.detect_signals(all_text)

        return ImageAnalysisResult(
            image_path=image_path,
            stock_names=stock_names,
            stock_codes=stock_codes,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            raw_text=all_text
        )

    def analyze_images(self, image_paths: List[str]) -> List[ImageAnalysisResult]:
        """
        批量分析图片

        Args:
            image_paths: 图片路径列表

        Returns:
            List[ImageAnalysisResult]: 分析结果列表
        """
        results = []
        for path in image_paths:
            result = self.analyze_image(path)
            results.append(result)
        return results

    def get_all_stocks_from_images(self, image_paths: List[str]) -> Dict:
        """
        从多张图片中汇总所有股票信息

        Args:
            image_paths: 图片路径列表

        Returns:
            Dict: 汇总结果，包含股票名称、代码、信号等
        """
        all_stock_names = set()
        all_stock_codes = set()
        total_buy_signals = 0
        total_sell_signals = 0
        image_results = []

        for path in image_paths:
            result = self.analyze_image(path)
            image_results.append(result)

            all_stock_names.update(result.stock_names)
            all_stock_codes.update(result.stock_codes)
            total_buy_signals += result.buy_signals
            total_sell_signals += result.sell_signals

        return {
            'stock_names': list(all_stock_names),
            'stock_codes': list(all_stock_codes),
            'buy_signals': total_buy_signals,
            'sell_signals': total_sell_signals,
            'image_results': image_results,
            'has_sell_signal': total_sell_signals > total_buy_signals,
            'has_buy_signal': total_buy_signals > total_sell_signals,
        }


def analyze_weibo_images(image_dir: str, weibo_id: str) -> Optional[Dict]:
    """
    分析微博图片

    Args:
        image_dir: 图片根目录
        weibo_id: 微博ID

    Returns:
        Dict: 分析结果
    """
    # 查找微博对应的图片目录
    weibo_image_dir = Path(image_dir) / weibo_id

    if not weibo_image_dir.exists():
        return None

    # 获取所有图片
    image_files = list(weibo_image_dir.glob('*.jpg')) + \
                  list(weibo_image_dir.glob('*.png')) + \
                  list(weibo_image_dir.glob('*.jpeg'))

    if not image_files:
        return None

    # 初始化OCR
    ocr = ImageOCR(use_gpu=False)

    # 分析图片
    image_paths = [str(f) for f in image_files]
    result = ocr.get_all_stocks_from_images(image_paths)

    return result


if __name__ == "__main__":
    # 测试
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"分析图片: {image_path}")

        ocr = ImageOCR(use_gpu=False)
        result = ocr.analyze_image(image_path)

        print(f"\n识别结果:")
        print(f"股票名称: {result.stock_names}")
        print(f"股票代码: {result.stock_codes}")
        print(f"买入信号: {result.buy_signals}")
        print(f"卖出信号: {result.sell_signals}")
        print(f"\n原始文本:\n{result.raw_text[:500]}")
    else:
        # 测试已知图片
        test_images = [
            'data/images/5297693669982989/fc5f3df9_0.jpg',
            'data/images/5297686779790771/3e666d28_0.jpg',
        ]

        print("测试图片OCR识别...")
        ocr = ImageOCR(use_gpu=False)

        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"\n{'='*50}")
                print(f"图片: {img_path}")
                result = ocr.analyze_image(img_path)
                print(f"股票名称: {result.stock_names}")
                print(f"股票代码: {result.stock_codes}")
                print(f"买入信号: {result.buy_signals}")
                print(f"卖出信号: {result.sell_signals}")
