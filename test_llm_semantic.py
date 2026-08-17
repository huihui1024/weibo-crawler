"""
LLM语义分析测试脚本
测试场景区分和复杂语义理解能力
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.analyzer.llm_semantic import SemanticAnalyzer, HybridAnalyzer, analyze_with_hybrid
from app.analyzer.stock_analyzer import StockAnalyzer


def test_keyword_analysis():
    """测试关键词分析"""
    print("\n" + "="*60)
    print("测试关键词分析")
    print("="*60)

    analyzer = StockAnalyzer(enable_llm=False)  # 禁用LLM，纯关键词分析

    test_cases = [
        # 正常买入信号
        ("今天买了中际旭创，看好光模块板块", "中际旭创"),
        # 正常卖出信号
        ("清仓了宁德时代，落袋为安", "宁德时代"),
        # 否定信号
        ("我不看好茅台，没有买", "贵州茅台"),
        # 复杂否定
        ("本来想买比亚迪的，但是没买", "比亚迪"),
        # 假设性讨论（关键词可能误判）
        ("如果明天中际旭创突破前高，我就考虑买入", "中际旭创"),
        # 反讽（关键词可能误判）
        ("呵呵，今天买了宁德时代，笑死，怎么可能", "宁德时代"),
        # 模拟盘讨论
        ("模拟盘里我买了寒武纪，实盘不敢碰", "寒武纪"),
        # 举例说明
        ("比如中际旭创这种票，突破后回踩就是买点", "中际旭创"),
    ]

    for content, stock in test_cases:
        result = analyzer._analyze_signal_with_context(content, stock)
        print(f"\n内容: {content}")
        print(f"股票: {stock}")
        print(f"结果: 信号={result[0]}, 置信度={result[1]:.2f}, 否定={result[2]}, 原因={result[3]}")


def test_llm_analysis():
    """测试LLM语义分析（需要API配置）"""
    print("\n" + "="*60)
    print("测试LLM语义分析")
    print("="*60)

    # 检查是否有API配置
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not has_openai and not has_claude:
        print("警告: 未配置LLM API，将使用本地模型（如果可用）")
        print("提示: 设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY 环境变量以启用云端LLM")

    analyzer = SemanticAnalyzer()

    test_cases = [
        # 假设性讨论 - 应识别为simulation
        ("如果明天中际旭创突破前高，我就考虑买入", "中际旭创"),
        # 反讽 - 应识别为sarcastic
        ("呵呵，今天买了宁德时代，笑死，怎么可能", "宁德时代"),
        # 模拟盘讨论 - 应识别为simulation
        ("模拟盘里我买了寒武纪，实盘不敢碰", "寒武纪"),
        # 举例说明 - 应识别为example
        ("比如中际旭创这种票，突破后回踩就是买点", "中际旭创"),
        # 市场讨论 - 应识别为discussion
        ("光模块板块今天表现不错，中际旭创领涨", "中际旭创"),
        # 后悔情绪
        ("早知道就买比亚迪了，可惜当时犹豫了", "比亚迪"),
        # 真实实盘操作
        ("今天实盘加了中际旭创，仓位到了30%", "中际旭创"),
    ]

    for content, stock in test_cases:
        print(f"\n内容: {content}")
        print(f"股票: {stock}")
        try:
            result = analyzer.analyze(content, stock)
            print(f"场景类型: {result.scenario_type} (置信度: {result.scenario_confidence:.2f})")
            print(f"真实意图: {result.true_sentiment} (置信度: {result.sentiment_confidence:.2f})")
            print(f"交易信号: {result.action_type} (置信度: {result.action_confidence:.2f})")
            print(f"是否正式观点: {result.is_formal_opinion}")
            print(f"是否实盘: {result.is_real_position}")
            print(f"情绪: {result.emotion}")
            print(f"分析依据: {result.reasoning}")
        except Exception as e:
            print(f"分析失败: {e}")


def test_hybrid_analysis():
    """测试混合分析（关键词+LLM）"""
    print("\n" + "="*60)
    print("测试混合分析（关键词优先，LLM兜底）")
    print("="*60)

    analyzer = HybridAnalyzer()

    test_cases = [
        # 高置信度关键词 - 应使用关键词
        ("今天买了中际旭创，看好光模块板块", "中际旭创"),
        # 低置信度关键词 - 应使用LLM
        ("如果明天中际旭创突破前高，我就考虑买入", "中际旭创"),
        # 复杂否定 - 应使用LLM
        ("我不看好茅台，但是今天买了", "贵州茅台"),
        # 反讽 - 应使用LLM
        ("呵呵，今天买了宁德时代，笑死", "宁德时代"),
    ]

    for content, stock in test_cases:
        print(f"\n内容: {content}")
        print(f"股票: {stock}")
        try:
            result, method = analyzer.analyze_with_fallback(content, stock)
            print(f"分析方法: {method}")
            print(f"场景类型: {result.scenario_type}")
            print(f"交易信号: {result.action_type} (置信度: {result.action_confidence:.2f})")
            print(f"分析依据: {result.reasoning}")
        except Exception as e:
            print(f"分析失败: {e}")


def test_stock_analyzer_with_llm():
    """测试集成LLM的StockAnalyzer"""
    print("\n" + "="*60)
    print("测试集成LLM的StockAnalyzer")
    print("="*60)

    # 启用LLM分析
    analyzer = StockAnalyzer(enable_llm=True)

    test_cases = [
        # 假设性讨论
        ("如果明天中际旭创突破前高，我就考虑买入", "中际旭创"),
        # 模拟盘
        ("模拟盘里我买了寒武纪，实盘不敢碰", "寒武纪"),
        # 反讽
        ("呵呵，今天买了宁德时代，笑死", "宁德时代"),
        # 正常买入（高置信度，可能不需要LLM）
        ("今天实盘加了中际旭创，仓位到了30%", "中际旭创"),
    ]

    for content, stock in test_cases:
        result = analyzer._analyze_signal_with_context(content, stock)
        print(f"\n内容: {content}")
        print(f"股票: {stock}")
        print(f"结果: 信号={result[0]}, 置信度={result[1]:.2f}, 否定={result[2]}, 原因={result[3]}")


if __name__ == "__main__":
    print("="*60)
    print("LLM语义分析功能测试")
    print("="*60)

    # 测试关键词分析
    test_keyword_analysis()

    # 测试LLM分析（需要API配置）
    test_llm_analysis()

    # 测试混合分析
    test_hybrid_analysis()

    # 测试集成LLM的StockAnalyzer
    test_stock_analyzer_with_llm()

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)