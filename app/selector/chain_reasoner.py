"""
产业链推理引擎
基于Serenity案例的核心洞察：真正的超额收益藏在产业链的"瓶颈节点"——
那些解决关键痛点、处于价值链重塑节点的隐形冠军。

功能：
1. 从博主提及的板块出发，沿产业链上下游推理
2. 发现瓶颈节点（具有定价权的公司）
3. LLM增强推理：当硬编码图谱无法覆盖时
"""
import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import yaml


@dataclass
class ChainReasoningResult:
    """产业链推理结果"""
    chain_path: str = ""           # 产业链路径 (如: GPU→互联→CPO→激光器)
    bottleneck_node: str = ""      # 瓶颈节点名称
    is_bottleneck: bool = False    # 股票是否处于瓶颈节点
    chain_position: str = "unknown"  # 上游/中游/下游/瓶颈节点/非瓶颈
    supply_chain_score: float = 50.0  # 产业链推理得分 [0,100]
    reasoning: str = ""            # 推理依据


# ==================== 产业链知识图谱 ====================

SUPPLY_CHAIN_GRAPH = {
    # AI算力产业链
    "GPU/算力": {
        "下游": ["数据中心", "AI服务器"],
        "瓶颈": ["互联带宽"],
    },
    "互联带宽": {
        "下游": ["光模块", "CPO", "硅光"],
        "瓶颈": ["激光器散热"],
    },
    "激光器散热": {
        "下游": ["ELS外部激光源", "DFB激光器", "CW光源"],
        "瓶颈": [],
    },
    "光模块": {
        "下游": ["800G光模块", "1.6T光模块"],
        "瓶颈": ["芯片供应"],
    },
    "CPO": {
        "下游": ["光引擎", "光纤连接器", "硅光芯片"],
        "瓶颈": ["良率", "封装产能"],
    },
    "硅光": {
        "下游": ["硅光芯片", "CW光源"],
        "瓶颈": ["工艺成熟度"],
    },
    "HBM": {
        "下游": ["TSV封装", "CoWoS", "先进封装"],
        "瓶颈": ["封装产能"],
    },
    "先进制程": {
        "下游": ["光刻机", "EDA", "掩膜板"],
        "瓶颈": ["光刻胶"],
    },

    # 新能源产业链
    "锂电池": {
        "下游": ["正极材料", "负极材料", "电解液", "隔膜"],
        "瓶颈": ["锂矿供应"],
    },
    "光伏": {
        "下游": ["硅料", "硅片", "电池片", "组件", "逆变器"],
        "瓶颈": ["硅料产能"],
    },
    "新能源车": {
        "下游": ["动力电池", "电机电控", "充电桩"],
        "瓶颈": ["充电基础设施"],
    },

    # 其他产业链
    "商业航天": {
        "下游": ["卫星制造", "火箭制造", "卫星导航", "地面设备"],
        "瓶颈": ["发射成本"],
    },
    "机器人": {
        "下游": ["减速器", "伺服电机", "控制器", "传感器"],
        "瓶颈": ["减速器精度"],
    },
    "创新药": {
        "下游": ["CRO", "CDMO", "原料药"],
        "瓶颈": ["临床审批"],
    },
    "半导体": {
        "下游": ["设计", "制造", "封测", "材料", "设备"],
        "瓶颈": ["光刻机", "EDA"],
    },
    "煤炭": {
        "下游": ["焦煤", "动力煤", "煤化工"],
        "瓶颈": [],
    },
}

# 板块关键词 → 图谱节点映射
SECTOR_TO_GRAPH_NODE = {
    "科技芯片": "半导体",
    "光模块": "光模块",
    "CPO": "CPO",
    "算力": "GPU/算力",
    "GPU": "GPU/算力",
    "AI服务器": "GPU/算力",
    "数据中心": "GPU/算力",
    "硅光": "硅光",
    "新能源": "锂电池",
    "锂电池": "锂电池",
    "光伏": "光伏",
    "储能": "光伏",
    "军工航天": "商业航天",
    "商业航天": "商业航天",
    "机器人": "机器人",
    "医药医疗": "创新药",
    "半导体": "半导体",
    "煤炭": "煤炭",
    "HBM": "HBM",
    "先进封装": "HBM",
    "光刻胶": "先进制程",
}


class ChainReasoner:
    """产业链推理引擎"""

    def __init__(self):
        self._llm_client = None
        self._init_llm()

    def _init_llm(self):
        """初始化LLM客户端（用于增强推理）"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                llm_cfg = cfg.get('llm', {})
            else:
                llm_cfg = {}

            api_key = llm_cfg.get('api_key', '')
            base_url = llm_cfg.get('base_url', 'https://api.deepseek.com/v1')
            model = llm_cfg.get('model', 'deepseek-chat')

            if not api_key:
                return

            from openai import OpenAI
            self._llm_client = OpenAI(api_key=api_key, base_url=base_url)
            self._llm_model = model
        except Exception as e:
            print(f"ChainReasoner LLM初始化失败: {e}")

    def reason_chain(self, stock_sector: str, stock_name: str = "",
                     mentioned_sectors: List[str] = None) -> ChainReasoningResult:
        """
        从板块出发，推理产业链位置和瓶颈节点

        Args:
            stock_sector: 股票所属板块
            stock_name: 股票名称
            mentioned_sectors: 博主提及的其他板块

        Returns:
            ChainReasoningResult
        """
        # 1. 映射板块到图谱节点
        graph_node = SECTOR_TO_GRAPH_NODE.get(stock_sector, stock_sector)

        # 2. 在图谱中推理
        result = self._reason_from_graph(graph_node, stock_name, mentioned_sectors)

        # 3. 如果图谱推理得分低，尝试LLM增强
        if result.supply_chain_score < 60 and self._llm_client:
            llm_result = self._reason_with_llm(stock_sector, stock_name, mentioned_sectors)
            if llm_result.supply_chain_score > result.supply_chain_score:
                result = llm_result

        return result

    def _reason_from_graph(self, graph_node: str, stock_name: str,
                           mentioned_sectors: List[str] = None) -> ChainReasoningResult:
        """基于硬编码图谱推理"""
        node_data = SUPPLY_CHAIN_GRAPH.get(graph_node)

        if not node_data:
            # 尝试模糊匹配
            for key, data in SUPPLY_CHAIN_GRAPH.items():
                if graph_node in key or key in graph_node:
                    node_data = data
                    graph_node = key
                    break

        if not node_data:
            return ChainReasoningResult(
                chain_path=graph_node,
                chain_position="unknown",
                supply_chain_score=50,
                reasoning=f"板块'{graph_node}'不在产业链图谱中"
            )

        # 构建产业链路径
        chain_path = self._build_chain_path(graph_node)

        # 判断是否为瓶颈节点
        bottlenecks = node_data.get("瓶颈", [])
        is_bottleneck = len(bottlenecks) > 0

        # 判断股票在产业链中的位置
        downstream = node_data.get("下游", [])
        chain_position = "瓶颈节点" if is_bottleneck else "中游"

        # 计算得分
        score = 50  # 基础分
        if is_bottleneck:
            score += 30  # 瓶颈节点加分（定价权）
        if bottlenecks and stock_name and any(b in stock_name for b in bottlenecks):
            score += 20  # 股票直接解决瓶颈加分

        # 检查博主提及的其他板块是否在上下游
        if mentioned_sectors:
            for sector in mentioned_sectors:
                other_node = SECTOR_TO_GRAPH_NODE.get(sector, sector)
                if other_node in downstream:
                    score += 5  # 下游有博主关注的板块加分

        # 检查是否有多条产业链交汇（交叉节点）
        intersection_count = 0
        for key, data in SUPPLY_CHAIN_GRAPH.items():
            if key != graph_node:
                for ds in data.get("下游", []):
                    if graph_node in ds or ds in graph_node:
                        intersection_count += 1
        if intersection_count >= 2:
            score += 10  # 多产业链交汇加分

        bottleneck_str = bottlenecks[0] if bottlenecks else ""

        return ChainReasoningResult(
            chain_path=chain_path,
            bottleneck_node=bottleneck_str,
            is_bottleneck=is_bottleneck,
            chain_position=chain_position,
            supply_chain_score=min(100, score),
            reasoning=f"产业链推理: {chain_path}, 瓶颈={bottleneck_str or '无'}, 位置={chain_position}"
        )

    def _build_chain_path(self, node: str) -> str:
        """构建从上游到当前节点的产业链路径"""
        path_parts = []
        current = node

        # 向上追溯最多3层
        visited = set()
        for _ in range(3):
            if current in visited:
                break
            visited.add(current)

            # 找谁指向当前节点（上游）
            upstream = None
            for key, data in SUPPLY_CHAIN_GRAPH.items():
                if current in data.get("下游", []) or current in data.get("瓶颈", []):
                    upstream = key
                    break

            if upstream:
                path_parts.insert(0, upstream)
                current = upstream
            else:
                break

        path_parts.append(node)
        return "→".join(path_parts)

    def _reason_with_llm(self, stock_sector: str, stock_name: str,
                         mentioned_sectors: List[str] = None) -> ChainReasoningResult:
        """LLM增强产业链推理"""
        sectors_str = ", ".join(mentioned_sectors or [stock_sector])

        prompt = f"""分析以下股票在产业链中的位置和定价权。

板块: {stock_sector}
股票: {stock_name}
博主同时关注: {sectors_str}

请推理：
1. 该股票所在产业链的上下游关系
2. 当前产业链的关键瓶颈在哪？
3. 该股票是否处于瓶颈节点（瓶颈节点具有定价权，能卡住下游脖子）
4. 产业链中还有哪些被低估的环节？

以JSON格式输出：
{{"chain_path": "上游→中游→下游路径", "bottleneck_node": "瓶颈节点名称", "is_bottleneck": true/false, "chain_position": "上游/中游/下游/瓶颈节点", "supply_chain_score": 0-100, "reasoning": "推理依据"}}"""

        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": "你是专业的A股产业链分析专家，擅长识别产业链瓶颈节点和定价权。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800,
                timeout=30.0,
            )
            raw = response.choices[0].message.content.strip()

            # 解析JSON
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                raw_json = re.sub(r'[\x00-\x1f]', '', json_match.group())
                data = json.loads(raw_json)
                return ChainReasoningResult(
                    chain_path=data.get("chain_path", stock_sector),
                    bottleneck_node=data.get("bottleneck_node", ""),
                    is_bottleneck=data.get("is_bottleneck", False),
                    chain_position=data.get("chain_position", "unknown"),
                    supply_chain_score=min(100, max(0, float(data.get("supply_chain_score", 50)))),
                    reasoning=data.get("reasoning", "")
                )
        except Exception as e:
            print(f"ChainReasoner LLM推理失败: {e}")

        return ChainReasoningResult(
            chain_path=stock_sector,
            chain_position="unknown",
            supply_chain_score=50,
            reasoning=f"LLM推理失败"
        )