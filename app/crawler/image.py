"""
图片下载模块
"""
import asyncio
import hashlib
from pathlib import Path
from typing import Optional, List
import httpx
from ..config import get_config


async def get_image_dir() -> Path:
    """获取图片存储目录"""
    config = get_config()
    image_dir = Path(config.storage.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def get_image_filename(url: str, index: int = 0) -> str:
    """根据URL生成图片文件名"""
    # 使用URL的hash作为文件名的一部分
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    # 尝试从URL获取扩展名
    ext = '.jpg'
    if '.png' in url.lower():
        ext = '.png'
    elif '.gif' in url.lower():
        ext = '.gif'
    elif '.jpeg' in url.lower() or '.jpg' in url.lower():
        ext = '.jpg'

    return f"{url_hash}_{index}{ext}"


async def download_image(client: httpx.AsyncClient, url: str, save_path: Path) -> bool:
    """下载单张图片"""
    try:
        config = get_config()
        headers = {
            'User-Agent': config.crawler.user_agent,
            'Referer': 'https://m.weibo.cn/'
        }

        response = await client.get(url, headers=headers, timeout=30, follow_redirects=True)
        if response.status_code == 200:
            save_path.write_bytes(response.content)
            return True
        return False
    except Exception as e:
        print(f"下载图片失败 {url}: {e}")
        return False


async def download_images(urls: List[str], weibo_id: str) -> List[str]:
    """批量下载图片，返回本地路径列表"""
    if not urls:
        return []

    image_dir = await get_image_dir()
    # 为每个微博创建子目录
    weibo_dir = image_dir / weibo_id
    weibo_dir.mkdir(parents=True, exist_ok=True)

    local_paths = []
    config = get_config()

    async with httpx.AsyncClient() as client:
        tasks = []
        for index, url in enumerate(urls):
            filename = get_image_filename(url, index)
            save_path = weibo_dir / filename

            # 如果文件已存在，跳过下载
            if save_path.exists():
                local_paths.append(str(save_path))
                continue

            tasks.append((url, save_path, index))

        # 并发下载
        for url, save_path, index in tasks:
            success = await download_image(client, url, save_path)
            if success:
                local_paths.append(str(save_path))
            else:
                # 下载失败，记录原始URL
                local_paths.append(url)

            # 添加延迟避免请求过快
            await asyncio.sleep(0.5)

    return local_paths
