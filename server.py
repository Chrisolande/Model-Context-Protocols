# server.py
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import aiofiles
from mcp.server.fastmcp import FastMCP
import base64
import fitz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a tool server
mcp = FastMCP("Multiple tool sever")

# @dataclass
@dataclass
class FileAnalysis:
    file_path: str
    line_count: int
    word_count: int
    char_count: int
    file_size: int
    analyzed_at: str = None

    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now().isoformat()

analysis_cache: Dict[str, FileAnalysis] = {}

@mcp.tool()
def statistics(numbers: list[float]) -> Dict[str, float]:
    """Perform basic statistics

    Args:
        numbers (list[float]): The list of numbers to return their statistics

    Returns:
        Dict[str, float]: A dictionary with the statistics
    """
    if not numbers:
        logger.error("Please enter a valid number")
        return {"error": "Empty list provided"}

    try:
        import statistics as stats
        result = {
            "length": len(numbers),
            "sum": sum(numbers),
            "minimum": min(numbers),
            "maximum": max(numbers),
            "range": max(numbers) - min(numbers),
            "mean": stats.mean(numbers),
            "median": stats.median(numbers)
        }

        if len(numbers) > 1:
            result["stdev"] = stats.stdev(numbers)
            result["variance"] = stats.variance(numbers)

        logger.info(f"Statistics calculated for {len(numbers)} numbers")
        return result

    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return {"error": str(e)}

@mcp.tool()
def get_system_info() -> Dict[str, Any]:
    """Get basic system information"""
    import platform
    import psutil

    try:
        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_usage_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
        }

        logger.info("System information retrieved")
        return {"success": True, "info": info}

    except Exception as e:
        logger.error(f"System info error: {e}")
        return {"error": str(e)}




# File Analysis Tools
@mcp.tool()
async def analyze_file(file_path: str) -> Dict[str, Any]:
    """Analyze a text file and provide statistics"""
    try:
        path = Path(file_path)
        if not path.exists():
            print("The file does not exist")

        if not path.is_file():
            print("The file path is not a valid file!")

        if file_path in analysis_cache:
            cached = analysis_cache[file_path]
            logger.info(f"Returning cached analysis for {file_path}")
            return {
                "success": True,
                "analysis": asdict(cached),
                "from_cache": True
            }

        # Read and analyze the file
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()

        embedded_content = base64.b64encode(content).decode("utf-8")

        # Extract text with fitz
        doc = fitz.open(stream = content, filetype = "pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        lines = full_text.splitlines()
        words = full_text.split()
        chars = len(full_text)

        analysis = FileAnalysis(
            file_path=file_path,
            line_count=len(lines),
            word_count=len(words),
            char_count=chars,
            file_size=path.stat().st_size
        )

        # Cache the analysis
        analysis_cache[file_path] = analysis

        logger.info(f"Analyzed file: {file_path}")
        return {
            "success": True,
            "analysis": asdict(analysis),
            "from_cache": False
        }

    except Exception as e:
        logger.error(f"File analysis error: {e}")
        return {"error": str(e)}

@mcp.tool()
def clear_analysis_cache() -> Dict[str, Any]:
    """Clear the file analysis cache"""
    cache_size = len(analysis_cache)
    analysis_cache.clear()
    logger.info(f"Cleared analysis cache ({cache_size} items)")

    return {
        "success": True,
        "message": f"Cleared {cache_size} cached analyses"
    }