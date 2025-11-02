import os
import requests
from tavily import TavilyClient
from utils.logger import logger


class WebHelper:
    tavily_client: TavilyClient = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def __init__(self) -> None:
        pass

    @classmethod
    def search_related_pages(
        cls, query: str, max_results: int = 2, topic: str = "general"
    ) -> list[str]:
        response = cls.tavily_client.search(
            query=query,
            topic=topic,
            max_results=max_results,
        )
        links = [item["url"] for item in response["results"]]
        return links

    @classmethod
    def extract_page_content(cls, url: str) -> str:
        parse_url = f"{os.getenv('JINA_BASE_URL')}{url}"
        try:
            resp = requests.get(parse_url, timeout=50)
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as http_err:
            status_code = (
                http_err.response.status_code if http_err.response else "unknown"
            )
            logger.error(
                f"使用 Jina 获取网页文本失败, HTTP Code {status_code}: {http_err}"
            )
        except Exception as err:
            logger.error(f"使用 Jina 获取网页文本失败, 错误信息: {err}")
        return ""
