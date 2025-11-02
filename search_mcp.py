from dotenv import load_dotenv

load_dotenv()

import os
import instructor
from openai import OpenAI
from pydantic import BaseModel
from utils.logger import logger
from utils.web_helper import WebHelper
from mcp.server.fastmcp import FastMCP

client = instructor.from_openai(
    OpenAI(
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
)


def expand_query(usr_query: str) -> list[str]:
    class ResponseModel(BaseModel):
        queries: list[str]

    return client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL"),
        messages=[
            {
                "role": "system",
                "content": "You are an expert research assistant. Given the user's query, generate up to 2 distinct, precise search queries that would help gather comprehensive information on the topic.",
            },
            {
                "role": "user",
                "content": usr_query,
            },
        ],
        response_model=ResponseModel,
    ).queries


def expand_query_with_context(
    usr_query: str, existed_queries_set: set[str], context: str
) -> list[str]:
    class ResponseModel(BaseModel):
        queries: list[str]

    return client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL"),
        messages=[
            {
                "role": "system",
                "content": "You are an analytical research assistant. Based on the original query, the search queries performed so far, and the extracted contexts from webpages, determine if further research is needed. If you believe no further research is needed, respond with an empty list.",
            },
            {
                "role": "user",
                "content": f"# User Query\n{usr_query}\n#Previous Search Queries\n{existed_queries_set}\n# Relevant Content\n{"\n".join(context)}",
            },
        ],
        response_model=ResponseModel,
    ).queries


def access_page_relevance(usr_query: str, page_content: str) -> bool:
    class ResponseModel(BaseModel):
        is_relevant: bool

    return client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL"),
        messages=[
            {
                "role": "system",
                "content": "You are a critical research evaluator. Given the user's query and the content of a webpage, determine if the webpage contains information relevant and useful for addressing the query.",
            },
            {
                "role": "user",
                "content": f"# User Query\n{usr_query}\n# Page Content (First 20000 characters)\n{page_content[:20000]}",
            },
        ],
        response_model=ResponseModel,
    ).is_relevant


def extract_relevant_page_content(
    usr_query: str, derived_query: str, page_content: str
) -> str:
    class ResponseModel(BaseModel):
        content: str

    return client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL"),
        messages=[
            {
                "role": "system",
                "content": "You are an expert information extractor. Given the user's query, the derived query that led to this page, and the webpage content, extract all pieces of information that are relevant to answering the user's query. Return only the relevant context as plain text without commentary",
            },
            {
                "role": "user",
                "content": f"# User Query\n{usr_query}\n# Derived Query\n{derived_query}\n# Page Content (First 20000 characters)\n{page_content[:20000]}",
            },
        ],
        response_model=ResponseModel,
    ).content


def fetch_relevant_page_content(
    usr_query: str, derived_query: str, url: str
) -> str | None:
    logger.info(f"获取网页 `{url}` 内容")
    page_content = WebHelper.extract_page_content(url)

    is_relevant = access_page_relevance(usr_query, page_content)
    logger.info(f"评估网页与用户查询 `{usr_query}` 的相关性: {is_relevant}")

    if is_relevant:
        relevant_page_content = extract_relevant_page_content(
            usr_query, derived_query, page_content
        )
        logger.debug(
            f"预览网页 `{url}` 信息(前 200 个字符): {relevant_page_content[:200]}"
        )
        return relevant_page_content
    return None


mcp = FastMCP("meow-search")


@mcp.tool()
def search(usr_query: str) -> str:
    aggregated_result = []

    iteration_limit = 1
    logger.info(f"原始用户查询: {usr_query}")
    logger.info(f"查询迭代限制: {iteration_limit}")
    expanded_queries = expand_query(usr_query)
    all_queries_set = set(expanded_queries).add(usr_query)

    for i in range(iteration_limit):
        iteration_result = []
        logger.info(f"第 [{i+1}] 轮检索")
        logger.info(f"本轮查询内容: {expanded_queries}")

        link2query = {}
        for query in expanded_queries:
            related_links = WebHelper.search_related_pages(query, max_results=1)
            for link in related_links:
                if link not in link2query:
                    link2query[link] = query

        logger.info(f"查询到 {len(link2query)} 个相关网站")

        for url, derived_query in link2query.items():
            _r = fetch_relevant_page_content(usr_query, derived_query, url)
            if _r:
                iteration_result.append(_r)
        aggregated_result.extend(iteration_result)

        new_queries = expand_query_with_context(
            usr_query, all_queries_set, aggregated_result
        )

        if len(new_queries) == 0:
            logger.info("当前检索结果已满足用户诉求, 无需进一步检索")
            break
        else:
            expanded_queries = new_queries
            if i == iteration_limit - 1:
                logger.info(f"当前检索结果无法满足用户需求, 但已达到检索最大迭代次数.")
            else:
                logger.info(
                    f"当前检索结果无法满足用户需求, 需继续检索以下内容: {new_queries}"
                )

    return "\n\n".join(aggregated_result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
