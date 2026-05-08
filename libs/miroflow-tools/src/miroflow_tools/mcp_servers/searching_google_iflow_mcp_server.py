# Copyright 2025 Miromind.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Google Search MCP Server using Serper API + iflow API for web scraping.

This is an alternative implementation that uses iflow API instead of Jina API
for web scraping, matching the same approach used to build the offline knowledge base.

This allows for ablation experiments comparing:
- Offline RAG (using pre-built knowledge base with Serper + iflow)
- Online search (using real-time Serper + iflow)
"""

import asyncio
import json
import os
import re
import sys

import aiohttp
import requests
from fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .utils import strip_markdown_links

# Serper API configuration (same as original)
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_BASE_URL = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev")

# iflow API configuration (replaces Jina)
IFLOW_API_URL = os.environ.get("IFLOW_API_URL", "https://apis.iflow.cn/v1/chat/webFetch")
IFLOW_API_KEY = os.environ.get("IFLOW_API_KEY", "")

# Google search result filtering environment variables
REMOVE_SNIPPETS = os.environ.get("REMOVE_SNIPPETS", "").lower() in ("true", "1", "yes")
REMOVE_KNOWLEDGE_GRAPH = os.environ.get("REMOVE_KNOWLEDGE_GRAPH", "").lower() in (
    "true",
    "1",
    "yes",
)
REMOVE_ANSWER_BOX = os.environ.get("REMOVE_ANSWER_BOX", "").lower() in (
    "true",
    "1",
    "yes",
)

# Content processing configuration (matching batch_fetch_urls_with_iflow.py)
MIN_CONTENT_LENGTH = 200
MAX_CONTENT_LENGTH = 7000  # 7k chars, ~1550 tokens

# Initialize FastMCP server
mcp = FastMCP("searching-google-iflow-mcp-server")


def clean_markdown_content(content: str) -> str:
    """Clean markdown content by removing images and normalizing whitespace."""
    if not content:
        return content
    # Remove image markdown
    content = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', content)
    # Normalize tabs and spaces
    content = re.sub(r'\t+', ' ', content)
    content = re.sub(r' +', ' ', content)
    # Normalize newlines
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    return content.strip()


def filter_google_search_result(result_content: str) -> str:
    """Filter google search result content based on environment variables.

    Args:
        result_content: The JSON string result from google search

    Returns:
        Filtered JSON string result
    """
    try:
        # Parse JSON
        data = json.loads(result_content)

        # Remove knowledgeGraph if requested
        if REMOVE_KNOWLEDGE_GRAPH and "knowledgeGraph" in data:
            del data["knowledgeGraph"]

        # Remove answerBox if requested
        if REMOVE_ANSWER_BOX and "answerBox" in data:
            del data["answerBox"]

        # Remove snippets if requested
        if REMOVE_SNIPPETS:
            # Remove snippets from organic results
            if "organic" in data:
                for item in data["organic"]:
                    if "snippet" in item:
                        del item["snippet"]

            # Remove snippets from peopleAlsoAsk
            if "peopleAlsoAsk" in data:
                for item in data["peopleAlsoAsk"]:
                    if "snippet" in item:
                        del item["snippet"]

        # Return filtered JSON
        return json.dumps(data, ensure_ascii=False, indent=2)

    except (json.JSONDecodeError, Exception):
        # If filtering fails, return original content
        return result_content


@mcp.tool()
async def google_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: str = None,
    num: int = 10,
    tbs: str = None,
    page: int = 1,
) -> str:
    """Perform google searches via Serper API and retrieve rich results.
    It is able to retrieve organic search results, people also ask, related searches, and knowledge graph.

    Args:
        q: Search query string.
        gl: Country context for search (e.g., 'us' for United States, 'cn' for China, 'uk' for United Kingdom). Influences regional results priority. Default is 'us'.
        hl: Google interface language (e.g., 'en' for English, 'zh' for Chinese, 'es' for Spanish). Affects snippet language preference. Default is 'en'.
        location: City-level location for search results (e.g., 'SoHo, New York, United States', 'California, United States').
        num: The number of results to return (default: 10).
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year).
        page: The page number of results to return (default: 1).

    Returns:
        The search results.
    """
    if SERPER_API_KEY == "":
        return (
            "[ERROR]: SERPER_API_KEY is not set, google_search tool is not available."
        )

    tool_name = "google_search"
    arguments = {
        "q": q,
        "gl": gl,
        "hl": hl,
        "num": num,
        "page": page,
        "autocorrect": False,
    }
    if location:
        arguments["location"] = location
    if tbs:
        arguments["tbs"] = tbs
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "miroflow_tools.mcp_servers.serper_mcp_server"],
        env={"SERPER_API_KEY": SERPER_API_KEY, "SERPER_BASE_URL": SERPER_BASE_URL},
    )
    result_content = ""

    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(
                    read, write, sampling_callback=None
                ) as session:
                    await session.initialize()
                    tool_result = await session.call_tool(
                        tool_name, arguments=arguments
                    )
                    result_content = (
                        tool_result.content[-1].text if tool_result.content else ""
                    )
                    assert (
                        result_content is not None and result_content.strip() != ""
                    ), "Empty result from google_search tool, please try again."
                    # Apply filtering based on environment variables
                    filtered_result = filter_google_search_result(result_content)
                    return filtered_result  # Success, exit retry loop
        except Exception as error:
            retry_count += 1
            if retry_count >= max_retries:
                return f"[ERROR]: google_search tool execution failed after {max_retries} attempts: {str(error)}"
            # Wait before retrying
            await asyncio.sleep(min(2**retry_count, 60))

    return "[ERROR]: Unknown error occurred in google_search tool, please try again."


@mcp.tool()
async def scrape_website(url: str) -> str:
    """This tool is used to scrape a website for its content using iflow API.
    Search engines are not supported by this tool. This tool can also be used to get 
    YouTube video non-visual information (however, it may be incomplete), such as 
    video subtitles, titles, descriptions, key moments, etc.

    Args:
        url: The URL of the website to scrape.
    Returns:
        The scraped website content.
    """
    # Validate URL format
    if not url or not url.startswith(("http://", "https://")):
        return f"Invalid URL: '{url}'. URL must start with http:// or https://"

    # Check for restricted domains
    if "huggingface.co/datasets" in url or "huggingface.co/spaces" in url:
        return "You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose."

    if IFLOW_API_KEY == "":
        return "[ERROR]: IFLOW_API_KEY is not set, scrape_website tool is not available."

    try:
        # Use iflow API to fetch URL content
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {IFLOW_API_KEY}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                IFLOW_API_URL, 
                headers=headers, 
                json={"url": url},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                result = await response.json()
                
                if not result.get("success"):
                    error_msg = result.get("message", "unknown_error")
                    return f"[ERROR]: iflow API error: {error_msg}"
                
                # Extract content from iflow response
                data = result.get("data", {}).get("outputs", {}).get("data", {}).get("data", [])
                if not data:
                    return f"[ERROR]: No content retrieved from URL: {url}"
                
                content = data[0].get("content", "")
                if not content:
                    return f"[ERROR]: Empty content from URL: {url}"
                
                # Clean and truncate content
                content = clean_markdown_content(content)
                content = strip_markdown_links(content)
                
                if len(content) > MAX_CONTENT_LENGTH:
                    content = content[:MAX_CONTENT_LENGTH] + "\n\n[Content truncated...]"
                
                if len(content) < MIN_CONTENT_LENGTH:
                    return f"[ERROR]: Content too short (less than {MIN_CONTENT_LENGTH} chars) from URL: {url}"
                
                return content

    except asyncio.TimeoutError:
        return f"[ERROR]: Timeout Error: Request timed out while scraping '{url}'. The website may be slow or unresponsive."

    except aiohttp.ClientError as e:
        return f"[ERROR]: Connection Error: Failed to connect to '{url}'. {str(e)}"

    except Exception as e:
        return f"[ERROR]: Unexpected Error: An unexpected error occurred while scraping '{url}': {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
