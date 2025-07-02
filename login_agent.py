import argparse
import asyncio
from playwright.async_api import async_playwright
from web_voyager import call_agent

async def auto_login(url: str, username: str, password: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(url)
        prompt = (
            f"Login to this website using username '{username}' and password '{password}'. "
            "After logging in, respond with 'ANSWER: done'."
        )
        answer = await call_agent(prompt, page)
        print("Agent answer:", answer)
        await browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic login using Web Voyager agent")
    parser.add_argument("--url", required=True, help="Login page URL")
    parser.add_argument("--username", required=True, help="Username")
    parser.add_argument("--password", required=True, help="Password")
    args = parser.parse_args()
    asyncio.run(auto_login(args.url, args.username, args.password))
