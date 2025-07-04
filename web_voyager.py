import os
import re
import asyncio
import base64
from typing import List, Optional
from typing_extensions import TypedDict
from playwright.async_api import Page
from dotenv import load_dotenv
from typing import Optional, Dict, List
from IPython import display
from playwright.async_api import async_playwright
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain as chain_decorator, RunnablePassthrough
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# === TypedDict Definitions for Structured Data ===
class BBox(TypedDict):
    x: float         # x-coordinate of the bounding box on screen
    y: float         # y-coordinate of the bounding box on screen
    text: str        # text content of the element
    type: str        # element type (e.g., 'button', 'link')
    ariaLabel: str   # accessible label for the element

class Prediction(TypedDict):
    action: str                 # the tool or action to perform
    args: Optional[List[str]]   # optional list of arguments for the action

class AgentState(TypedDict):
    page: Page                  # Playwright Page object representing the browser context
    input: str                  # user's input prompt
    img: str                    # screenshot image encoded as base64
    bboxes: List[BBox]          # list of detected bounding boxes on the page
    prediction: Prediction      # tool prediction from LLM
    scratchpad: List[BaseMessage]  # history of observations and actions
    observation: str            # last observation result of a tool call

import asyncio
import platform

# === Tool Implementations ===
async def click(state: AgentState):
    """
    Simulates clicking on an element using its bounding box index.
    """
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = int(click_args[0])
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"

async def type_text(state: AgentState):
    """
    Types text into an input element specified by bounding box index, then submits.
    """
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return f"Failed to type in element from bounding box labeled as number {type_args}"
    bbox_id = int(type_args[0])
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    # Focus the input box by clicking, then clear existing content
    await page.mouse.click(x, y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    # Type new content and press Enter
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"

async def scroll(state: AgentState):
    """
    Scrolls either the entire window or a specific element up/down.
    """
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."
    target, direction = scroll_args
    # If WINDOW, scroll the page viewport; otherwise scroll a specific element
    if target.upper() == "WINDOW":
        scroll_amount = 500
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)
    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

async def wait(state: AgentState):
    """
    Pauses execution for a fixed duration (5 seconds).
    """
    await asyncio.sleep(5)
    return "Waited for 5s."

async def go_back(state: AgentState):
    """
    Navigates back in browser history.
    """
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."

async def to_google(state: AgentState):
    """
    Navigates the browser to Google homepage.
    """
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."

# === CAPTCHA Handling ===
async def pause_for_captcha(state: AgentState):
    """
    Alerts user to solve CAPTCHA manually by displaying screenshot.
    """
    screenshot = base64.b64decode(state['img'])
    print("🚨 CAPTCHA detected! Vui lòng giải CAPTCHA trên cửa sổ trình duyệt.")
    input("Sau khi giải xong, nhấn Enter để tiếp tục...")
    return "User solved captcha manually"

# === Page Marking Logic ===
# Reads JavaScript to highlight clickable regions on the page
with open("mark_page.js") as f:
    mark_page_script = f.read()

@chain_decorator
async def mark_page(page):
    """
    Injects JS to mark page elements and capture screenshot + bounding boxes.
    """
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")  # returns list of BBox dicts
            break
        except Exception:
            await asyncio.sleep(3)
    screenshot_bytes = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {"img": base64.b64encode(screenshot_bytes).decode(), "bboxes": bboxes}

async def annotate(state):
    """
    Runs mark_page and merges its output into the agent state.
    """
    marked = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked}

def format_descriptions(state):
    """
    Formats the list of bounding boxes into human-readable labels for the LLM.
    """
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    return {**state, "bbox_descriptions": "\nValid Bounding Boxes:\n" + "\n".join(labels)}


# def parse(text: str) -> dict:
#     """
#     Parses the LLM's output to extract an action and args.
#     Expects lines ending with 'Action: <ACTION> <args>'.
#     """
#     action_prefix = "Action: "
#     if not text.strip().split("\n")[-1].startswith(action_prefix):
#         return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
#     action_block = text.strip().split("\n")[-1]
#     action_str = action_block[len(action_prefix):]
#     parts = action_str.split(" ", 1)
#     action = parts[0].strip()
#     args = parts[1] if len(parts) > 1 else None
#     if args is not None:
#         args = [a.strip().strip("[]") for a in args.split(";")]
#     return {"action": action, "args": args}

def parse(text: str) -> Dict[str, Optional[List[str]]]:
    """
    Parses the LLM's output to extract an action and args.
    Hỗ trợ cả hai dạng:
      - "Action: <ACTION> <args>"
      - "ANSWER: <your answer>" / "ANSWER;: <your answer>"
    """
    last_line = text.strip().split("\n")[-1]

    # 1) Chuẩn "Action: ACTION args"
    action_prefix = "Action: "
    if last_line.startswith(action_prefix):
        action_block = last_line[len(action_prefix):]
    else:
        # 2) Thử match "ANSWER:" hoặc "ANSWER;:"
        m = re.match(r'^(ANSWER)[;:]?\s*(.*)', last_line, re.IGNORECASE)
        if m:
            action = m.group(1).upper()
            arg_str = m.group(2).strip()
            args = [arg_str] if arg_str else []
            return {"action": action, "args": args}

        # Nếu vẫn không khớp thì retry
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}

    # Parse tiếp phần Action
    parts = action_block.split(" ", 1)
    action = parts[0].strip()
    args = parts[1] if len(parts) > 1 else None
    if args:
        args = [a.strip().strip("[]") for a in args.split(";")]
    return {"action": action, "args": args}

# === LLM & Agent Graph Setup ===
# Initialize GPT-4o via langchain-openai integration
dotenv_api = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", max_tokens=4096, api_key=dotenv_api)

# Compose the pipeline: annotate -> format descriptions -> LLM -> parse
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | hub.pull("wfh/web-voyager") | llm | StrOutputParser() | parse
)

import re
from langchain_core.messages import SystemMessage

def update_scratchpad(state: AgentState):
    """
    Records observations/results in the scratchpad for context in subsequent steps.
    """
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"
    return {**state, "scratchpad": [SystemMessage(content=txt)]}

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

# Build the state graph for tool selection and execution
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent)
graph_builder.add_edge(START, "agent")
graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

# Register all tool nodes in the graph
tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
    "PauseForCaptcha": pause_for_captcha
}

for name, fn in tools.items():
    graph_builder.add_node(name, RunnableLambda(fn) | (lambda obs: {"observation": obs}))
    graph_builder.add_edge(name, "update_scratchpad")

# Decide next tool based on parsed action
def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "PauseForCaptcha":
        return "PauseForCaptcha"
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action

graph_builder.add_conditional_edges("agent", select_tool)
graph = graph_builder.compile()

async def call_agent(question: str, page, max_steps: int = 150):
    """
    Drives the agent through the browser to answer a user question.
    """
    event_stream = graph.astream(
        {"page": page, "input": question, "scratchpad": []},
        {"recursion_limit": max_steps},
    )
    final_answer = None
    steps = []
    async for event in event_stream:
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        args = pred.get("args")
        display.clear_output(wait=False)
        steps.append(f"{len(steps)+1}. {action}: {args}")
        print("\n".join(steps))
        display.display(display.Image(base64.b64decode(event["agent"]["img"])))
        if action == "ANSWER":
            final_answer = args[0]
            break
    return final_answer

async def main():
    async with async_playwright() as p:
        
        prompt = input("Enter your prompt (make sure to include the account if you want to login): ")

        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.google.com")
        #prompt = "Could you explain the WebVoyager paper (on arxiv)?"
        
        answer = await call_agent(prompt, page)
        print("Final Answer:", answer)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())