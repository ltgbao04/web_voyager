# Web Voyager: Visual Web-Browsing Agent with LangGraph

## 🚀 Overview
This project implements two types of visual web-browsing agents using [LangGraph](https://github.com/langchain-ai/langgraph), OpenAI GPT-4o, and Playwright:

- `web_voyager.py`: Agent interacts with web pages using HTML and CSS structure (DOM-based).
- `visual_login_agent.py`: Agent interacts with web pages using screenshots (vision-based, suitable for visual login and CAPTCHA scenarios).

---

## 📁 Project Structure
```
langgraph_playwright/
├── .env                    # API keys and configuration 
├── mark_page.js            # JavaScript for annotating web pages with clickable boxes
├── web_voyager.py          # DOM-based agent (HTML/CSS logic)
├── visual_login_agent.py   # Screenshot-based agent (vision logic)
├── requirements.txt        # Python dependencies
└── README.md           
```

---

## ⚙️ Environment Setup 

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd langgraph_playwright
```

### 2. Create Conda Environment
```bash
conda create -n webvoyager python=3.11 -y
conda activate webvoyager
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# Additionally install and setup Playwright
playwright install
```

### 4. Create `.env` File
Create a file named `.env` in the root directory with the following content:
```env
OPENAI_API_KEY= ""
LANGSMITH_API_KEY= ""  
```

---

## 🏃 Run the Agents

### 1. DOM-based Agent (HTML/CSS)
Run the main script to start the DOM-based agent:
```bash
python web_voyager.py
```

### 2. Screenshot-based Agent (Vision)
Run the vision-based agent for visual login and automation:
```bash
python visual_login_agent.py
```

**_NOTE:_**  
- Prompt must contain the target URL and, if login is required, must include both username and password (e.g., `{username: your_user password: your_pass}`).
- If a CAPTCHA is encountered during browsing, please solve it manually in the browser window.

Ex: Login into Kaggle.com, try to sign in with this account {username: ltgbao04@gmail.com, password: password}