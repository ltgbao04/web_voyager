# Web Voyager: Visual Web-Browsing Agent with LangGraph

## 🚀 Overview
This project implements a visual web-browsing agent using [LangGraph](https://github.com/langchain-ai/langgraph), OpenAI GPT-4o, and Playwright. The agent is capable of navigating websites via screenshots and executing actions like click, type, scroll, etc.

---

## 📁 Project Structure
```
web-voyager/
├── .env                    # API keys and configuration 
├── mark_page.js            # JavaScript for annotating web pages with clickable boxes
├── web_voyager.py          # LangGraph implementation and state logic
├── requirements.txt        # Python dependencies
└── README.md           
```

---

## ⚙️ Environment Setup 

### 1. Clone the Repository
```bash
git clone https://github.com/ltgbao04/web-voyager.git
cd web-voyager
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

## 🏃 Run the Agent
Run the main script to start the agent:
```bash
python web_voyager.py
```

**_NOTE:_**  If a CAPTCHA is encountered during browsing, please solve it manually in the browser window.
## 🔐 Automatic Login
Run `login_agent.py` to log in to any website automatically:
```bash
python login_agent.py --url <login_page> --username <user> --password <pass>
```

