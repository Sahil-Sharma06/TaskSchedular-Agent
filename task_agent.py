import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else genai.Client()

def read_tasks(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def summarize_tasks(tasks: str, model: str = MODEL) -> str:
    prompt = f"""
You are a smart task planning agent.
Given a list of tasks, categorize them into 3 priority buckets:
High Priority
Medium Priority
Low Priority

Tasks:
{tasks}

Return the response in this exact format (use the headers shown and list tasks under them):

High Priority:
task 1
task 2

Medium Priority:
task 1
task 2

Low Priority:
task 1
task 2
"""
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"[Error calling Gemini/GenAI SDK] {e}"

if __name__ == "__main__":
    TASK_FILE = "tasks.txt"
    if not os.path.exists(TASK_FILE):
        print(f"Create a {TASK_FILE} file with one task per line and re-run.")
    else:
        tasks_text = read_tasks(TASK_FILE)
        print("Calling Gemini to summarize tasks... (this may take a moment)")
        summary = summarize_tasks(tasks_text)
        print("Task Summary:")
        print("-" * 30)
        print(summary)
