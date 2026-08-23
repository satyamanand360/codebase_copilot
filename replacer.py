import os

replacements = [
    ("from langchain_anthropic import ChatAnthropic", "from langchain_google_genai import ChatGoogleGenerativeAI"),
    ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY"),
    ("your_anthropic_api_key_here", "your_google_api_key_here"),
    ("ChatAnthropic(model=\"claude-3-5-sonnet-20240620\", temperature=0)", "ChatGoogleGenerativeAI(model=\"gemini-1.5-pro\", temperature=0)"),
    ("ChatAnthropic(model=\"claude-3-haiku-20240307\", temperature=0)", "ChatGoogleGenerativeAI(model=\"gemini-1.5-flash\", temperature=0)"),
    ("anthropic_key", "google_key")
]

files_to_update = [
    'agents/qa_agent.py',
    'agents/review_agent.py',
    'agents/test_agent.py',
    'agents/docs_agent.py',
    'agents/supervisor.py',
    '.env.example',
    'eval/run_eval.py',
    '.github/workflows/pr-review.yml'
]

for filepath in files_to_update:
    if not os.path.exists(filepath):
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Replacement complete.")
