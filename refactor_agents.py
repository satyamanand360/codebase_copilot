import os
import re

files_to_update = [
    'agents/qa_agent.py',
    'agents/review_agent.py',
    'agents/test_agent.py',
    'agents/docs_agent.py',
    'agents/supervisor.py'
]

for filepath in files_to_update:
    if not os.path.exists(filepath): continue
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add import
    if 'from utils.llm_factory import get_llm' not in content:
        content = 'from utils.llm_factory import get_llm\n' + content

    # 2. Remove old Google GenAI import
    content = re.sub(r'from langchain_google_genai import ChatGoogleGenerativeAI\n', '', content)
    
    # 3. Replace initialization logic
    # Find the __init__ block and replace the api_key checking and llm instantiation
    # Because each file is slightly different, we'll use a regex to match the block
    
    pattern = re.compile(r'        # Initialize.*?(?=        self\.)', re.DOTALL)
    
    if 'supervisor.py' in filepath:
        # Supervisor
        content = re.sub(
            r'        api_key = os\.environ\.get\("GOOGLE_API_KEY"\).*?self\.llm = ChatGoogleGenerativeAI.*?temperature=0\)',
            '        self.llm = get_llm("fast")',
            content, flags=re.DOTALL
        )
    elif 'qa_agent.py' in filepath:
        # QA Agent (has verifier)
        content = re.sub(
            r'        # Initialize Anthropic LLM.*?self\.verifier_llm = ChatGoogleGenerativeAI.*?temperature=0\)',
            '        self.llm = get_llm("main")\n        self.verifier_llm = get_llm("fast")',
            content, flags=re.DOTALL
        )
        # Also remove the check in answer_question
        content = re.sub(r'        if not self\.llm:\n\s+return "Error: Cannot generate answer\..*?"\n\s+', '', content)
    else:
        # Review, Test, Docs
        content = re.sub(
            r'        # Initialize Anthropic LLM.*?self\.llm = ChatGoogleGenerativeAI.*?temperature=0\)',
            '        self.llm = get_llm("main")',
            content, flags=re.DOTALL
        )
        content = re.sub(r'        if not self\.llm:\n\s+return "Error: Cannot generate (review|test|docs).*?"\n\s+', '', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print('Agent refactoring complete.')
