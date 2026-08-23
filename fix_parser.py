import os

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
        
    if 'StrOutputParser' not in content:
        content = 'from langchain_core.output_parsers import StrOutputParser\n' + content
        
    content = content.replace(' | self.llm\n', ' | self.llm | StrOutputParser()\n')
    content = content.replace(' | self.verify_llm\n', ' | self.verify_llm | StrOutputParser()\n')
    
    # Replace .content calls on invoke
    content = content.replace('}).content', '})')
    content = content.replace('}).content.strip', '}).strip')
    content = content.replace('question}).content', 'question})')
    content = content.replace('content}).content', 'content})')
    content = content.replace('output"}).content', 'output"})')
    content = content.replace('target}).content', 'target})')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print('Added StrOutputParser to all agents.')
