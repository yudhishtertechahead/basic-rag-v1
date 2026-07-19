import re

log_file = r'C:\Users\yudhi\.gemini\antigravity-ide\brain\93e4c6a0-32db-40d2-9fd9-2bf17526f50e\.system_generated\tasks\task-80.log'

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

tests = text.split('Running Test: ')[1:]

md = '# RAG Pipeline Test Results\n\n'
md += '| Test Case | Query | Retrieval Time | LLM Gen Time | Total Time | Answer (Snippet) |\n'
md += '|---|---|---|---|---|---|\n'

for t in tests:
    try:
        name = t.split('\n')[0].strip()
        query_match = re.search(r"Query: '(.*?)'", t)
        query = query_match.group(1) if query_match else "N/A"
        
        ret_time = re.search(r'\[TIMING\] Step 1 - Retrieval: ([\d\.]+)s', t)
        ret_time = ret_time.group(1) if ret_time else "N/A"
        
        llm_time = re.search(r'\[TIMING\] Step 3 - LLM generation.*?: ([\d\.]+)s', t)
        llm_time = llm_time.group(1) if llm_time else "N/A"
        
        tot_time = re.search(r'\[TIMING\] Total end-to-end: ([\d\.]+)s', t)
        tot_time = tot_time.group(1) if tot_time else "N/A"
        
        ans_match = re.search(r'Answer: (.*?)(?=\nSources used:|\n=)', t, re.DOTALL)
        answer = ans_match.group(1).replace('\n', ' ').strip() if ans_match else 'Error/No Answer'
        answer_snip = (answer[:70] + '...') if len(answer) > 70 else answer
        
        md += f'| {name} | {query} | {ret_time}s | {llm_time}s | {tot_time}s | {answer_snip} |\n'
    except Exception as e:
        print(f'Failed parsing a test: {e}')

with open('test_results.md', 'w', encoding='utf-8') as out:
    out.write(md)

print('Markdown file created successfully.')
