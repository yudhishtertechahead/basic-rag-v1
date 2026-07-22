# RAG Pipeline Test Results

| Test Case | Query | Retrieval Time | LLM Gen Time | Total Time | Answer (Snippet) |
|---|---|---|---|---|---|
| 1. Relevant question | What is the dress code policy? | 9.016s | 1.439s | 10.456s | The dress code policy at TechAhead is designed to project a profession... |
| 2. Irrelevant question | How do I bake a chocolate cake? | 0.409s | 0.397s | 0.806s | I'm happy to help you with anything, but I'm afraid baking a chocolate... |
| 3. Ambiguous question | What is the policy? | 0.411s | 0.511s | 0.922s | Hello! It's great to hear from you. Based on the provided context, the... |
| 4. Empty query |  | 0.500s | 0.413s | 0.913s | Hello! It's great to hear from you. I'm Aria, TechAhead's HR Assistant... |
| 5. Multi-doc question | Summarize the dress code policy and the POSH policy. | 0.526s | 0.359s | 0.886s | I'd be happy to help you with that. It's a lovely day today, isn't it?... |
| 6. referal related | tell me how much incentive acc to Employee Referral Policy? | 0.412s | 0.383s | 0.795s | As per the Referral Incentive Policy, the incentive amounts are as fol... |
