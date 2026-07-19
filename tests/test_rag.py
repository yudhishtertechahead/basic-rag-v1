"""
tests/test_rag.py

Test Cases for RAG Pipeline:
1. Relevant question
2. Irrelevant question
3. Ambiguous question
4. Empty query
5. Multi-doc question
"""
import sys
import os
import logging

# Set up logging to see the debug logs
logging.basicConfig(level=logging.DEBUG)

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.rag_service import ask

def run_tests():
    test_cases = [
        {
            "name": "1. Relevant question",
            "query": "What is the dress code policy?",
        },
        {
            "name": "2. Irrelevant question",
            "query": "How do I bake a chocolate cake?",
        },
        {
            "name": "3. Ambiguous question",
            "query": "What is the policy?",
        },
        {
            "name": "4. Empty query",
            "query": "",
        },
        {
            "name": "5. Multi-doc question",
            "query": "Summarize the dress code policy and the POSH policy.",
        },
        {
            "name": "6. referal related",
            "query": "tell me how much incentive acc to Employee Referral Policy?",
        }
    ]

    for tc in test_cases:
        print(f"\n{'='*50}")
        print(f"Running Test: {tc['name']}")
        print(f"Query: '{tc['query']}'")
        print(f"{'-'*50}")
        try:
            # We can use the default LLM provider (google)
            response = ask(tc['query'])
            print(f"Answer: {response['answer']}")
            print(f"Sources used: {len(response['sources'])}")
            for idx, source in enumerate(response['sources']):
                print(f"  [{idx+1}] {source.get('source')}")
        except Exception as e:
            print(f"Error during test: {e}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    run_tests()
