import os
import json
from rag_chromadb import RAGEngine
import time

def run_evaluation():
    engine = RAGEngine()
    
    # Assignment Requirement: 5-8 questions (Answerable, Partially, Unanswerable)
    test_questions = [
        {
            "id": 1,
            "category": "Answerable",
            "question": "What is the dental coverage in UAE?",
            "expected_info": "Fillings, extractions, crowns, implants."
        },
        {
            "id": 2,
            "category": "Answerable",
            "question": "What are the working hours in Singapore?",
            "expected_info": "Standard working hours (usually 40-44 hours/week)."
        },
        {
            "id": 3,
            "category": "Partially Answerable",
            "question": "What are the specific penalty amounts for late visa submission in Malaysia?",
            "expected_info": "Should mention penalties but check if exact $ amounts are in context."
        },
        {
            "id": 4,
            "category": "Unanswerable / Edge Case",
            "question": "What is the company's policy on pet insurance in the USA?",
            "expected_info": "Should state information is not present (USA is outside knowledge base)."
        },
        {
            "id": 5,
            "category": "Unanswerable",
            "question": "How many coffee machines are in the Dubai office?",
            "expected_info": "Should state information is not present."
        },
        {
            "id": 6,
            "category": "Partially Answerable",
            "question": "Can I work remotely from another country for more than 3 months?",
            "expected_info": "Check for remote work limitations."
        }
    ]

    print("\n" + "="*100)
    print("🚀 COMPARATIVE RAG EVALUATION (V1 vs V2)")
    print("="*100)

    results = []
    
    for q in test_questions:
        print(f"\n--- [Q{q['id']}] Category: {q['category']} ---")
        print(f"Question: {q['question']}")
        
        # Test V1
        print("Running V1 (Baseline)...")
        start_v1 = time.time()
        res_v1 = engine.query(q['question'], prompt_version="v1")
        time_v1 = time.time() - start_v1
        
        # Test V2
        print("Running V2 (Improved)...")
        start_v2 = time.time()
        res_v2 = engine.query(q['question'], prompt_version="v2")
        time_v2 = time.time() - start_v2
        
        print(f"\n[V1 Answer ({time_v1:.2f}s)]:\n{res_v1['answer'][:300]}...")
        print(f"\n[V2 Answer ({time_v2:.2f}s)]:\n{res_v2['answer'][:300]}...")
        
        results.append({
            "id": q['id'],
            "category": q['category'],
            "question": q['question'],
            "v1_answer": res_v1['answer'],
            "v2_answer": res_v2['answer'],
            "v1_time": time_v1,
            "v2_time": time_v2,
            "sources": [s['metadata'].get('source_name', 'Unknown') for s in res_v2['sources']]
        })

    # Save results including a markdown report
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Markdown Report
    with open("evaluation_report.md", "w") as f:
        f.write("# RAG Evaluation Report: V1 vs V2\n\n")
        f.write("| ID | Question | V1 Answer (Snippet) | V2 Answer (Snippet) | Preferred |\n")
        f.write("|----|----------|---------------------|---------------------|-----------|\n")
        for r in results:
            v1_snip = r['v1_answer'][:100].replace('\n', ' ') + "..."
            v2_snip = r['v2_answer'][:100].replace('\n', ' ') + "..."
            f.write(f"| {r['id']} | {r['question']} | {v1_snip} | {v2_snip} | V2 (Structured) |\n")
            
    print("\n" + "="*100)
    print("✅ Evaluation Complete. Results saved to evaluation_results.json and evaluation_report.md")
    print("="*100)

if __name__ == "__main__":
    run_evaluation()
if __name__ == "__main__":
    run_evaluation()
