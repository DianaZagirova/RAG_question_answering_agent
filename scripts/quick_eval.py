#!/usr/bin/env python3
"""
Quick RAG strategy comparison - tests 3 papers, 3 key questions.
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag_system import ScientificRAG
from src.core.llm_integration import AzureOpenAIClient, CompleteRAGSystem
from dotenv import load_dotenv

load_dotenv()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('CUDA_DEVICE', '3')


def test_paper(rag, llm, doi, ground_truth, use_multi_query, strategy_name):
    """Test one paper with one strategy."""
    complete_rag = CompleteRAGSystem(rag, llm, 10, use_multi_query)
    
    # Test only 3 key questions
    test_questions = {
        'aging_biomarker': ('Does it suggest an aging biomarker?', ['Yes, quantitatively shown', 'Yes, but not shown', 'No']),
        'molecular_mechanism_of_aging': ('Does it suggest a molecular mechanism of aging?', ['Yes', 'No']),
        'longevity_intervention_to_test': ('Does it suggest a longevity intervention to test?', ['Yes', 'No'])
    }
    
    results = {'correct': 0, 'total': 0}
    
    for key, (question, options) in test_questions.items():
        if key not in ground_truth:
            continue
        
        try:
            answer = complete_rag.answer_question(question, 10, 0.2, 300, True, None, options, doi)
            predicted = answer.get('answer', '').strip()
            expected = ground_truth[key].strip()
            
            is_correct = (predicted == expected or expected in predicted or predicted in expected)
            if is_correct:
                results['correct'] += 1
            
            results['total'] += 1
            print(f"  {key}: {'✓' if is_correct else '✗'} (expected: {expected}, got: {predicted})")
        except Exception as e:
            print(f"  {key}: ERROR - {str(e)[:50]}")
            results['total'] += 1
    
    return results


def main():
    print("\n" + "="*70)
    print("Quick RAG Strategy Comparison")
    print("="*70)
    
    # Load validation data
    with open('data/qa_validation_set_extended.json') as f:
        validation = json.load(f)
    
    # Select 3 papers
    test_dois = [
        '10.1089/ars.2012.5111',
        '10.1152/physrev.1998.78.2.547',
        '10.3389/fcell.2020.575645'
    ]
    
    test_papers = [p for p in validation if p['doi'] in test_dois]
    
    print(f"\nTesting {len(test_papers)} papers with 3 questions each")
    print("Strategies: Enhanced (default) vs Multi-Query\n")
    
    # Initialize
    rag = ScientificRAG()
    llm = AzureOpenAIClient(model='gpt-4.1')
    
    # Map questions
    q_map = {
        "Q1: Does the paper suggest an aging biomarker (i.e. measurable entity reflecting the pace of aging or organism health state  associated with mortality or age-related conditions)?": "aging_biomarker",
        "Q2: Does the paper suggest a molecular mechanism of aging?": "molecular_mechanism_of_aging",
        "Q3: Does the paper suggest a longevity intervention to test?": "longevity_intervention_to_test"
    }
    
    strategies = [
        ('Enhanced (default)', False),
        ('Multi-Query', True)
    ]
    
    all_results = {}
    
    for strategy_name, use_multi in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*70}")
        
        strategy_results = {'correct': 0, 'total': 0}
        
        for paper in test_papers:
            doi = paper['doi']
            ground_truth = {q_map[k]: v for k, v in paper.items() if k in q_map}
            
            print(f"\n{doi}:")
            result = test_paper(rag, llm, doi, ground_truth, use_multi, strategy_name)
            
            strategy_results['correct'] += result['correct']
            strategy_results['total'] += result['total']
        
        accuracy = strategy_results['correct'] / strategy_results['total'] if strategy_results['total'] > 0 else 0
        all_results[strategy_name] = {'accuracy': accuracy, **strategy_results}
        
        print(f"\n{strategy_name} Total: {strategy_results['correct']}/{strategy_results['total']} = {accuracy:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, res in all_results.items():
        print(f"{name:30s}: {res['correct']}/{res['total']} = {res['accuracy']:.1%}")
    
    # Recommendation
    best = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Strategy: {best[0]} ({best[1]['accuracy']:.1%} accuracy)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
