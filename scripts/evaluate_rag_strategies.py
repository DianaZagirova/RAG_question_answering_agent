#!/usr/bin/env python3
"""
Evaluate different RAG strategies on validation set.
Compares: regular, enhanced, multi-query, and combinations.
"""
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag_system import ScientificRAG
from src.core.llm_integration import AzureOpenAIClient, CompleteRAGSystem
from dotenv import load_dotenv

load_dotenv()

# Set CUDA device
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    cuda_device = os.getenv('CUDA_DEVICE', '3')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device


def load_validation_set(file_path: str) -> List[Dict]:
    """Load validation set from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def map_validation_to_new_format(validation_entry: Dict) -> Dict:
    """Map old validation format to new question format."""
    # Load new question format
    with open('data/questions_part2.json', 'r') as f:
        questions = json.load(f)
    
    # Map old questions to new keys
    mapping = {
        "Q1: Does the paper suggest an aging biomarker (i.e. measurable entity reflecting the pace of aging or organism health state  associated with mortality or age-related conditions)?": "aging_biomarker",
        "Q2: Does the paper suggest a molecular mechanism of aging?": "molecular_mechanism_of_aging",
        "Q3: Does the paper suggest a longevity intervention to test?": "longevity_intervention_to_test",
        "Q4: Does the paper claim that aging cannot be reversed? (Yes / No)": "aging_cannot_be_reversed",
        "Q5: Does the paper suggest a biomarker that explains differences in maximum lifespan between species? (Yes / No)": "cross_species_longevity_biomarker",
        "Q6: Does the paper explain why the naked mole rat can live 40+ years despite its small size?": "naked_mole_rat_lifespan_explanation",
        "Q7: Does the paper explain why birds live much longer than mammals on average?": "birds_lifespan_explanation",
        "Q8: Does the paper explain why large animals live longer than small ones?": "large_animals_lifespan_explanation",
        "Q9: Does the paper explain why calorie restriction increases the lifespan of vertebrates?": "calorie_restriction_lifespan_explanation"
    }
    
    ground_truth = {}
    for old_q, new_key in mapping.items():
        if old_q in validation_entry:
            ground_truth[new_key] = validation_entry[old_q]
    
    return ground_truth


def evaluate_strategy(
    rag_system: ScientificRAG,
    llm_client: AzureOpenAIClient,
    doi: str,
    ground_truth: Dict,
    use_multi_query: bool = False,
    strategy_name: str = "default"
) -> Dict:
    """Evaluate a single RAG strategy on one paper."""
    
    # Load questions
    with open('data/questions_part2.json', 'r') as f:
        questions_data = json.load(f)
    
    # Initialize RAG system with strategy
    complete_rag = CompleteRAGSystem(
        rag_system=rag_system,
        llm_client=llm_client,
        default_n_results=10,
        use_multi_query=use_multi_query
    )
    
    results = {
        'strategy': strategy_name,
        'doi': doi,
        'correct': 0,
        'total': 0,
        'questions': {}
    }
    
    print(f"\n{'='*70}")
    print(f"Testing Strategy: {strategy_name}")
    print(f"DOI: {doi}")
    print(f"{'='*70}")
    
    for question_key, question_obj in questions_data.items():
        if question_key not in ground_truth:
            continue
        
        question_text = question_obj['question']
        options_str = question_obj['answers']
        options = [opt.strip() for opt in options_str.split('/') if opt.strip()]
        
        expected_answer = ground_truth[question_key]
        
        print(f"\n{question_key}: ", end='', flush=True)
        
        try:
            answer = complete_rag.answer_question(
                question=question_text,
                n_results=10,
                temperature=0.2,
                max_tokens=300,
                answer_options=options,
                doi=doi
            )
            
            predicted_answer = answer.get('answer')
            confidence = answer.get('confidence', 0.0)
            parse_error = answer.get('parse_error', False)
            
            # Normalize answers for comparison
            predicted_normalized = predicted_answer.strip() if predicted_answer else None
            expected_normalized = expected_answer.strip()
            
            # Check if correct
            is_correct = False
            if predicted_normalized and not parse_error:
                # Exact match
                if predicted_normalized == expected_normalized:
                    is_correct = True
                # Partial match (e.g., "Yes, quantitatively shown" vs "Yes")
                elif expected_normalized in predicted_normalized or predicted_normalized in expected_normalized:
                    is_correct = True
            
            if is_correct:
                results['correct'] += 1
                print(f"✓ (conf: {confidence:.2f})")
            else:
                print(f"✗ Expected: {expected_normalized}, Got: {predicted_normalized}")
            
            results['total'] += 1
            results['questions'][question_key] = {
                'expected': expected_normalized,
                'predicted': predicted_normalized,
                'correct': is_correct,
                'confidence': confidence,
                'parse_error': parse_error
            }
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results['questions'][question_key] = {
                'expected': expected_normalized,
                'predicted': None,
                'correct': False,
                'error': str(e)
            }
            results['total'] += 1
    
    accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
    print(f"\n{'─'*70}")
    print(f"Accuracy: {results['correct']}/{results['total']} = {accuracy:.2%}")
    print(f"{'─'*70}")
    
    return results


def main():
    print("\n" + "="*70)
    print("RAG Strategy Evaluation on Validation Set")
    print("="*70)
    
    # Load validation set
    validation_data = load_validation_set('data/qa_validation_set_extended.json')
    print(f"\nTotal papers in validation set: {len(validation_data)}")
    
    # Select 5 diverse papers
    random.seed(42)
    selected_papers = random.sample(validation_data, min(5, len(validation_data)))
    
    print("\nSelected papers:")
    for i, paper in enumerate(selected_papers, 1):
        print(f"  {i}. {paper['doi']}")
    
    # Initialize RAG system (shared across strategies)
    print("\nInitializing RAG system...")
    rag = ScientificRAG(
        collection_name=os.getenv('COLLECTION_NAME', 'scientific_papers_optimal'),
        persist_directory=os.getenv('PERSIST_DIR', './chroma_db_optimal'),
        embedding_model=os.getenv('EMBEDDING_MODEL', 'allenai/specter2'),
        backup_embedding_model=os.getenv('BACKUP_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
    )
    
    llm_client = AzureOpenAIClient(model='gpt-4.1')
    
    # Define strategies to test
    strategies = [
        {'name': 'Enhanced (default)', 'use_multi_query': False},
        {'name': 'Multi-Query (Enhanced + HyDE + Expanded)', 'use_multi_query': True},
    ]
    
    # Evaluate each strategy on each paper
    all_results = []
    
    for strategy in strategies:
        strategy_results = []
        
        for paper in selected_papers:
            doi = paper['doi']
            ground_truth = map_validation_to_new_format(paper)
            
            result = evaluate_strategy(
                rag_system=rag,
                llm_client=llm_client,
                doi=doi,
                ground_truth=ground_truth,
                use_multi_query=strategy['use_multi_query'],
                strategy_name=strategy['name']
            )
            
            strategy_results.append(result)
            time.sleep(1)  # Rate limiting
        
        all_results.append({
            'strategy': strategy['name'],
            'papers': strategy_results
        })
    
    # Calculate overall statistics
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    
    for strategy_result in all_results:
        strategy_name = strategy_result['strategy']
        total_correct = sum(r['correct'] for r in strategy_result['papers'])
        total_questions = sum(r['total'] for r in strategy_result['papers'])
        accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        print(f"\n{strategy_name}:")
        print(f"  Accuracy: {total_correct}/{total_questions} = {accuracy:.2%}")
        
        # Per-paper breakdown
        for paper_result in strategy_result['papers']:
            paper_acc = paper_result['correct'] / paper_result['total'] if paper_result['total'] > 0 else 0
            print(f"    {paper_result['doi']}: {paper_result['correct']}/{paper_result['total']} = {paper_acc:.2%}")
    
    # Save results
    output_file = 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
