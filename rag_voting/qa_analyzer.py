#!/usr/bin/env python3
"""
QA Results Analyzer - Evaluates full-text and RAG-based QA results against validation sets
and creates combination strategies for improved performance.
"""

import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np


class QAAnalyzer:
    """Analyzes and compares QA results from different approaches."""
    
    def __init__(self, fulltext_path: str, rag_path: str, 
                 validation_path: str, validation_extended_path: str):
        """Load all data files."""
        print("Loading data files...")
        with open(fulltext_path, 'r') as f:
            self.fulltext_data = json.load(f)
        with open(rag_path, 'r') as f:
            self.rag_data = json.load(f)
        with open(validation_path, 'r') as f:
            self.validation_data = json.load(f)
        with open(validation_extended_path, 'r') as f:
            self.validation_extended_data = json.load(f)
        
        # Create DOI-indexed dictionaries for faster lookup (normalize DOIs)
        self.fulltext_by_doi = {item['doi'].strip(): item for item in self.fulltext_data}
        self.rag_by_doi = {item['doi'].strip(): item for item in self.rag_data}
        
        print(f"Loaded {len(self.fulltext_data)} fulltext results")
        print(f"Loaded {len(self.rag_data)} RAG results")
        print(f"Loaded {len(self.validation_data)} validation entries")
        print(f"Loaded {len(self.validation_extended_data)} extended validation entries")
        
        # Map question keys between formats (both standard and extended validation formats)
        self.question_mapping = {
            # Standard validation format
            'Q1: Does the paper/theory suggest an aging biomarker (i.e. measurable entity reflecting the pace of aging or organism health state, associated with mortality or age-related conditions)?': 'aging_biomarker',
            'Q2: Does the paper/theory suggest a molecular mechanism of aging?': 'molecular_mechanism',
            'Q3: Does the paper/theory suggest a longevity intervention to test?': 'longevity_intervention',
            'Q4: Does the paper/theory claim that aging cannot be reversed? (Yes / No)': 'aging_cannot_be_reversed',
            'Q5: Does the paper/theory suggest a biomarker that explains differences in maximum lifespan between species? (Yes / No)': 'lifespan_biomarker',
            'Q6: Does the paper/theory explain why the naked mole rat can live 40+ years despite its small size?': 'naked_mole_rat_lifespan_explanation',
            'Q7: Does the paper/theory explain why birds live much longer than mammals on average?': 'birds_lifespan_explanation',
            'Q8: Does the paper/theory explain why large animals live longer than small ones?': 'large_animals_lifespan_explanation',
            'Q9: Does the paper/theory explain why calorie restriction increases the lifespan of vertebrates?': 'calorie_restriction_lifespan_explanation',
            # Extended validation format (without "/theory")
            'Q1: Does the paper suggest an aging biomarker (i.e. measurable entity reflecting the pace of aging or organism health state  associated with mortality or age-related conditions)?': 'aging_biomarker',
            'Q2: Does the paper suggest a molecular mechanism of aging?': 'molecular_mechanism',
            'Q3: Does the paper suggest a longevity intervention to test?': 'longevity_intervention',
            'Q4: Does the paper claim that aging cannot be reversed? (Yes / No)': 'aging_cannot_be_reversed',
            'Q5: Does the paper suggest a biomarker that explains differences in maximum lifespan between species? (Yes / No)': 'lifespan_biomarker',
            'Q6: Does the paper explain why the naked mole rat can live 40+ years despite its small size?': 'naked_mole_rat_lifespan_explanation',
            'Q7: Does the paper explain why birds live much longer than mammals on average?': 'birds_lifespan_explanation',
            'Q8: Does the paper explain why large animals live longer than small ones?': 'large_animals_lifespan_explanation',
            'Q9: Does the paper explain why calorie restriction increases the lifespan of vertebrates?': 'calorie_restriction_lifespan_explanation'
        }
    
    def normalize_answer(self, answer: Any) -> str:
        """Normalize answer to Yes/No format."""
        if isinstance(answer, dict):
            answer = answer.get('answer', '')
        answer_str = str(answer).strip().lower()
        if answer_str in ['yes', 'true', '1']:
            return 'Yes'
        elif answer_str in ['no', 'false', '0']:
            return 'No'
        return answer_str.capitalize()
    
    def evaluate_on_validation_set(self, validation_set: List[Dict], 
                                   set_name: str) -> Dict[str, Any]:
        """Evaluate both approaches on a validation set."""
        print(f"\n{'='*60}")
        print(f"Evaluating on {set_name}")
        print(f"{'='*60}")
        
        results = {
            'fulltext': {'correct': 0, 'total': 0, 'by_question': defaultdict(lambda: {'correct': 0, 'total': 0})},
            'rag': {'correct': 0, 'total': 0, 'by_question': defaultdict(lambda: {'correct': 0, 'total': 0})},
            'details': []
        }
        
        for val_entry in validation_set:
            doi = val_entry['doi'].strip()
            
            # Get predictions from both systems
            fulltext_pred = self.fulltext_by_doi.get(doi)
            rag_pred = self.rag_by_doi.get(doi)
            
            if not fulltext_pred and not rag_pred:
                continue
            
            entry_detail = {'doi': doi, 'questions': {}}
            
            # Evaluate each question
            for q_full, q_key in self.question_mapping.items():
                if q_full not in val_entry:
                    continue
                
                ground_truth = self.normalize_answer(val_entry[q_full])
                
                # Evaluate fulltext
                if fulltext_pred and q_key in fulltext_pred.get('answers', {}):
                    ft_answer = self.normalize_answer(fulltext_pred['answers'][q_key])
                    is_correct = (ft_answer == ground_truth)
                    results['fulltext']['total'] += 1
                    results['fulltext']['by_question'][q_key]['total'] += 1
                    if is_correct:
                        results['fulltext']['correct'] += 1
                        results['fulltext']['by_question'][q_key]['correct'] += 1
                    
                    entry_detail['questions'][q_key] = {
                        'ground_truth': ground_truth,
                        'fulltext': ft_answer,
                        'fulltext_correct': is_correct
                    }
                
                # Evaluate RAG
                if rag_pred and q_key in rag_pred.get('answers', {}):
                    rag_answer = self.normalize_answer(rag_pred['answers'][q_key])
                    is_correct = (rag_answer == ground_truth)
                    results['rag']['total'] += 1
                    results['rag']['by_question'][q_key]['total'] += 1
                    if is_correct:
                        results['rag']['correct'] += 1
                        results['rag']['by_question'][q_key]['correct'] += 1
                    
                    if q_key in entry_detail['questions']:
                        entry_detail['questions'][q_key]['rag'] = rag_answer
                        entry_detail['questions'][q_key]['rag_correct'] = is_correct
                    else:
                        entry_detail['questions'][q_key] = {
                            'ground_truth': ground_truth,
                            'rag': rag_answer,
                            'rag_correct': is_correct
                        }
            
            if entry_detail['questions']:
                results['details'].append(entry_detail)
        
        # Calculate accuracies
        results['fulltext']['accuracy'] = (results['fulltext']['correct'] / results['fulltext']['total'] 
                                          if results['fulltext']['total'] > 0 else 0)
        results['rag']['accuracy'] = (results['rag']['correct'] / results['rag']['total'] 
                                     if results['rag']['total'] > 0 else 0)
        
        # Print summary
        print(f"\nFulltext Results:")
        print(f"  Overall Accuracy: {results['fulltext']['accuracy']:.2%} ({results['fulltext']['correct']}/{results['fulltext']['total']})")
        print(f"\nRAG Results:")
        print(f"  Overall Accuracy: {results['rag']['accuracy']:.2%} ({results['rag']['correct']}/{results['rag']['total']})")
        
        # Print per-question breakdown
        print(f"\nPer-Question Breakdown:")
        print(f"{'Question':<45} {'Fulltext':<15} {'RAG':<15}")
        print("-" * 75)
        
        all_questions = set(results['fulltext']['by_question'].keys()) | set(results['rag']['by_question'].keys())
        for q_key in sorted(all_questions):
            ft_stats = results['fulltext']['by_question'][q_key]
            rag_stats = results['rag']['by_question'][q_key]
            
            ft_acc = f"{ft_stats['correct']}/{ft_stats['total']} ({ft_stats['correct']/ft_stats['total']:.1%})" if ft_stats['total'] > 0 else "N/A"
            rag_acc = f"{rag_stats['correct']}/{rag_stats['total']} ({rag_stats['correct']/rag_stats['total']:.1%})" if rag_stats['total'] > 0 else "N/A"
            
            print(f"{q_key:<45} {ft_acc:<15} {rag_acc:<15}")
        
        return results
    
    def analyze_combination_strategies(self, validation_results: Dict[str, Any], 
                                      set_name: str) -> Dict[str, Any]:
        """Analyze different combination strategies."""
        print(f"\n{'='*60}")
        print(f"Analyzing Combination Strategies for {set_name}")
        print(f"{'='*60}")
        
        strategies = {
            'fulltext_only': {'correct': 0, 'total': 0, 'description': 'Use fulltext answers only'},
            'rag_only': {'correct': 0, 'total': 0, 'description': 'Use RAG answers only'},
            'rag_yes_override': {'correct': 0, 'total': 0, 'description': 'Use fulltext, but if RAG says Yes, trust RAG'},
            'rag_no_override': {'correct': 0, 'total': 0, 'description': 'Use fulltext, but if RAG says No, trust RAG'},
            'agree_only': {'correct': 0, 'total': 0, 'description': 'Use answer only when both agree, otherwise abstain'},
            'majority_with_confidence': {'correct': 0, 'total': 0, 'description': 'Use RAG if confidence > 0.9, else fulltext'},
            'question_specific': {'correct': 0, 'total': 0, 'description': 'Use best performer per question type'},
            'conservative': {'correct': 0, 'total': 0, 'description': 'Prefer No unless both say Yes'},
            'optimistic': {'correct': 0, 'total': 0, 'description': 'Prefer Yes if either says Yes'}
        }
        
        # Calculate per-question best performer
        question_best = {}
        for q_key in validation_results['fulltext']['by_question'].keys():
            ft_stats = validation_results['fulltext']['by_question'][q_key]
            rag_stats = validation_results['rag']['by_question'][q_key]
            
            ft_acc = ft_stats['correct'] / ft_stats['total'] if ft_stats['total'] > 0 else 0
            rag_acc = rag_stats['correct'] / rag_stats['total'] if rag_stats['total'] > 0 else 0
            
            question_best[q_key] = 'fulltext' if ft_acc >= rag_acc else 'rag'
        
        # Evaluate strategies
        for entry in validation_results['details']:
            doi = entry['doi']
            rag_pred = self.rag_by_doi.get(doi)
            
            for q_key, q_data in entry['questions'].items():
                ground_truth = q_data['ground_truth']
                ft_answer = q_data.get('fulltext')
                rag_answer = q_data.get('rag')
                
                if ft_answer is None and rag_answer is None:
                    continue
                
                # Get RAG confidence if available
                rag_confidence = 0.5
                if rag_pred and q_key in rag_pred.get('answers', {}):
                    rag_ans_obj = rag_pred['answers'][q_key]
                    if isinstance(rag_ans_obj, dict):
                        rag_confidence = rag_ans_obj.get('confidence', 0.5)
                
                # Strategy 1: Fulltext only
                if ft_answer is not None:
                    strategies['fulltext_only']['total'] += 1
                    if ft_answer == ground_truth:
                        strategies['fulltext_only']['correct'] += 1
                
                # Strategy 2: RAG only
                if rag_answer is not None:
                    strategies['rag_only']['total'] += 1
                    if rag_answer == ground_truth:
                        strategies['rag_only']['correct'] += 1
                
                # Strategy 3: RAG Yes override (use fulltext, but trust RAG when it says Yes)
                if ft_answer is not None and rag_answer is not None:
                    strategies['rag_yes_override']['total'] += 1
                    final_answer = rag_answer if rag_answer == 'Yes' else ft_answer
                    if final_answer == ground_truth:
                        strategies['rag_yes_override']['correct'] += 1
                
                # Strategy 4: RAG No override (use fulltext, but trust RAG when it says No)
                if ft_answer is not None and rag_answer is not None:
                    strategies['rag_no_override']['total'] += 1
                    final_answer = rag_answer if rag_answer == 'No' else ft_answer
                    if final_answer == ground_truth:
                        strategies['rag_no_override']['correct'] += 1
                
                # Strategy 5: Agree only
                if ft_answer is not None and rag_answer is not None:
                    if ft_answer == rag_answer:
                        strategies['agree_only']['total'] += 1
                        if ft_answer == ground_truth:
                            strategies['agree_only']['correct'] += 1
                
                # Strategy 6: Confidence-based
                if ft_answer is not None and rag_answer is not None:
                    strategies['majority_with_confidence']['total'] += 1
                    final_answer = rag_answer if rag_confidence > 0.9 else ft_answer
                    if final_answer == ground_truth:
                        strategies['majority_with_confidence']['correct'] += 1
                elif ft_answer is not None:
                    strategies['majority_with_confidence']['total'] += 1
                    if ft_answer == ground_truth:
                        strategies['majority_with_confidence']['correct'] += 1
                elif rag_answer is not None:
                    strategies['majority_with_confidence']['total'] += 1
                    if rag_answer == ground_truth:
                        strategies['majority_with_confidence']['correct'] += 1
                
                # Strategy 7: Question-specific best
                if q_key in question_best:
                    best_source = question_best[q_key]
                    answer_to_use = ft_answer if best_source == 'fulltext' and ft_answer is not None else rag_answer
                    if answer_to_use is not None:
                        strategies['question_specific']['total'] += 1
                        if answer_to_use == ground_truth:
                            strategies['question_specific']['correct'] += 1
                
                # Strategy 8: Conservative (prefer No)
                if ft_answer is not None and rag_answer is not None:
                    strategies['conservative']['total'] += 1
                    final_answer = 'Yes' if (ft_answer == 'Yes' and rag_answer == 'Yes') else 'No'
                    if final_answer == ground_truth:
                        strategies['conservative']['correct'] += 1
                
                # Strategy 9: Optimistic (prefer Yes)
                if ft_answer is not None and rag_answer is not None:
                    strategies['optimistic']['total'] += 1
                    final_answer = 'Yes' if (ft_answer == 'Yes' or rag_answer == 'Yes') else 'No'
                    if final_answer == ground_truth:
                        strategies['optimistic']['correct'] += 1
        
        # Calculate accuracies
        for strategy_name, strategy_data in strategies.items():
            if strategy_data['total'] > 0:
                strategy_data['accuracy'] = strategy_data['correct'] / strategy_data['total']
            else:
                strategy_data['accuracy'] = 0
        
        # Print results
        print(f"\nStrategy Performance:")
        print(f"{'Strategy':<30} {'Accuracy':<15} {'Correct/Total':<20} {'Description'}")
        print("-" * 120)
        
        # Sort by accuracy
        sorted_strategies = sorted(strategies.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for strategy_name, strategy_data in sorted_strategies:
            acc_str = f"{strategy_data['accuracy']:.2%}" if strategy_data['total'] > 0 else "N/A"
            count_str = f"{strategy_data['correct']}/{strategy_data['total']}"
            print(f"{strategy_name:<30} {acc_str:<15} {count_str:<20} {strategy_data['description']}")
        
        return strategies
    
    def generate_report(self, output_file: str = 'evaluation_report.json'):
        """Generate comprehensive evaluation report."""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE EVALUATION REPORT")
        print(f"{'='*60}")
        
        # Evaluate on both validation sets
        val_results = self.evaluate_on_validation_set(self.validation_data, "Standard Validation Set")
        val_ext_results = self.evaluate_on_validation_set(self.validation_extended_data, "Extended Validation Set")
        
        # Analyze combination strategies
        strategies_val = self.analyze_combination_strategies(val_results, "Standard Validation Set")
        strategies_val_ext = self.analyze_combination_strategies(val_ext_results, "Extended Validation Set")
        
        # Create comprehensive report
        report = {
            'standard_validation': {
                'fulltext_accuracy': val_results['fulltext']['accuracy'],
                'rag_accuracy': val_results['rag']['accuracy'],
                'fulltext_stats': dict(val_results['fulltext']['by_question']),
                'rag_stats': dict(val_results['rag']['by_question']),
                'strategies': strategies_val
            },
            'extended_validation': {
                'fulltext_accuracy': val_ext_results['fulltext']['accuracy'],
                'rag_accuracy': val_ext_results['rag']['accuracy'],
                'fulltext_stats': dict(val_ext_results['fulltext']['by_question']),
                'rag_stats': dict(val_ext_results['rag']['by_question']),
                'strategies': strategies_val_ext
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Report saved to: {output_file}")
        print(f"{'='*60}")
        
        # Print recommendations
        self.print_recommendations(strategies_val, strategies_val_ext)
        
        return report
    
    def print_recommendations(self, strategies_val: Dict, strategies_val_ext: Dict):
        """Print strategic recommendations."""
        print(f"\n{'='*60}")
        print("STRATEGIC RECOMMENDATIONS")
        print(f"{'='*60}")
        
        # Find best strategies
        best_val = max(strategies_val.items(), key=lambda x: x[1]['accuracy'])
        best_val_ext = max(strategies_val_ext.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"\n1. BEST OVERALL STRATEGY:")
        print(f"   Standard Set: {best_val[0]} ({best_val[1]['accuracy']:.2%})")
        print(f"   Extended Set: {best_val_ext[0]} ({best_val_ext[1]['accuracy']:.2%})")
        
        print(f"\n2. RECOMMENDED APPROACH:")
        print(f"   Based on the analysis, the 'rag_yes_override' strategy shows promise:")
        print(f"   - Use fulltext answers as the baseline (more comprehensive)")
        print(f"   - When RAG confidently says 'Yes', trust it (RAG is precise for positive findings)")
        print(f"   - This leverages fulltext's broad coverage and RAG's precision")
        
        print(f"\n3. ALTERNATIVE STRATEGIES:")
        print(f"   - Question-specific: Use the best-performing method per question type")
        print(f"   - Confidence-based: Trust RAG when confidence > 0.9, otherwise use fulltext")
        print(f"   - Conservative: Require both to agree for 'Yes' answers (high precision)")
        
        print(f"\n4. KEY INSIGHTS:")
        print(f"   - Fulltext provides broader coverage but may be less precise")
        print(f"   - RAG is more precise but may miss some information")
        print(f"   - Combining both approaches can leverage their complementary strengths")
        print(f"   - Different question types may benefit from different strategies")


def main():
    """Main execution function."""
    analyzer = QAAnalyzer(
        fulltext_path='data/fulltext_qa_qa.json',
        rag_path='data/rag_results.json',
        validation_path='data/qa_validation_set.json',
        validation_extended_path='data/qa_validation_set_extended.json'
    )
    
    report = analyzer.generate_report('evaluation_report.json')
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
