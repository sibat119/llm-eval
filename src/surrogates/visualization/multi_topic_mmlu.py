import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_zero_shot_results(base_path):
    """Load zero-shot results for all domains."""
    domains = ['high_school_computer_science', 'philosophy', 'public_relations']
    results = {}
    
    for domain in domains:
        file_path = os.path.join(base_path, domain, '0', 'results.json')
        with open(file_path, 'r') as f:
            results[domain] = json.load(f)
    
    return results

def load_surrogate_results(base_path, shot=5, selection_strategy="random"):
    """Load surrogate results for all domains."""
    domains = ['high_school_computer_science', 'philosophy', 'public_relations']
    results = {}
    
    for domain in domains:
        domain_path = os.path.join(base_path, 'surrogate', domain, f'{shot}-shot-{selection_strategy}-selection')
        results[domain] = {}
        
        # Look for surrogate result files
        for filename in os.listdir(domain_path):
            if filename.endswith('.json'):
                file_path = os.path.join(domain_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle complex model names in filename
                if "candidate" in filename and "surrogate" in filename:
                    # Split by "surrogate" to separate candidate and surrogate parts
                    parts = filename.replace('.json', '').split('-surrogate-')
                    
                    # Get candidate model (after "candidate-")
                    candidate_part = parts[0].replace('candidate-', '')
                    
                    # Get surrogate model
                    surrogate_part = parts[1]
                    
                    # Simplify model names to short versions
                    if "qwen" in candidate_part.lower():
                        candidate = "Qwen"
                    elif "llama" in candidate_part.lower():
                        candidate = "Llama"
                    else:
                        candidate = candidate_part
                    
                    if "qwen" in surrogate_part.lower():
                        surrogate = "Qwen"
                    elif "llama" in surrogate_part.lower():
                        surrogate = "Llama"
                    else:
                        surrogate = surrogate_part
                    
                    key = f"{candidate}->{surrogate}"
                    results[domain][key] = data
    
    return results

def create_metric_dashboard(zero_shot_results, surrogate_results, selection_strategy="random"):
    """Create standardized metric dashboard for each domain."""
    domains = list(zero_shot_results.keys())
    metrics = ['accuracy_ranking', 'f1_score_token_agreement', 'f1_score_ranking', 'sbert_similarity', 'agreement_score', 'both_ground_truth_match']
    metric_labels = ['Accuracy', 'F1 Score token', 'F1 Score rank', 'SBERT Similarity', 'Agreement Score', 'GT match score']
    
    for domain in domains:
        plt.figure(figsize=(18, 12))
        
        # Prepare data for zero-shot metrics
        zero_shot_data = []
        for model in ['Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct']:
            model_short = 'Qwen' if 'Qwen' in model else 'Llama'
            for metric, label in zip(metrics[:4], metric_labels[:4]):  # First 4 metrics available in zero-shot
                if metric in zero_shot_results[domain][model]:
                    zero_shot_data.append({
                        'Model': model_short,
                        'Metric': label,
                        'Value': zero_shot_results[domain][model][metric],
                        'Type': 'Zero-Shot'
                    })
        
        # Prepare data for surrogate metrics
        surrogate_data = []
        model_pairs = []
        
        for result_key in surrogate_results[domain]:
            if 'qwen' in result_key.lower() and 'llama' in result_key.lower():
                source_model = 'Qwen' if 'qwen' in result_key.lower().split('->')[0] else 'Llama'
                target_model = 'Llama' if source_model == 'Qwen' else 'Qwen'
                pair = f"{source_model}->{target_model}"
                model_pairs.append(pair)
                
                for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                    if metric in surrogate_results[domain][result_key]['wrt_gt']:
                        surrogate_data.append({
                            'Model': pair,
                            'Metric': label,
                            'Value': surrogate_results[domain][result_key]['wrt_gt'][metric],
                            'Type': 'Surrogate'
                        })
                    elif metric in surrogate_results[domain][result_key]['wrt_gt']['agreement_without_mind_model']:
                        zero_shot_aggrement = surrogate_results[domain][result_key]['wrt_gt']['agreement_without_mind_model'][metric]
                        
                        few_shot_aggrement = surrogate_results[domain][result_key]['wrt_gt']['agreement_after_mind_model'][metric]
                        
                        surrogate_data.append({
                            'Model': 'zero-shot',
                            'Metric': label,
                            'Value': zero_shot_aggrement,
                            'Type': 'Zero-Shot'
                        })
                        
                        surrogate_data.append({
                            'Model': pair,
                            'Metric': label,
                            'Value': few_shot_aggrement,
                            'Type': 'Surrogate'
                        })
        
        # Combine data
        df = pd.DataFrame(zero_shot_data + surrogate_data)
        
        # Create subplot for each metric
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            plt.subplot(2, 3, i+1)  # Changed to 2 rows x 3 columns for 6 metrics
            
            # Filter data for this metric
            metric_df = df[df['Metric'] == label]
            
            if not metric_df.empty:
                # Create grouped bar chart
                sns.barplot(x='Model', y='Value', hue='Type', data=metric_df)
                
                plt.title(label)
                plt.xlabel('')
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1.0)
                
                if i % 3 == 0:  # Only show y-label for leftmost plots
                    plt.ylabel('Value')
                else:
                    plt.ylabel('')
                
                if i == 1:  # Only show legend for one subplot
                    plt.legend(title='')
                else:
                    plt.legend([])
        
        plt.suptitle(f"{domain.replace('_', ' ').title()} - Metric Dashboard", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        directory = f"data/results/{domain}"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{directory}/{selection_strategy}_metric_dashboard.png', dpi=300)
        plt.close()
        

def create_agreement_analysis_plots(surrogate_results, selection_strategy="random"):
    """
    Create a figure with four subplots for agreement analysis based on our metrics.py implementation
    
    Args:
        surrogate_results: Dictionary containing surrogate model results with transition metrics
    """
    # Create plot for each domain
    domains = list(surrogate_results.keys())
    
    for domain in domains:
        # Extract the transition metrics for this domain
        domain_data = {}
        
        for model_pair, results in surrogate_results[domain].items():
            if 'wrt_gt' in results and 'transition_metrics' in results['wrt_gt']:
                domain_data[model_pair] = results['wrt_gt']['transition_metrics']
        
        if not domain_data:
            print(f"No transition metrics found for domain: {domain}")
            continue
            
        # Set up the figure and subplots
        plt.figure(figsize=(18, 14))
        plt.suptitle(f"{domain.replace('_', ' ').title()} - Agreement Analysis", fontsize=20, fontweight='bold')
        
        # Extract model combinations for labeling
        model_combinations = list(domain_data.keys())
        
        # Subplot 1: Agreement Transition Matrix
        plt.subplot(2, 2, 1)
        create_transition_matrix_plot(domain_data, model_combinations)
        
        # Subplot 2: Response Length vs Agreement
        plt.subplot(2, 2, 2)
        create_response_length_plot(domain_data, model_combinations)
        
        # Subplot 3: Ground Truth Match Analysis
        plt.subplot(2, 2, 3)
        create_ground_truth_match_plot(surrogate_results[domain])
        
        # Subplot 4: Agreement Pattern by Topic
        plt.subplot(2, 2, 4)
        create_confidence_correlation_plot(surrogate_results[domain])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        directory = f"data/results/{domain}"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{directory}/{selection_strategy}_agreement_analysis.png', dpi=300)
        plt.close()

def create_transition_matrix_plot(data, model_combinations):
    """Create a bar plot showing agreement transitions from zero-shot to few-shot for each model combination"""
    
    # Set up the plot
    plt.title("Agreement Transitions by Model Direction", fontsize=14)
    
    # Prepare data structure for plotting
    transitions_types = ['Remain in Agreement', 'Agreement → Disagreement', 
                        'Disagreement → Agreement', 'Remain in Disagreement']
    transition_keys = ['zero_agree_few_agree', 'zero_agree_few_disagree', 
                      'zero_disagree_few_agree', 'zero_disagree_few_disagree']
    
    # Get data for all model combinations
    plot_data = []
    for model_key in data.keys():
        for transition_type, key in zip(transitions_types, transition_keys):
            plot_data.append({
                'Model': model_key,
                'Transition': transition_type,
                'Value': data[model_key]['agreement_transitions'][key]
            })
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(plot_data)
    
    # Set color mapping
    color_map = {
        'Remain in Agreement': 'darkgreen',
        'Agreement → Disagreement': 'tomato',
        'Disagreement → Agreement': 'mediumseagreen',
        'Remain in Disagreement': 'darkred'
    }
    
    # Create grouped bar chart
    ax = sns.barplot(x='Transition', y='Value', hue='Model', data=df, palette='Set2')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')
    
    plt.ylim(0, 1.0)
    plt.xlabel("Transition Type")
    plt.ylabel("Proportion of Test Cases")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Model Direction")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

def create_response_length_plot(data, model_combinations):
    """Create grouped bar chart comparing response lengths for agree/disagree cases for each model combination"""
    
    # Set up the plot
    plt.title("Response Length vs Agreement Status by Model", fontsize=14)
    
    # Prepare data for plotting
    plot_data = []
    
    for model_key in data.keys():
        length_data = data[model_key]['response_length_metrics']
        
        # Add zero-shot data
        plot_data.append({
            'Model': model_key,
            'Condition': 'Zero-Shot\nAgree',
            'Length': length_data['zero_shot']['agree_avg_combined_length']
        })
        
        plot_data.append({
            'Model': model_key,
            'Condition': 'Zero-Shot\nDisagree',
            'Length': length_data['zero_shot']['disagree_avg_combined_length']
        })
        
        # Add few-shot data
        plot_data.append({
            'Model': model_key,
            'Condition': 'Few-Shot\nAgree',
            'Length': length_data['few_shot']['agree_avg_combined_length']
        })
        
        plot_data.append({
            'Model': model_key,
            'Condition': 'Few-Shot\nDisagree',
            'Length': length_data['few_shot']['disagree_avg_combined_length']
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    
    # Create grouped bar chart
    ax = sns.barplot(x='Condition', y='Length', hue='Model', data=df, palette='Set2')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge')
    
    plt.xlabel("Condition")
    plt.ylabel("Average Response Length (tokens)")
    plt.legend(title="Model Direction")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

def create_ground_truth_match_plot(domain_results):
    """
    Create a plot showing the relationship between model agreement and ground truth match
    """
    plt.title("Agreement vs Ground Truth Match", fontsize=14)
    
    # Collect data
    plot_data = []
    
    for model_pair, results in domain_results.items():
        if 'wrt_gt' in results:
            # Get agreement scores from the metrics
            if 'agreement_after_mind_model' in results['wrt_gt']:
                # With mind model (few-shot)
                few_shot_agreement = results['wrt_gt']['agreement_after_mind_model']['agreement_score']
                few_shot_model_gt = results['wrt_gt']['agreement_after_mind_model']['model_ground_truth_match']
                few_shot_blackbox_gt = results['wrt_gt']['agreement_after_mind_model']['blackbox_ground_truth_match']
                few_shot_both_gt = results['wrt_gt']['agreement_after_mind_model']['both_ground_truth_match']
                
                plot_data.append({
                    'Model': model_pair,
                    'Condition': 'With Mind Model',
                    'Agreement Score': few_shot_agreement,
                    'Surrogate GT Match': few_shot_model_gt,
                    'Blackbox GT Match': few_shot_blackbox_gt,
                    'Both GT Match': few_shot_both_gt
                })
            
            if 'agreement_without_mind_model' in results['wrt_gt']:
                # Without mind model (zero-shot)
                zero_shot_agreement = results['wrt_gt']['agreement_without_mind_model']['agreement_score']
                zero_shot_model_gt = results['wrt_gt']['agreement_without_mind_model']['model_ground_truth_match']
                zero_shot_blackbox_gt = results['wrt_gt']['agreement_without_mind_model']['blackbox_ground_truth_match']
                zero_shot_both_gt = results['wrt_gt']['agreement_without_mind_model']['both_ground_truth_match']
                
                plot_data.append({
                    'Model': model_pair,
                    'Condition': 'Without Mind Model',
                    'Agreement Score': zero_shot_agreement,
                    'Surrogate GT Match': zero_shot_model_gt,
                    'Blackbox GT Match': zero_shot_blackbox_gt,
                    'Both GT Match': zero_shot_both_gt
                })
    
    if not plot_data:
        plt.text(0.5, 0.5, "No ground truth match data available", 
                 ha='center', va='center', fontsize=12)
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    
    # Melt the DataFrame for easier plotting of GT metrics
    melted_df = pd.melt(df, 
                        id_vars=['Model', 'Condition', 'Agreement Score'],
                        value_vars=['Surrogate GT Match', 'Blackbox GT Match', 'Both GT Match'],
                        var_name='Match Type', value_name='Match Rate')
    
    # Create grouped bar chart
    ax = sns.barplot(x='Condition', y='Match Rate', hue='Match Type', 
                     data=melted_df, palette='Blues')
    
    # Add horizontal lines for agreement scores
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        for _, row in model_data.iterrows():
            plt.axhline(y=row['Agreement Score'], color='red', linestyle='--', alpha=0.5)
            plt.text(plt.xlim()[1] * 0.95, row['Agreement Score'], 
                     f"Agreement ({row['Condition']}): {row['Agreement Score']:.2f}", 
                     va='center', ha='right', color='red', fontsize=9)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')
    
    plt.ylim(0, 1.0)
    plt.xlabel("Model Configuration")
    plt.ylabel("Match Rate")
    plt.legend(title="Match Type")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

def create_confidence_correlation_plot(domain_results):
    """
    Create a bar plot showing average similarity scores for different agreement transition types
    for each model pair (Llama->Qwen and Qwen->Llama).
    """
    plt.title("Similarity by Agreement Transition Type", fontsize=14)
    
    # Collect data from samples
    similarity_data = []
    
    for model_pair, results in domain_results.items():
        if 'wrt_gt' not in results:
            continue
        
        # Check if we have both with and without mind model data
        if ('agreement_after_mind_model' not in results['wrt_gt'] or 
            'agreement_without_mind_model' not in results['wrt_gt']):
            continue
        
        # Try to get semantic similarity data from transition metrics
        if 'transition_metrics' in results['wrt_gt'] and 'symantic_similarity' in results['wrt_gt']['transition_metrics']:
            sim_data = results['wrt_gt']['transition_metrics']['symantic_similarity']
            
            # Extract the similarity metrics for different transition types
            similarity_data.append({
                'Model': model_pair,
                'Transition': 'Remain in Agreement',
                'Value': sim_data.get('zero_agree_few_agree_cosine_distance', 0.5)
            })
            
            similarity_data.append({
                'Model': model_pair,
                'Transition': 'Agreement → Disagreement',
                'Value': sim_data.get('zero_agree_few_disagree_cosine_distance', 0.5)
            })
            
            similarity_data.append({
                'Model': model_pair,
                'Transition': 'Disagreement → Agreement',
                'Value': sim_data.get('zero_disagree_few_agree_cosine_distance', 0.5)
            })
            
            similarity_data.append({
                'Model': model_pair,
                'Transition': 'Remain in Disagreement',
                'Value': sim_data.get('zero_disagree_few_disagree_cosine_distance', 0.5)
            })
        else:
            # Use default values if data is not available
            for transition in ['Remain in Agreement', 'Agreement → Disagreement', 
                              'Disagreement → Agreement', 'Remain in Disagreement']:
                similarity_data.append({
                    'Model': model_pair,
                    'Transition': transition,
                    'Value': 0.5  # Default value as requested
                })
    
    if not similarity_data:
        plt.text(0.5, 0.5, "No similarity data available, using default values", 
                ha='center', va='center', fontsize=12)
        
        # Create default data for visualization
        default_models = ['Qwen->Llama', 'Llama->Qwen']
        default_transitions = ['Remain in Agreement', 'Agreement → Disagreement', 
                           'Disagreement → Agreement', 'Remain in Disagreement']
        
        for model in default_models:
            for transition in default_transitions:
                similarity_data.append({
                    'Model': model,
                    'Transition': transition,
                    'Value': 0.5  # Default value as requested
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(similarity_data)
    
    # Create grouped bar chart
    ax = sns.barplot(x='Transition', y='Value', hue='Model', data=df, palette='Set2')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')
    
    plt.xlabel("Transition Type")
    plt.ylabel("Similarity Score")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Model Direction")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

def main():
    # Define base path
    base_path = 'data/dataset'
    
    # Load results
    zero_shot_results = load_zero_shot_results(base_path)
    surrogate_results = load_surrogate_results(base_path, selection_strategy="similarity")
    
    # Create visualizations
    create_metric_dashboard(zero_shot_results, surrogate_results, selection_strategy="similarity")
    
    # Create agreement analysis plots
    create_agreement_analysis_plots(surrogate_results, selection_strategy="similarity")
    
    print("Visualizations generated successfully!")

if __name__ == "__main__":
    main()