import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.metrics import f1_score
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support


def compute_exact_match(predictions, ground_truths):
    """
    Compute exact match score between predictions and ground truths
    with proper text normalization for seq2seq tasks
    """
    if not predictions or not ground_truths:
        return 0.0
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    def normalize_text(text):
        if text is None:
            return None
        # Convert to lowercase
        text = text.lower()
        # Remove leading/trailing whitespace
        text = text.strip()
        # Normalize whitespace between words
        text = " ".join(text.split())
        return text
    
    # Compare normalized versions
    matches = [1 if normalize_text(p) == normalize_text(g) and p is not None and g is not None else 0 
              for p, g in zip(predictions, ground_truths)]
    
    return sum(matches) / len(matches) if matches else 0.0

def compute_f1_score(predictions, ground_truths):
    """
    Compute token-level F1 score preserving token frequencies
    """
    if not predictions or not ground_truths:
        return 0.0
    
    def calculate_f1(pred, truth):
        if not pred or not truth:
            return 0.0
            
        # Tokenize and keep duplicates (using lists, not sets)
        pred_tokens = pred.lower().split()
        truth_tokens = truth.lower().split()
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        # Count token frequencies
        pred_counter = Counter(pred_tokens)
        truth_counter = Counter(truth_tokens)
        
        # Calculate intersection using counters
        # Takes the minimum count for each shared token
        true_positives = sum((pred_counter & truth_counter).values())
        
        # Handle empty predictions or truths
        if true_positives == 0:
            return 0.0
            
        precision = true_positives / len(pred_tokens)
        recall = true_positives / len(truth_tokens)
        
        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1
    
    # Calculate F1 for each pair
    f1_scores = [calculate_f1(pred, truth) 
                for pred, truth in zip(predictions, ground_truths)]
    
    return np.mean(f1_scores) if f1_scores else 0.0

def compute_classification_f1_from_rankings(predictions, ground_truths, options):
    """
    Compute classification F1 score by checking if the highest-ranked option
    matches the ground truth.
    
    Args:
        predictions: List of model predictions (text strings)
        ground_truths: List of ground truth answers (text strings)
        options: List of lists, where each inner list contains the options for a question
        
    Returns:
        F1 score based on whether the correct option was ranked highest
    """
    if not predictions or not ground_truths or not options:
        return 0.0
    
    # Load SBERT model
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Track correct classifications
    y_true = []
    y_pred = []
    
    for pred, truth, opts in zip(predictions, ground_truths, options):
        if not pred or not truth or not opts:
            continue
        
        # Get embeddings for prediction and options
        opts = eval(opts)
        pred_embedding = model.encode([pred], convert_to_numpy=True)
        opts_embeddings = model.encode(opts, convert_to_numpy=True)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(pred_embedding, opts_embeddings)[0]
        
        # Get the index of the highest-ranked option
        top_option_idx = np.argmax(similarities)
        predicted_option = opts[top_option_idx]
        
        # Find the ground truth in the options
        try:
            truth_idx = opts.index(truth)
            
            # Add to classification arrays
            y_true.append(truth_idx)
            y_pred.append(top_option_idx)
        except ValueError:
            # Ground truth not in options, skip this example
            continue
    
    if not y_true:
        return 0.0
    
    # Calculate F1 score (macro average for multi-class)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    
    return f1

def compute_rouge_scores(predictions, ground_truths):
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    """
    if not predictions or not ground_truths:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, truth in zip(predictions, ground_truths):
        # Handle None or empty strings
        if pred is None or truth is None or pred.strip() == "" or truth.strip() == "":
            for key in scores:
                scores[key].append(0.0)
            continue
            
        try:
            score = scorer.score(truth, pred)
            for key in scores:
                scores[key].append(score[key].fmeasure)
        except Exception as e:
            print(f"ROUGE calculation failed for '{pred}' and '{truth}': {e}")
            for key in scores:
                scores[key].append(0.0)
    
    return {k: np.mean(v) if v else 0.0 for k, v in scores.items()}

def compute_bleu_score(predictions, ground_truths):
    """
    Compute BLEU score for the predictions
    """
    if not predictions or not ground_truths:
        return 0.0
        
    smoother = SmoothingFunction().method1
    scores = []
    
    for pred, truth in zip(predictions, ground_truths):
        # Handle None or empty strings
        if pred is None or truth is None:
            scores.append(0.0)
            continue
            
        # Convert strings to lists of tokens
        pred_tokens = pred.split() if pred.strip() else []
        truth_tokens = truth.split() if truth.strip() else []
        
        # Skip empty references or hypotheses
        if not pred_tokens or not truth_tokens:
            scores.append(0.0)
            continue
            
        try:
            # Calculate BLEU score for this pair
            score = sentence_bleu([truth_tokens], pred_tokens, 
                                smoothing_function=smoother)
            scores.append(score)
        except Exception as e:
            print(f"BLEU calculation failed for '{pred}' and '{truth}': {e}")
            scores.append(0.0)
    
    return np.mean(scores) if scores else 0.0

def compute_sbert_similarity(predictions, ground_truths):
    """
    Compute cosine similarity using Sentence-BERT embeddings
    """
    if not predictions or not ground_truths:
        return 0.0
        
    # Filter out None values and empty strings
    valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths) 
                  if p is not None and g is not None and p.strip() and g.strip()]
    
    if not valid_pairs:
        return 0.0
        
    valid_preds, valid_truths = zip(*valid_pairs)
    
    try:
        # Load the model
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Generate embeddings
        pred_embeddings = model.encode(valid_preds)
        truth_embeddings = model.encode(valid_truths)
        
        # Compute cosine similarities
        similarities = []
        for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings):
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(pred_emb).unsqueeze(0),
                torch.tensor(truth_emb).unsqueeze(0)
            )
            similarities.append(similarity.item())
        
        return np.mean(similarities) if similarities else 0.0
    except Exception as e:
        print(f"SBERT similarity calculation failed: {e}")
        return 0.0
    
    
def compute_all_metrics(predictions, ground_truths, options):
    """
    Compute all metrics at once and return as a dictionary
    """
    metrics = {
        'exact_match': compute_exact_match(predictions, ground_truths),
        'f1_score_token_agreement': compute_f1_score(predictions, ground_truths),
        'f1_score_ranking': compute_classification_f1_from_rankings(predictions, ground_truths, options),
        'accuracy_ranking': compute_accuracy_from_rankings(predictions, ground_truths, options),
        'rouge_scores': compute_rouge_scores(predictions, ground_truths),
        'bleu_score': compute_bleu_score(predictions, ground_truths),
        'sbert_similarity': compute_sbert_similarity(predictions, ground_truths)
    }
    return metrics


def compute_metrics_from_csv(csv_filename):
    """
    Reads the CSV file and computes all metrics with error handling
    """
    try:
        df = pd.read_csv(csv_filename)
        
        # Handle missing values
        df['model_output'] = df['model_output'].fillna('')
        df['ground_truth'] = df['ground_truth'].fillna('')
        
        predictions = df['model_output'].tolist()
        ground_truths = df['ground_truth'].tolist()
        options = df['options'].tolist()
        
        return compute_all_metrics(predictions, ground_truths, options)
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return {
            'exact_match': 0.0,
            'f1_score': 0.0,
            'rouge_scores': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
            'bleu_score': 0.0,
            'sbert_similarity': 0.0
        }

def compute_accuracy_from_rankings(predictions, ground_truths, options):
    """
    Compute accuracy by checking if the highest-ranked option matches the ground truth.
    
    Args:
        predictions: List of model predictions (text strings)
        ground_truths: List of ground truth answers (text strings)
        options: List of lists, where each inner list contains the options for a question
        
    Returns:
        Accuracy score based on whether the correct option was ranked highest
    """
    if not predictions or not ground_truths or not options:
        return 0.0
    
    # Load SBERT model
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Track correct predictions
    correct_count = 0
    total_count = 0
    
    for pred, truth, opts in zip(predictions, ground_truths, options):
        if not pred or not truth or not opts:
            continue
        
        # Get embeddings for prediction and options
        opts = eval(opts)
        pred_embedding = model.encode([pred], convert_to_numpy=True)
        opts_embeddings = model.encode(opts, convert_to_numpy=True)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(pred_embedding, opts_embeddings)[0]
        
        # Get the index of the highest-ranked option
        top_option_idx = np.argmax(similarities)
        predicted_option = opts[top_option_idx]
        
        # Find the ground truth in the options
        try:
            truth_idx = opts.index(truth)
            
            # Increment counter if prediction matches ground truth
            if top_option_idx == truth_idx:
                correct_count += 1
            
            total_count += 1
        except ValueError:
            # Ground truth not in options, skip this example
            continue
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return accuracy