import json
import pandas as pd
from camel_tools.tokenizers.word import simple_word_tokenize
import re
import unicodedata

def lemmatize_arabic(text):
    """
    Tokenize Arabic text using CAMeL Tools simple word tokenizer.
    Note: For full lemmatization, you would need additional CAMeL Tools components.
    This uses simple tokenization as a baseline.
    """
    if pd.isna(text) or text == "" or text is None:
        return []
    text = str(text)
    tokens = simple_word_tokenize(text)
    # Convert to lowercase equivalent (normalize) and remove empty tokens
    return [token.strip().lower() for token in tokens if token.strip()]

def normalize_text_basic(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()

def normalize_token(token: str) -> str:
    token = normalize_text_basic(token)
    # very naive stemming, but enough for plurals like croissants -> croissant
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    es_plural_suffixes = ("ses", "shes", "ches", "xes", "zes")
    if token.endswith(es_plural_suffixes) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def tokenize_english(text):
    """
    Tokenize English text:
    - Remove accents
    - Extract word tokens
    - Normalize and lightly stem each token
    """
    if pd.isna(text) or text == "" or text is None:
        return []

    text = normalize_text_basic(str(text))
    tokens = re.findall(r'\b\w+\b', text)

    return [normalize_token(tok) for tok in tokens if tok.strip()]



def s_qc(y_prediction, all_annotated_answers, use_english=True):
    """
    Calculate s_{q,c}(y) as defined in equation (1).
    Returns 1 if prediction y includes any answer from annotated_answers, 0 otherwise.
    
    Args:
        y_prediction: The LLM prediction text (string)
        all_annotated_answers: List of all human annotated answers
        use_english: If True, use English tokenization; if False, use Arabic tokenization
    
    Returns:
        1 if ∃a ∈ A_q such that a ⊆ y, 0 otherwise
    """
    # Choose tokenization method
    if use_english:
        y_tokens = set(tokenize_english(y_prediction))
        tokenize_func = tokenize_english
    else:
        y_tokens = set(lemmatize_arabic(y_prediction))
        tokenize_func = lemmatize_arabic
    
    if not y_tokens:
        return 0

    # Check if any annotated answer is a subset of the prediction
    for answer in all_annotated_answers:
        answer_tokens = set(tokenize_func(answer))
        # Check if answer tokens are subset of prediction tokens
        if answer_tokens and answer_tokens.issubset(y_tokens):
            return 1
        if y_tokens.issubset(answer_tokens):
            return 1
    
    return 0




def extract_all_answers(annotations, use_english=True):
    """
    Extract all answers from the annotations list.
    
    Args:
        annotations: List of annotation dictionaries
        use_english: If True, extract 'en_answers'; if False, extract 'answers' (Arabic)
    
    Returns:
        List of all answers
    """
    all_answers = []
    answer_key = 'en_answers' if use_english else 'answers'
    
    for annotation in annotations:
        if answer_key in annotation:
            all_answers.extend(annotation[answer_key])
    return all_answers

def calculate_score_S(json_file_path, csv_file_path, response_column='answer', use_english=True):
    """
    Calculate the score S(c) for questions.
    
    S(c) = (1/|Q|) * Σ_{q∈Q} s_{q,c}(f_m(q,c)) × 100
    
    Args:
        json_file_path: Path to the JSON file containing annotations (e.g., 'Algeria_data.json')
        csv_file_path: Path to the CSV file containing predictions (e.g., 'gpt-4-1106-preview-Algeria_Arabic_result.csv')
        response_column: Name of the column containing the model predictions (default: 'response')
        use_english: If True, match against 'en_answers'; if False, match against 'answers' (Arabic)
    
    Returns:
        Tuple of (overall_score, detailed_results_dataframe)
    """
    # Load JSON data with annotations
    with open(json_file_path, 'r', encoding='utf-8') as f:
        annotations_data = json.load(f)
    
    # Load CSV data with predictions
    predictions_df = pd.read_csv(csv_file_path)
    
    # Ensure the response column exists
    if response_column not in predictions_df.columns:
        raise ValueError(f"Column '{response_column}' not found in CSV. Available columns: {predictions_df.columns.tolist()}")
    
    # Get question IDs from both sources
    # Assuming CSV has an ID column or similar identifier
    if 'ID' in predictions_df.columns:
        id_column = 'ID'
    elif 'id' in predictions_df.columns:
        id_column = 'id'
    elif 'question_id' in predictions_df.columns:
        id_column = 'question_id'
    else:
        # Use index as ID
        predictions_df['ID'] = predictions_df.index
        id_column = 'ID'
        print(f"Warning: No ID column found. Using row index as ID.")
    
    total_questions = 0
    total_score = 0
    results = []
    
    language_type = "English (en_answers)" if use_english else "Arabic (answers)"
    print(f"Matching predictions against: {language_type}")
    
    # For each question in the predictions
    for idx, row in predictions_df.iterrows():
        question_id = str(row[id_column])
        y_prediction = str(row[response_column])
        
        # Check if this question has annotations
        if question_id not in annotations_data:
            print(f"Warning: Question ID '{question_id}' not found in annotations JSON. Skipping.")
            continue
        
        # Extract all answers from annotations (English or Arabic based on parameter)
        annotations = annotations_data[question_id].get('annotations', [])
        all_annotated_answers = extract_all_answers(annotations, use_english=use_english)
        
        if not all_annotated_answers:
            print(f"Warning: No annotated answers found for question '{question_id}'. Skipping.")
            continue
        
        # Calculate s_{q,c}(f_m(q,c))
        score_qc = s_qc(y_prediction, all_annotated_answers, use_english=use_english)
        total_score += score_qc
        total_questions += 1
        
        # Store result for this question
        results.append({
            'question_id': question_id,
            'score': score_qc,
            'num_annotated_answers': len(all_annotated_answers),
            'annotated_answers': ', '.join(all_annotated_answers[:3]) + ('...' if len(all_annotated_answers) > 3 else ''),
            'prediction_preview': y_prediction[:100] + '...' if len(y_prediction) > 100 else y_prediction
        })
    
    if total_questions == 0:
        print("Error: No valid questions found for evaluation.")
        return 0.0, pd.DataFrame()
    
    # Calculate final score
    S_c = (total_score / total_questions) * 100
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    return S_c, results_df


# Example usage:
if __name__ == "__main__":
    # File paths - adjust these to your actual file paths
    json_file = 'Algeria_data.json'
    csv_file = 'gpt-4-Algeria_result.csv'
    
    print("Calculating Score S(c) for predictions...")
    print("=" * 60)
    
    try:
        # Calculate the score - NOW USING ENGLISH ANSWERS (en_answers)
        overall_score, detailed_results = calculate_score_S(
            json_file_path=json_file,
            csv_file_path=csv_file,
            response_column='answer',  # Change this if your column has a different name
            use_english=True  # Set to True to use en_answers, False to use Arabic answers
        )
        
        # Print overall score
        print(f"\n{'='*60}")
        print(f"Overall Score S(c): {overall_score:.2f}%")
        print(f"{'='*60}\n")
        
        # Print summary statistics
        if not detailed_results.empty:
            num_correct = detailed_results['score'].sum()
            num_total = len(detailed_results)
            print(f"Questions with matching answers: {num_correct}/{num_total}")
            print(f"\nDetailed results:")
            print(detailed_results.to_string(index=False))
            
            # Save detailed results to CSV
            output_file = 'evaluation_results.csv'
            detailed_results.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\nDetailed results saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("\nPlease ensure the following files exist:")
        print(f"  1. {json_file} (contains annotations)")
        print(f"  2. {csv_file} (contains predictions)")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Alternative: If you want to specify different column names
    # overall_score, detailed_results = calculate_score_S(
    #     json_file_path='Algeria_data.json',
    #     csv_file_path='gpt-4-1106-preview-Algeria_Arabic_result.csv',
    #     response_column='generated_text'  # or whatever your column is called
    # )



    #json_file = './data/annotations/Algeria_data.json'
    #csv_file = './model_inference_results/gpt-4-1106-preview-Algeria_Arabic_result.csv'