import re
import ast

def levenshtein_distance(seq1, seq2):
    """
    Computes Levenshtein edit distance between two sequences (lists or strings).
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]
    
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y
        
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = matrix[x-1][y-1]
            else:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,    # deletion
                    matrix[x][y-1] + 1,    # insertion
                    matrix[x-1][y-1] + 1   # substitution
                )
    return matrix[size_x-1][size_y-1]

def calculate_wer(ref_text, hyp_text):
    """
    Calculates Word Error Rate (WER) between reference and hypothesis texts.
    """
    ref_words = ref_text.lower().split()
    hyp_words = hyp_text.lower().split()
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0
        
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)

def parse_entity_frame(frame_str):
    """
    Safely parses the semantic frame JSON-like string into a sorted list of entity tuples.
    Example: "{'action': 'air'| 'entities': [{'type': 'city'| 'filler': 'los angeles'}]}"
    Returns:
        [('action', 'air'), ('city', 'los angeles')]
    """
    # Standardize string by replacing | with ,
    cleaned = frame_str.replace('|', ',').strip()
    
    tuples = []
    try:
        # Safe evaluation of dict-like text
        frame_dict = ast.literal_eval(cleaned)
        
        # 1. Action
        if 'action' in frame_dict:
            tuples.append(('action', str(frame_dict['action']).strip().lower()))
            
        # 2. Entities
        if 'entities' in frame_dict:
            for ent in frame_dict['entities']:
                ent_type = str(ent.get('type', '')).strip().lower()
                ent_filler = str(ent.get('filler', '')).strip().lower()
                if ent_type or ent_filler:
                    tuples.append((ent_type, ent_filler))
    except Exception:
        # Fallback to regex parser if literal_eval fails due to syntax anomalies
        action_match = re.search(r"'action':\s*'([^']*)'", cleaned)
        if action_match:
            tuples.append(('action', action_match.group(1).lower()))
            
        # Parse all entity blocks
        ent_blocks = re.findall(r"\{'type':\s*'([^']*)',\s*'filler':\s*'([^']*)'\}", cleaned)
        for ent_type, ent_filler in ent_blocks:
            tuples.append((ent_type.lower(), ent_filler.lower()))
            
    # Sort for consistent evaluation matching
    return sorted(tuples)

def calculate_seer(ref_frame_str, hyp_frame_str):
    """
    Calculates Single Entity Error Rate (SEER) between reference and hypothesis frames.
    """
    ref_tuples = parse_entity_frame(ref_frame_str)
    hyp_tuples = parse_entity_frame(hyp_frame_str)
    
    if not ref_tuples:
        return 1.0 if hyp_tuples else 0.0
        
    dist = levenshtein_distance(ref_tuples, hyp_tuples)
    # Clip error rate at 1.0 maximum
    return min(1.0, dist / len(ref_tuples))

def calculate_ser(ref_frame_str, hyp_frame_str):
    """
    Calculates Sentence Error Rate (SER). 1.0 if the predicted entities 
    do not match the reference exactly, 0.0 otherwise.
    """
    ref_tuples = parse_entity_frame(ref_frame_str)
    hyp_tuples = parse_entity_frame(hyp_frame_str)
    
    return 0.0 if ref_tuples == hyp_tuples else 1.0
