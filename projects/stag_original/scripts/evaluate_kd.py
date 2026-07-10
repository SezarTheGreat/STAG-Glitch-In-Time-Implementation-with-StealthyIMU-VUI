import os
import torch
import jiwer
from torch.utils.data import DataLoader
from src.training.train_slu import CachedKDDataset
from src.models.slu_dnn import CharacterTokenizer, PaperSLUModel

def evaluate_models(save_dir="c:/Users/jyoti/OneDrive/Desktop/STAG Implementation with StealthyIMU VUI/models"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}...")
    
    tokenizer = CharacterTokenizer()
    
    # Load dataset
    test_dataset = CachedKDDataset(os.path.join(save_dir, "test_data.pt"))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Load models
    teacher_model = PaperSLUModel(vocab_size=tokenizer.vocab_size).to(device)
    student_model = PaperSLUModel(vocab_size=tokenizer.vocab_size).to(device)
    
    teacher_model.load_state_dict(torch.load(os.path.join(save_dir, "teacher_model.pt"), map_location=device))
    student_model.load_state_dict(torch.load(os.path.join(save_dir, "student_model.pt"), map_location=device))
    
    teacher_model.eval()
    student_model.eval()
    
    print("Building trie from unique targets in test set...")
    valid_sequences = []
    all_targets_str = []
    for t in test_dataset.target_tokens:
        t_list = t.tolist()
        try:
            # find eos
            eos_idx = t_list.index(tokenizer.eos_id)
            t_list = t_list[:eos_idx+1]
        except ValueError:
            pass
        if t_list not in valid_sequences:
            valid_sequences.append(t_list)
            
        all_targets_str.append(tokenizer.decode(t_list))
    
    valid_sequences_str = list(set(all_targets_str))
    print(f"Found {len(valid_sequences_str)} unique valid commands for the Trie constraint.")

    print("\nEvaluating Teacher Model (Audio)...")
    teacher_preds = []
    targets_str = []
    
    for batch in test_loader:
        speech_feat = batch['speech_feat'].to(device)
        targets = batch['target_tokens'].to(device)
        
        preds = teacher_model.predict(speech_feat, max_len=targets.size(1), 
                                      sos_id=tokenizer.sos_id, eos_id=tokenizer.eos_id,
                                      valid_sequences=valid_sequences)
                                      
        for i in range(targets.size(0)):
            pred_str = tokenizer.decode(preds[i].tolist())
            targ_str = tokenizer.decode(targets[i].tolist())
            teacher_preds.append(pred_str)
            targets_str.append(targ_str)

    teacher_wer = jiwer.wer(targets_str, teacher_preds)
    print(f"Teacher Model (Audio) WER: {teacher_wer * 100:.2f}%")
    
    print("\nEvaluating Student Model (IMU)...")
    student_preds = []
    for batch in test_loader:
        imu_feat = batch['imu_feat'].to(device)
        
        preds = student_model.predict(imu_feat, max_len=batch['target_tokens'].size(1), 
                                      sos_id=tokenizer.sos_id, eos_id=tokenizer.eos_id,
                                      valid_sequences=valid_sequences)
                                      
        for i in range(batch['target_tokens'].size(0)):
            pred_str = tokenizer.decode(preds[i].tolist())
            student_preds.append(pred_str)

    student_wer = jiwer.wer(targets_str, student_preds)
    print(f"Student Model (IMU) WER: {student_wer * 100:.2f}%")
    
    with open(os.path.join(save_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Teacher Model (Audio) WER: {teacher_wer * 100:.2f}%\n")
        f.write(f"Student Model (IMU) WER: {student_wer * 100:.2f}%\n")

if __name__ == "__main__":
    evaluate_models()
