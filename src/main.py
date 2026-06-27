import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import pickle
import torch
import numpy as np
from src.pipeline.dataset import load_splits, get_stag_bifurcation
from src.pipeline.features import extract_spectrogram
from src.models.slu_dnn import CharacterTokenizer, SLUModel
from src.training.train_upscaler import train_stag_upscaler
from src.training.train_slu import train_kd_pipeline
from src.evaluation.metrics import calculate_wer, calculate_seer, calculate_ser
from src.evaluation.attacks import trace_recovery_simulation, estimate_home_address, aggregate_city_probabilities

def run_dryrun(meta_file, data_root, save_dir):
    print("==================================================")
    # 1. Train the STAG upscaler (LightGBM) on a tiny subset
    print("Step 1: Training the STAG Upscaler (LightGBM)...")
    upscaler_model_path = os.path.join(save_dir, "upscaler.pkl")
    upscaler = train_stag_upscaler(meta_file, data_root, upscaler_model_path, max_samples=20)
    
    # 2. Train the SLU model on a tiny subset using Knowledge Distillation
    print("\nStep 2: Training the SLU models via Knowledge Distillation...")
    train_kd_pipeline(
        metadata_file=meta_file,
        dataset_root=data_root,
        upscaler_path=upscaler_model_path,
        save_dir=save_dir,
        epochs=1,
        batch_size=2
    )
    
    # 3. Evaluate the models
    print("\nStep 3: Running Evaluation & Metrics Validation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CharacterTokenizer()
    
    student_path = os.path.join(save_dir, "student_model.pt")
    student_model = SLUModel(vocab_size=tokenizer.vocab_size).to(device)
    student_model.load_state_dict(torch.load(student_path, map_location=device))
    student_model.eval()
    
    _, _, test_rows = load_splits(meta_file)
    test_rows = test_rows[:10] # evaluate 10 samples
    
    wers, seers, sers = [], [], []
    
    for row in test_rows:
        uuid = row[0]
        duration = float(row[1])
        wav_path_rel = row[2]
        semantic_frame = row[3]
        
        base_dir = os.path.dirname(wav_path_rel)
        acc_path = os.path.join(data_root, base_dir.replace('./', ''), f"{uuid}.acc")
        gyro_path = os.path.join(data_root, base_dir.replace('./', ''), f"{uuid}.gyro")
        
        if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
            continue
            
        try:
            acc_odd, gyro_even, acc_even_target, t_even, t_odd = get_stag_bifurcation(
                acc_path, gyro_path, duration
            )
            # Upscale accelerometer Z-axis to 400 Hz
            acc_z_400 = upscaler.reconstruct_signal(acc_odd, gyro_even, t_odd, t_even)
            
            # Extract STFT spectrogram features
            imu_spec = extract_spectrogram(acc_z_400, fs_source=400, fs_target=500, n_bins=30)
            
            # Pad
            max_frames = 300
            if imu_spec.shape[0] > max_frames:
                imu_spec = imu_spec[:max_frames, :]
            else:
                padding = np.zeros((max_frames - imu_spec.shape[0], imu_spec.shape[1]))
                imu_spec = np.vstack([imu_spec, padding])
                
            input_tensor = torch.FloatTensor(imu_spec).unsqueeze(0).to(device) # Shape: (1, 300, 30)
            
            # Decode
            pred_tokens = student_model.predict(input_tensor, sos_id=tokenizer.sos_id, eos_id=tokenizer.eos_id)
            pred_text = tokenizer.decode(pred_tokens[0].tolist())
            
            ref_text = semantic_frame
            
            # Metrics
            wer = calculate_wer(ref_text, pred_text)
            seer = calculate_seer(ref_text, pred_text)
            ser = calculate_ser(ref_text, pred_text)
            
            wers.append(wer)
            seers.append(seer)
            sers.append(ser)
            
            print(f"\nSample: {uuid}")
            print(f"  Target  : {ref_text}")
            print(f"  Predicted: {pred_text}")
            print(f"  Metrics  : WER={wer:.4f}, SEER={seer:.4f}, SER={ser:.4f}")
        except Exception as e:
            continue
            
    if wers:
        print("\n==================================================")
        print("--- Final Eavesdropping Pipeline Performance ---")
        print(f"Average Sentence Error Rate (SER)     : {np.mean(sers)*100:.2f}%")
        print(f"Average Single Entity Error Rate (SEER): {np.mean(seers)*100:.2f}%")
        print(f"Average Word Error Rate (WER)         : {np.mean(wers)*100:.2f}%")
        print("==================================================")
        
    # 4. Downstream attack simulations
    print("\nStep 4: Simulating Downstream Attacks...")
    # City aggregation simulation
    city_probs = [
        {'new york': 0.8, 'chicago': 0.1},
        {'new york': 0.9, 'chicago': 0.05},
        {'new york': 0.95, 'chicago': 0.02}
    ]
    best_city, agg = aggregate_city_probabilities(city_probs)
    print(f"Aggregated city location: {best_city} (confidence: {agg[best_city]:.4f})")
    
    # Home address centroid simulation
    coords = [
        [-73.9718, 40.7579],
        [-73.9719, 40.7580],
        [-73.9717, 40.7578],
        [-73.9720, 40.7581],
        [-74.5000, 41.2000] # outlier
    ]
    home = estimate_home_address(coords)
    print(f"Estimated Home Address Centroid (excluding outlier): {home}")
    
    print("\nDryrun successfully completed!")

def main():
    parser = argparse.ArgumentParser(description="STAG and StealthyIMU Pipeline Orchestrator")
    parser.add_argument("--mode", type=str, default="dryrun", choices=["train_upscaler", "train_slu", "evaluate", "dryrun"],
                        help="Execution mode")
    parser.add_argument("--meta", type=str, default="StealthyIMU_dataset/metadata/stealthyIMU_all_relative.csv",
                        help="Path to metadata relative CSV file")
    parser.add_argument("--data_root", type=str, default="StealthyIMU_dataset",
                        help="Path to dataset root folder")
    parser.add_argument("--save_dir", type=str, default="models",
                        help="Folder to save trained models")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to load (default: None for all)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train SLU model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    args = parser.parse_args()
    
    # Resolve absolute paths
    meta_abs = os.path.abspath(args.meta)
    data_root_abs = os.path.abspath(args.data_root)
    save_dir_abs = os.path.abspath(args.save_dir)
    
    if args.mode == "dryrun":
        run_run_max_samples = 150 # dryrun has fixed small sizes inside run_dryrun
        run_dryrun(meta_abs, data_root_abs, save_dir_abs)
    else:
        print(f"Running mode: {args.mode}")
        if args.mode == "train_upscaler":
            train_stag_upscaler(meta_abs, data_root_abs, os.path.join(save_dir_abs, "upscaler.pkl"), max_samples=args.max_samples)
        elif args.mode == "train_slu":
            train_kd_pipeline(meta_abs, data_root_abs, os.path.join(save_dir_abs, "upscaler.pkl"), save_dir_abs,
                              epochs=args.epochs, batch_size=args.batch_size, max_samples=args.max_samples)

if __name__ == "__main__":
    main()
