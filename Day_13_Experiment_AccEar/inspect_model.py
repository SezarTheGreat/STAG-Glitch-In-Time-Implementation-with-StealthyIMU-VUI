import torch

checkpoint_path = r"C:\Users\jyoti\OneDrive\Desktop\STAG Implementation with StealthyIMU VUI\Day_13_Experiment_AccEar\checkpoints\accear_cgan_best_model.pt"

try:
    data = torch.load(checkpoint_path, map_location='cpu')
    print("Loaded successfully!")
    g_state = data['generator_state_dict']
    for k, v in g_state.items():
        print(f"{k}: {v.shape}")
except Exception as e:
    print(f"Error loading: {e}")
