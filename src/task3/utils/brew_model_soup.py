"""
Ensemble Model Soup

Averages the weights of the 5 cross-validated Dual Encoder checkpoints 
into a single "Super Model" to maximize accuracy without increasing inference cost.
"""

import torch
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

def create_model_soup():
    ckpt_dir = _PROJECT_ROOT / "checkpoints/task3/phase3_fusion"
    fold_files = sorted(list(ckpt_dir.glob("phase3_atten_fusion_fold*_best.pt")))
    
    if len(fold_files) != 5:
        print(f"Error: Found {len(fold_files)} checkpoints. We need exactly 5 to brew the soup!")
        return
        
    print(f"Brewing Model Soup from {len(fold_files)} checkpoints...")
    for f in fold_files:
        print(f"  - Loading {f.name}")
    
    # Load the first checkpoint to serve as base template
    base_ckpt = torch.load(fold_files[0], map_location="cpu", weights_only=True)
    soup_state_dict = base_ckpt["state_dict"]
    
    # Add the weights from the remaining 4 models
    for file in fold_files[1:]:
        ckpt = torch.load(file, map_location="cpu", weights_only=True)
        for key in soup_state_dict.keys():
            soup_state_dict[key] += ckpt["state_dict"][key]
            
    # Divide by 5 to get the exact mathematical average
    for key in soup_state_dict.keys():
        # Handle integer tensors (like num_batches_tracked) safely
        if soup_state_dict[key].dtype in [torch.int64, torch.int32]:
            soup_state_dict[key] //= len(fold_files)
        else:
            soup_state_dict[key] /= len(fold_files)
            
    # Save the new Model
    base_ckpt["state_dict"] = soup_state_dict
    out_path = ckpt_dir / "phase3_atten_fusion_ensemble_soup.pt"
    
    # Save with standard settings so it can be loaded easily by the wrapper
    torch.save(base_ckpt, out_path)
    
    print("\n" + "="*50)
    print(f"Success! Ensemble Soup saved to: {out_path.name}")
    print("="*50)

if __name__ == "__main__":
    create_model_soup()