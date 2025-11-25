import os, yaml, torch
from omegaconf import OmegaConf
from weighted_mtp.data.dataloader import create_dataloader

def load_config():
    cfg_path = os.path.join('configs','critic','critic_linear.yaml')
    with open(cfg_path,'r') as f:
        cfg = OmegaConf.load(f)
    return cfg

def main():
    cfg = load_config()
    # Build validation dataloader
    val_cfg = cfg.dataset
    # Use same sampling config for validation
    sampling_cfg = cfg.data_sampling
    # Create dataloader (the function expects config with dataset paths etc.)
    # The create_dataloader expects a config dict with dataset and sampling settings.
    # We'll construct a minimal config namespace.
    from weighted_mtp.data.dataloader import create_dataloader
    # Build a simple config object
    class SimpleConfig:
        pass
    cfg_obj = SimpleConfig()
    cfg_obj.dataset = cfg.dataset
    cfg_obj.data_sampling = cfg.data_sampling
    cfg_obj.runtime = cfg.runtime
    cfg_obj.distributed = cfg.distributed
    cfg_obj.training = cfg.training
    # Validation dataloader
    val_loader = create_dataloader(cfg_obj, split='validation')
    total = 0
    correct = 0
    difficulty_counts = {}
    for batch in val_loader:
        is_correct = batch['is_correct']
        total += is_correct.size(0)
        correct += is_correct.sum().item()
        # difficulty field may be present in metadata
        if 'difficulty' in batch:
            diffs = batch['difficulty']
            for d in diffs.tolist():
                difficulty_counts[d] = difficulty_counts.get(d,0)+1
    incorrect = total - correct
    print('Validation samples:', total)
    print('Correct ratio:', correct/total)
    print('Incorrect ratio:', incorrect/total)
    print('Difficulty distribution:', difficulty_counts)

if __name__=='__main__':
    main()
