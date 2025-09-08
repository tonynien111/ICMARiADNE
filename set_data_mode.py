import sys
import re

def set_data_mode(mode):
    if mode not in ['train', 'val', 'test']:
        print(f"Error: Mode must be 'train', 'val', or 'test', not '{mode}'")
        return False
    
    # Read parameter.py
    try:
        with open('parameter.py', 'r') as f:
            content = f.read()
        
        # Replace the DATA_SPLIT_MODE line
        pattern = r"DATA_SPLIT_MODE\s*=\s*['\"][^'\"]*['\"]"
        replacement = f"DATA_SPLIT_MODE = '{mode}'"
        
        new_content = re.sub(pattern, replacement, content)
        
        # Write back to file
        with open('parameter.py', 'w') as f:
            f.write(new_content)
        
        print(f"✅ Successfully set DATA_SPLIT_MODE to '{mode}'")
        
        # Show current split sizes
        import parameter
        # Reload the module to get updated values
        import importlib
        importlib.reload(parameter)
        
        return True
        
    except Exception as e:
        print(f"Error updating parameter.py: {e}")
        return False

def show_current_mode():
    try:
        from parameter import DATA_SPLIT_MODE, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
        import os
        
        # Count maps
        map_dir = 'maps'
        if os.path.exists(map_dir):
            map_list = [f for f in os.listdir(map_dir) if f.endswith('.png')]
            total_maps = len(map_list)
            
            train_size = int(TRAIN_SPLIT * total_maps)
            val_size = int(VAL_SPLIT * total_maps) 
            test_size = total_maps - train_size - val_size
            
            print(f"Current mode: {DATA_SPLIT_MODE}")
            print(f"Dataset sizes:")
            print(f"  Train: {train_size} maps ({TRAIN_SPLIT*100:.1f}%)")
            print(f"  Val:   {val_size} maps ({VAL_SPLIT*100:.1f}%)")
            print(f"  Test:  {test_size} maps ({(1-TRAIN_SPLIT-VAL_SPLIT)*100:.1f}%)")
            print(f"  Total: {total_maps} maps")
        else:
            print(f"Current mode: {DATA_SPLIT_MODE}")
            print("Maps directory not found")
            
    except ImportError as e:
        print(f"error reading parameters: {e}")

def main():
    if len(sys.argv) == 1:
        print("Data Split Mode Utility")
        print("=" * 30)
        show_current_mode()
        print("\nUsage:")
        print("  python set_data_mode.py train   # Set to training mode")
        print("  python set_data_mode.py val     # Set to validation mode") 
        print("  python set_data_mode.py test    # Set to test mode")
        print("  python set_data_mode.py status  # Show current status")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == 'status':
        show_current_mode()
    elif mode in ['train', 'val', 'test']:
        success = set_data_mode(mode)
        if success:
            print("\nCurrent status:")
            show_current_mode()
    else:
        print(f"❌ Invalid mode: {mode}")
        print("Valid modes: train, val, test, status")

if __name__ == "__main__":
    main()