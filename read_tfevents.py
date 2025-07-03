import sys
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf

def get_hparams(tfevents_file):
    hparams = {}
    try:
        for summary in tf.compat.v1.train.summary_iterator(tfevents_file):
            for value in summary.summary.value:
                if value.tag == '_hparams_/session_start_info':
                    # The hparams are stored in a text summary.
                    # We need to parse the text to get the key-value pairs.
                    # This is not ideal, but it's what we have to work with.
                    
                    # The text is in Markdown format.
                    # We can split it by lines and then by the '|' character.
                    lines = value.tensor.string_val[0].decode('utf-8').split('\n')
                    for line in lines[2:]: # Skip the header
                        parts = [p.strip() for p in line.split('|') if p.strip()]
                        if len(parts) == 2:
                            hparams[parts[0]] = parts[1]

    except Exception as e:
        print(f"Error reading hparams from {tfevents_file}: {e}")
        
    return hparams


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python read_tfevents.py <tfevents_file_1> <tfevents_file_2>")
        sys.exit(1)
        
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    print(f"Hyperparameters for {file1}:")
    hparams1 = get_hparams(file1)
    print(hparams1)
    
    print(f"\nHyperparameters for {file2}:")
    hparams2 = get_hparams(file2)
    print(hparams2)
    
    # Compare the hyperparameters
    print("\n--- Comparison ---")
    all_keys = set(hparams1.keys()) | set(hparams2.keys())
    for key in sorted(all_keys):
        val1 = hparams1.get(key)
        val2 = hparams2.get(key)
        if val1 != val2:
            print(f"{key}: {val1} vs {val2}")