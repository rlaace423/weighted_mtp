
import json
import collections
import os

def analyze_metadata():
    file_path = "/Users/wesley/Desktop/wooshikwon/weighted_mtp/storage/datasets/codecontests/processed/train_metadata.json"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading metadata from {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            metadata = data.get('metadata', [])
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    stats = collections.defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    
    print(f"Processing {len(metadata)} items...")
    for item in metadata:
        difficulty = item.get('difficulty', 'Unknown')
        is_correct = item.get('is_correct', False)
        
        if is_correct:
            stats[difficulty]['correct'] += 1
        else:
            stats[difficulty]['incorrect'] += 1

    # Sort by difficulty (assuming integer or string that sorts nicely)
    # Try to convert to int for sorting if possible
    def sort_key(k):
        try:
            return int(k)
        except:
            return float('inf')

    sorted_difficulties = sorted(stats.keys(), key=sort_key)

    print("\n| Difficulty | Correct | Incorrect | Total | Correct Ratio (%) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    total_correct = 0
    total_incorrect = 0

    for diff in sorted_difficulties:
        c = stats[diff]['correct']
        i = stats[diff]['incorrect']
        t = c + i
        ratio = (c / t * 100) if t > 0 else 0
        
        print(f"| {diff} | {c:,} | {i:,} | {t:,} | {ratio:.2f}% |")
        
        total_correct += c
        total_incorrect += i

    grand_total = total_correct + total_incorrect
    grand_ratio = (total_correct / grand_total * 100) if grand_total > 0 else 0
    print(f"| **Total** | **{total_correct:,}** | **{total_incorrect:,}** | **{grand_total:,}** | **{grand_ratio:.2f}%** |")

if __name__ == "__main__":
    analyze_metadata()
