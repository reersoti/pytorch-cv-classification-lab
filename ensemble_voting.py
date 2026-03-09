import pandas as pd
from collections import Counter

def main():
    # Define files and weights
    submissions = [
        ("submission_effb2_dense.csv", 3),      # Best model (0.909)
        ("submission_effb1_dense.csv", 2),      # Second best (0.902)
        ("submission_effnet_b0_dense.csv", 1)   # Third best (0.902)
    ]
    
    print("Loading submissions...")
    dfs = []
    for filename, weight in submissions:
        try:
            df = pd.read_csv(filename)
            # Add weight column for internal processing
            df['weight'] = weight
            dfs.append(df)
            print(f"Loaded {filename} with weight {weight}")
        except FileNotFoundError:
            print(f"Warning: {filename} not found! Skipping.")
            
    if not dfs:
        print("No submissions loaded!")
        return

    # Assuming all files have the same 'id' order/set. 
    # Let's verify by merging or just iterating. Iterating is safer for simple scripts.
    
    # Create a dictionary to store votes: id -> {label: weighted_count}
    final_preds = []
    
    # We'll use the first dataframe as the base for IDs
    base_df = dfs[0]
    ids = base_df['id'].tolist()
    
    print("Voting...")
    for i, img_id in enumerate(ids):
        votes = Counter()
        
        for df in dfs:
            # Get label for this specific ID
            # Assuming sorted by ID as per our training scripts, but robust lookup is better
            row = df[df['id'] == img_id]
            if not row.empty:
                label = row.iloc[0]['label']
                weight = df['weight'].iloc[0] # All rows in this df have same weight
                votes[label] += weight
        
        # Get the label with the most votes
        best_label, _ = votes.most_common(1)[0]
        final_preds.append({"id": img_id, "label": best_label})
        
    # Save result
    sub = pd.DataFrame(final_preds).sort_values("id")
    output_file = "submission_ensemble_dense_trio.csv"
    sub.to_csv(output_file, index=False)
    print(f"Done! Ensemble saved to {output_file}")

if __name__ == "__main__":
    main()
