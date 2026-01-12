import matplotlib.pyplot as plt

def group_seismic_events(event_type):
    """Group seismic events into meaningful categories"""
    event_lower = event_type.lower().strip()
    
    if event_lower == 'earthquake':
        return 'earthquake'
    
    elif any(word in event_lower for word in ['explosion', 'blast', 'nuclear', 'experimental']):
        return 'explosion'
    
    elif any(word in event_lower for word in ['ice', 'landslide', 'slide', 'sonic boom']):
        return 'natural_event'
    
    elif any(word in event_lower for word in ['mine', 'mining', 'quarry', 'rock burst']):
        return 'mining_activity'
    
    elif 'volcanic' in event_lower:
        return 'volcanic'
    
    else:
        return 'other'

def print_distribution(df, column_name):
    """Print the distribution of categories in a specified column of a DataFrame"""
    counts = df[column_name].value_counts()
    print(f"\nDistribution for '{column_name}':")
    for category, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"{category}: {count:,} ({pct:.2f}%)")

def save_binary_classification_dataset(df_balanced, output_path="../data/raw/seismic_data_binary.csv"):
    """
    Save dataset with binary target: 1 for earthquake, 0 for non-earthquake
    
    Args:
        df_balanced: DataFrame after downsampling
        output_path: Path to save the CSV file
    
    Returns:
        DataFrame with binary target column
    """
    print("="*60)
    print("CREATING BINARY CLASSIFICATION DATASET")
    print("="*60)
    
    # Create a copy to avoid modifying original
    df_binary = df_balanced.copy()
    
    # Create binary target: 1 for earthquake, 0 for non-earthquake
    df_binary['target'] = (df_binary['type_grouped'] == 'earthquake').astype(int)

    # Show distribution
    print("\nBinary Target Distribution:")
    target_counts = df_binary['target'].value_counts().sort_index()
    print(target_counts)
    
    print("\nDetailed breakdown:")
    print(f"  0 (non-earthquake): {target_counts[0]:,} samples ({target_counts[0]/len(df_binary)*100:.2f}%)")
    print(f"  1 (earthquake):     {target_counts[1]:,} samples ({target_counts[1]/len(df_binary)*100:.2f}%)")
    
    # Visualize binary distribution
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Non-Earthquake (0)', 'Earthquake (1)'], 
                   [target_counts[0], target_counts[1]],
                   color=['#3498db', '#e74c3c'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/len(df_binary)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.ylabel('Count', fontsize=12)
    plt.title('Binary Classification Target Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Save to CSV using polars
    import polars as pl
    
    print("\n" + "="*60)
    print("SAVING BINARY DATASET")
    print("="*60)
    

    pl.DataFrame(df_binary[["target", "mag", "depth", "nst", "gap", "longitude", "latitude", "id"]]).write_csv(output_path)
    
    print(f"✓ Binary dataset saved to: {output_path}")
    print(f"✓ Total rows: {len(df_binary):,}")
    print(f"✓ Columns: {list(df_binary.columns)}")
    print(f"✓ Target column 'target' added (0=non-earthquake, 1=earthquake)")
    
    # Show sample rows
    print("\n" + "="*60)
    print("SAMPLE DATA (first 5 rows)")
    print("="*60)
    print(df_binary[["target", "mag", "depth", "nst", "gap", "longitude", "latitude", "id"]].head())

    return df_binary

def preprocess_data(df):
    """Preprocess the seismic data for classification"""
    
    df_processed = df.copy()
    
    # Extract features and target (without 'id')
    X = df_processed[["mag", "depth", "nst", "gap", "longitude", "latitude"]].copy()
    y = df_processed['target'].copy()
    
    print("Handling missing values...")
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Remove any remaining rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Features used: {list(X.columns)}")
    print(f"Number of samples: {len(X)}")
    
    return X, y
