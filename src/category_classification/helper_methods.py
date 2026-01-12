import matplotlib.pyplot as plt
import polars as pl

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

def save_classification_dataset(df_balanced, output_path="../data/raw/final_seismic_dataset.csv"):
    """
    Save dataset with both binary target and multiclass target for seismic event classification.
    
    Args:
        df_balanced: DataFrame after downsampling
        output_path: Path to save the CSV file
    
    Returns:
        DataFrame with binary target and multiclass target columns
    """
    print("="*60)
    print("CREATING CLASSIFICATION DATASET")
    print("="*60)
    
    df_classification = df_balanced.copy()
    
    df_classification['target_binary'] = (df_classification['type_grouped'] == 'earthquake').astype(int)
    
    print("\n" + "-"*60)
    print("BINARY TARGET DISTRIBUTION")
    print("-"*60)
    binary_counts = df_classification['target_binary'].value_counts().sort_index()
    print(binary_counts)
    
    print("\nDetailed breakdown:")
    print(f"  0 (non-earthquake): {binary_counts[0]:,} samples ({binary_counts[0]/len(df_classification)*100:.2f}%)")
    print(f"  1 (earthquake):     {binary_counts[1]:,} samples ({binary_counts[1]/len(df_classification)*100:.2f}%)")
    
    class_mapping = {
        'earthquake': 0,
        'explosion': 1,
        'natural_event': 2,
        'mining_activity': 3,
        'other': 4,
        'volcanic': 5
    }
    
    df_classification['target_multiclass'] = df_classification['type_grouped'].map(class_mapping)
    
    print("\n" + "-"*60)
    print("MULTICLASS TARGET DISTRIBUTION")
    print("-"*60)
    multiclass_counts = df_classification.groupby('type_grouped')['target_multiclass'].agg(['count', 'first']).sort_values('first')
    multiclass_counts.columns = ['count', 'class_id']
    print(multiclass_counts)
    
    print("\nDetailed breakdown:")
    for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
        count = (df_classification['target_multiclass'] == class_id).sum()
        percentage = count / len(df_classification) * 100
        print(f"  {class_id} ({class_name}): {count:,} samples ({percentage:.2f}%)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    bars1 = ax1.bar(['Non-Earthquake (0)', 'Earthquake (1)'], 
                    [binary_counts[0], binary_counts[1]],
                    color=['#3498db', '#e74c3c'])
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/len(df_classification)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Binary Classification Target Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
    class_counts_sorted = [multiclass_counts.loc[name, 'count'] for name in class_names]
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22']
    bars2 = ax2.bar(class_names, class_counts_sorted, color=colors)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/len(df_classification)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Event Type', fontsize=12, fontweight='bold')
    ax2.set_title('Multi-Class Classification Target Distribution', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("SAVING DATASET")
    print("="*60)
    
    columns_to_save = [
        "target_binary",
        "target_multiclass",
        "type_grouped",
        "mag", 
        "depth", 
        "nst", 
        "gap", 
        "longitude", 
        "latitude", 
        "id"
    ]
    
    pl.DataFrame(df_classification[columns_to_save]).write_csv(output_path)
    
    print(f"Dataset saved to: {output_path}")
    print(f"Total rows: {len(df_classification):,}")
    print(f"\nColumns saved:")
    print(f"  - target_binary: Binary classification (0=non-earthquake, 1=earthquake)")
    print(f"  - target_multiclass: Multi-class classification (0-5)")
    print(f"  - type_grouped: Original event type names")
    print(f"  - Features: mag, depth, nst, gap, longitude, latitude")
    print(f"  - id: Sample identifier")
    
    print("\n" + "="*60)
    print("SAMPLE DATA (first 10 rows)")
    print("="*60)
    sample_df = df_classification[columns_to_save].head(5)
    print(sample_df.to_string(index=False))
    
    print("="*60)
    print("Multi-class target encoding:")
    for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
        print(f"  {class_id} -> {class_name}")
    
    return df_classification


def preprocess_data(df, multiclass=False):
    """Preprocess the seismic data for classification"""
    
    df_processed = df.copy()
    
    # Extract features and target (without 'id')
    X = df_processed[["mag", "depth", "nst", "gap", "longitude", "latitude"]].copy()
    if multiclass:
        y = df_processed['target_multiclass'].copy()
    else:
        y = df_processed['target_binary'].copy()
    
    print("Handling missing values...")
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Remove NaNs
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Features used: {list(X.columns)}")
    print(f"Number of samples: {len(X)}")
    
    return X, y
