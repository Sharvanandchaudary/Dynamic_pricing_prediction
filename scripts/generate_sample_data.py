import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate timestamps for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Generate data
    data = {
        'timestamp': timestamps,
        'product_id': [f'PROD_{i:04d}' for i in np.random.randint(1, 101, n_samples)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_samples),
        'base_price': np.random.uniform(10, 1000, n_samples),
        'inventory_level': np.random.randint(0, 200, n_samples),
        'demand_last_24h': np.random.randint(0, 100, n_samples),
        'competitor_price': None,  # Will be filled based on base_price
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
        'day_of_week': None,  # Will be filled from timestamp
        'time_of_day': None,  # Will be filled from timestamp
        'special_event': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # 10% chance of special event
        'customer_segment': np.random.choice(['Budget', 'Regular', 'Premium'], n_samples),
        'rating': np.random.uniform(1, 5, n_samples).round(1),
        'historical_sales': np.random.randint(0, 1000, n_samples),
        'final_price': None  # Will be calculated
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add competitor prices (randomly distributed around base price)
    df['competitor_price'] = df['base_price'] * np.random.uniform(0.8, 1.2, n_samples)
    
    # Add day of week and time of day
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['time_of_day'] = df['timestamp'].dt.hour
    
    # Calculate final price based on various factors
    df['final_price'] = df.apply(calculate_final_price, axis=1)
    
    # Round monetary values to 2 decimal places
    df['base_price'] = df['base_price'].round(2)
    df['competitor_price'] = df['competitor_price'].round(2)
    df['final_price'] = df['final_price'].round(2)
    
    # Save to CSV
    output_path = 'data/raw/dynamic_pricing_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Sample data generated and saved to {output_path}")
    
    return df

def calculate_final_price(row):
    base_price = row['base_price']
    
    # Inventory adjustment factor (lower inventory = higher price)
    inventory_factor = 1 + (0.1 * (1 - row['inventory_level']/200))
    
    # Demand adjustment factor (higher demand = higher price)
    demand_factor = 1 + (0.15 * row['demand_last_24h']/100)
    
    # Competitor price adjustment
    competitor_factor = 0.95 if row['competitor_price'] < base_price else 1.05
    
    # Special event adjustment
    event_factor = 1.2 if row['special_event'] else 1.0
    
    # Customer segment adjustment
    segment_factor = {
        'Budget': 0.9,
        'Regular': 1.0,
        'Premium': 1.15
    }[row['customer_segment']]
    
    # Calculate final price
    final_price = base_price * inventory_factor * demand_factor * competitor_factor * event_factor * segment_factor
    
    # Ensure price doesn't go below 50% or above 150% of base price
    final_price = max(base_price * 0.5, min(base_price * 1.5, final_price))
    
    return final_price

if __name__ == "__main__":
    df = generate_sample_data()
    print("\nFirst few rows of generated data:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nData summary:")
    print(df.describe()) 