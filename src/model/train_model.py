import logging
from .price_model import PricingModel
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize model
    logger.info("Initializing pricing model...")
    model = PricingModel()
    
    # Train models
    logger.info("Training models...")
    results = model.train_models('data/raw/dynamic_pricing_data.csv')
    
    # Print results
    logger.info("\nModel Evaluation Results:")
    logger.info("\nRandom Forest Results:")
    for metric, value in results['random_forest'].items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    
    logger.info("\nXGBoost Results:")
    for metric, value in results['xgboost'].items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    
    # Save models
    logger.info("\nSaving models...")
    model.save_models()
    
    # Test prediction
    sample_features = {
        'category': 'Electronics',
        'base_price': 500.0,
        'inventory_level': 50,
        'demand_last_24h': 20,
        'competitor_price': 550.0,
        'season': 'Summer',
        'day_of_week': 'Monday',
        'time_of_day': 14,
        'special_event': 0,
        'customer_segment': 'Regular',
        'rating': 4.5,
        'historical_sales': 300
    }
    
    # Make predictions using both models
    rf_price = model.predict_price(sample_features, 'random_forest')
    xgb_price = model.predict_price(sample_features, 'xgboost')
    
    logger.info("\nSample Prediction Results:")
    logger.info(f"Random Forest Predicted Price: ${rf_price:.2f}")
    logger.info(f"XGBoost Predicted Price: ${xgb_price:.2f}")

if __name__ == "__main__":
    main()
