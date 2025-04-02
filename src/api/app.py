from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from src.model.price_model import PricingModel

app = FastAPI()

# Mount templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the pricing model
model = PricingModel()
try:
    model.load_models()
except:
    print("Warning: No pre-trained models found. Please train the models first.")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict")
async def predict(
    request: Request,
    category: str = Form(...),
    base_price: float = Form(...),
    inventory_level: int = Form(...),
    demand_last_24h: int = Form(...),
    competitor_price: float = Form(...),
    season: str = Form(...),
    day_of_week: str = Form(...),
    time_of_day: int = Form(...),
    special_event: int = Form(...),
    customer_segment: str = Form(...),
    rating: float = Form(...),
    historical_sales: int = Form(...),
):
    features = {
        'category': category,
        'base_price': base_price,
        'inventory_level': inventory_level,
        'demand_last_24h': demand_last_24h,
        'competitor_price': competitor_price,
        'season': season,
        'day_of_week': day_of_week,
        'time_of_day': time_of_day,
        'special_event': special_event,
        'customer_segment': customer_segment,
        'rating': rating,
        'historical_sales': historical_sales
    }
    
    rf_price = model.predict_price(features, 'random_forest')
    xgb_price = model.predict_price(features, 'xgboost')
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "rf_price": rf_price,
            "xgb_price": xgb_price,
            "features": features
        }
    )
