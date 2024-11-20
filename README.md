# Stockmovementanalysis_telegram.py

# Stock Movement Analysis Based on Social Media Sentiment

This project develops a machine learning model to predict stock movements using sentiment analysis from Telegram messages and historical stock data.

## Repository Structure

- `telegram_scraper.py`: Script for scraping Telegram messages
- `preprocess.py`: Script for preprocessing scraped data and performing sentiment analysis
- `stock_prediction_model.py`: Main script for feature engineering, model training, and evaluation
- `requirements.txt`: List of Python dependencies
- `README.md`: This file, containing setup and running instructions
- `Stock_Movement_Analysis_Report.pdf`: Detailed report on the project (not included in this repository)

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stock-movement-analysis.git
   cd stock-movement-analysis
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up Telegram API credentials:
   - Create a `.env` file in the project root
   - Add your Telegram API credentials:
     ```
     TELEGRAM_API_ID=your_api_id
     TELEGRAM_API_HASH=your_api_hash
     TELEGRAM_PHONE_NUMBER=your_phone_number
     ```

## Running the Project

1. Scrape Telegram data:
   ```
   python telegram_scraper.py
   ```

2. Preprocess the scraped data:
   ```
   python preprocess.py
   ```

3. Run the stock prediction model:
   ```
   python stock_prediction_model.py
   ```

## Data Sources

- Telegram channels: @StockMarketChat, @WallStreetBets (example channels, replace with actual ones)
- Stock data: S&P 500 index, retrieved using yfinance library

## Model Details

The project uses a Random Forest Regressor to predict stock returns based on historical price data, technical indicators, and sentiment scores derived from Telegram messages.

For more detailed information about the project, including challenges faced, model performance, and future improvements, please refer to the `Stock_Movement_Analysis_Report.pdf`.

