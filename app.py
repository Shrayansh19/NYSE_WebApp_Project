from flask import Flask, render_template, request
import yfinance as yf
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import requests

load_dotenv()

from category_map import category_map  # Category-to-best-performer mapping
from company_to_sector import company_to_sector  # Company-to-sector mapping

# Flask app
app = Flask(__name__)

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set it as an environment variable.")
# NewsAPI Key
NEWS_API_KEY = "37ccccdcb6c944a890b94464485a2205"

def get_stock_data(symbol):
    #Fetching stock data for the past year using Yahoo Finance.
    
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        if hist.empty:
            return {"error": f"No data found for symbol '{symbol}'."}
        return hist
    except Exception as e:
        return {"error": str(e)}

def get_news(company_name):
    """
    Fetching the recent news articles about the company using NewsAPI.
    """
    try:
        url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        news_headlines = [{"title": article["title"]} for article in articles[:5]]
        return news_headlines
    except Exception as e:
        return [{"title": f"Error fetching news: {str(e)}"}]

def analyze_sentiment_with_openai(news_headlines):
    """
    Analyzing news sentiment using OpenAI GPT API to recommend Buy, Hold, or Sell. 
    """
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        messages = [
            {"role": "system", "content": "You are a financial analyst providing stock recommendations."},
            {
                "role": "user",
                "content": (
                    "Analyze the sentiment of the following news headlines and provide a recommendation: "
                    "either 'Buy', 'Hold', or 'Sell' the stock. Your recommendation should always end with one of these three options.\n"
                    + "\n".join([f"- {headline['title']}" for headline in news_headlines])
                )
            },
        ]
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stock', methods=['POST'])
def stock_info():
    company_symbol = request.form['company'].strip().upper()

    # Validating the company symbol
    sector = company_to_sector.get(company_symbol)
    if not sector:
        return f"Error: Company code '{company_symbol}' not found in the database."

    best_performer = category_map.get(sector)
    if not best_performer:
        return f"Error: No best performer found for sector '{sector}'."

    # Fetching the stock data
    company_data = get_stock_data(company_symbol)
    best_performer_data = get_stock_data(best_performer)
    if "error" in company_data:
        return f"Error: {company_data['error']}"
    if "error" in best_performer_data:
        return f"Error: {best_performer_data['error']}"

    # Check if the selected stock is the best in its sector
    is_best_performer = (company_symbol == best_performer)

    news = get_news(company_symbol)
    recommendation = analyze_sentiment_with_openai(news)

    # Plotting stock performance
    plt.figure(figsize=(10, 6))
    plt.plot(company_data["Close"].index, company_data["Close"], label=f"{company_symbol} Closing Prices", color="blue")
    if is_best_performer:
        plt.plot(company_data["Close"].index, company_data["Close"], linestyle="dotted", label=f"{company_symbol} (Best Performer)", color="green")
    else:
        plt.plot(best_performer_data["Close"].index, best_performer_data["Close"], label=f"{best_performer} Closing Prices (Best Performer)", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"Stock Performance Comparison: {company_symbol} vs {best_performer}")
    plt.legend()
    plt.grid()

    if not os.path.exists("static"):
        os.makedirs("static")

    plot_filename = f"static/{company_symbol}_comparison_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    return render_template(
        'stock.html',
        company_symbol=company_symbol,
        sector=sector,
        is_best_performer=is_best_performer,
        best_performer=best_performer,
        recommendation=recommendation,
        plot_filename=plot_filename
    )

if __name__ == "__main__":
    app.run(debug=True)
