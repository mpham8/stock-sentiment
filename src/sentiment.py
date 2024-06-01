import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import requests
from newspaper import Article
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
import numpy as np
import yfinance as yf



class News:
  def __init__(self, ticker, articles):
    self.ticker = ticker
    self.articles = articles
    

  def get_news_txt(self):
    """
    This function gets the text of news articles from Finviz of a given ticker
    return: list containing each article's text as a string
    """

    finwiz_url = 'https://finviz.com/quote.ashx?t=' + self.ticker

    #URL request, get html of news table from Finviz
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(finwiz_url, headers=headers)
    
    if response.status_code != 200:
      print(f"Failed to retrieve articles for ticker {self.ticker}. HTTP Status Code: {response.status_code}")
      return
    
    print("Pulling URLs...")
    soup = BeautifulSoup(response.content, 'html.parser')
    news_table = soup.find(id='news-table')
    
    if news_table == False:
      print(f"No news table found for ticker {self.ticker}")
      return 

    #get news urls from finwiz news table
    # news_table = str(news_table)
    news_urls = []
    
    for url in news_table.find_all('a', href=True):
      news_urls.append(url['href'])
    
    news_urls = [url for url in news_urls if 'yahoo' not in url]

    
    # news_urls = news_urls[:self.articles]
    print("Found URLS:")
    for url in news_urls:
      print(url)

    #pull text from news article urls list
    news_text_ls = []
    for url in news_urls:
      try:
        if len(news_text_ls) >= self.articles:
          break
        # response = requests.get(url)
        # if response.status_code != 200:
        #   print(f"Failed to retrieve articles for url: {url}. HTTP Status Code: {response.status_code}")
        # else:
        #   # soup = BeautifulSoup(response.text, 'html.parser')
        #   # paragraphs = soup.find_all('p')
        #   # article_text = ''.join(p.get_text() for p in paragraphs)
        #   # news_text_ls.append(str(news_text_ls))
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if text != '':
          news_text_ls.append(text)


      except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve articles for url: {url}. {e}")


    return news_text_ls


class SentimentAnal:
  def __init__(self):

    MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
    self.config = AutoConfig.from_pretrained(MODEL)
    self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
  
  def get_sentiment_analysis(self, news_text_ls):
    cum_scores_ls = []
    for text in news_text_ls:
      tokens = self.tokenizer.tokenize(text)
      chunks = []

      for i in range(0, len(tokens), 500):
        chunk = tokens[i:i + 500]
        chunks.append(self.tokenizer.convert_tokens_to_string(chunk))

      cum_array = np.array([0,0,0], dtype=np.float64)      
      for chunk in chunks:
        encoded_input = self.tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
        output = self.model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        cum_array += scores
        
      avr_array = cum_array/len(chunks)
      cum_scores_ls.append(avr_array)

    
    return np.array(cum_scores_ls)
  

  def analyze_sentiment_score(self, cum_array):
    average_array = np.mean(cum_array, axis=0)
    print(cum_array)
    print(average_array)


    negative_count, neutral_count, positive_count = 0, 0, 0
    for i in range(len(cum_array)):
      index_highest = np.argmax(cum_array[i])
      if index_highest == 0:
        negative_count += 1
      elif index_highest == 2:
        positive_count += 1
      else:
        neutral_count += 1


    print(f"analyzed 10 $RIVN articles and found:")
    print(f"average sentiment: {round(average_array[0], 2)} negative, {round(average_array[1], 2)} neutral, {round(average_array[2], 2)} positive")
    print(f"negative articles: {negative_count}, neutral articles: {neutral_count}, positive articles: {positive_count}")

    
    #mixed
    #mostly postive - more than half positive
    #mostly negative - more than half negative
    #neutral - more than half neutral
    p_neg = average_array[0]/len(cum_array)
    p_nuet = average_array[1]/len(cum_array)
    p_pos = average_array[2]/len(cum_array)

    sentiment = "mixed"
    if p_pos > 0.5:
      sentiment = "positive"
    elif p_neg > 0.5:
      sentiment = "negative"
    elif p_nuet > 0.5:
      sentiment = "nuetral"

    print(f"sentiment is {sentiment}")




    return


class TechnicalIndicators:
  def __init__(self, ticker):
    self.ticker = ticker
    
    # Fetch data from Yahoo Finance
    self.data_15m = yf.download(self.ticker, period="1d", interval="15m")  # For day trading (15-minute interval)
    self.data_daily = yf.download(self.ticker, period="1mo", interval="1d")  # For swing trading (daily interval)
    self.data_weekly = yf.download(self.ticker, period="1y", interval="1wk")  # For long term trading (weekly interval)
      

    return 
  

  def get_rsi(self): 
    """
    day trade - 14 period 15 min
    swing trade - 14 period 1 day
    long term - 21 period 1 week

    over 70 - overbought
    under 30 - oversold
    """

    # Function to calculate RSI
    def calculate_rsi(data, period):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Calculate RSI for each timeframe
    self.data_15m['RSI_14_15m'] = calculate_rsi(self.data_15m, 14)  # Day trading
    self.data_daily['RSI_14_1d'] = calculate_rsi(self.data_daily, 14)  # Swing trading
    self.data_weekly['RSI_21_1w'] = calculate_rsi(self.data_weekly, 21)  # Long term trading


    day_rsi = round(self.data_15m['RSI_14_15m'].dropna().iloc[-1], 2)
    swing_rsi = round(self.data_daily['RSI_14_1d'].dropna().iloc[-1], 2)
    long_rsi = round(self.data_weekly['RSI_21_1w'].dropna().iloc[-1], 2)


    # Display the most recent RSI values for each timeframe
    rsi_values = {
        "Day": day_rsi,
        "Swing": swing_rsi,
        "Long": long_rsi
    }

    rsi_trends = {}
    for key in rsi_values:
      if rsi_values[key] >= 70:
        rsi_trends[key] = "overbought"
      elif rsi_values[key] <= 30:
        rsi_trends[key] = "oversold"
      else:
        rsi_trends[key] = "neutral"

    return rsi_values, rsi_trends
  

  def get_macd(self):
      """
      Fetches stock data from Yahoo Finance and calculates the MACD indicator.
      
      Parameters:
      - ticker: str, the stock ticker symbol.
      - start_date: str, the start date for the data in 'YYYY-MM-DD' format.
      - end_date: str, the end date for the data in 'YYYY-MM-DD' format.
      - slow: int, the period for the slow EMA.
      - fast: int, the period for the fast EMA.
      - signal: int, the period for the signal line.
      
      Returns:
      - str, "Overbought", "Oversold", or "Neutral" based on MACD analysis.
      """

      # Function to calculate macd
      def calculate_macd(data, period):
        slow, fast, signal = 26, 12, 9
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        macd_diff = macd - signal_line
        return macd_diff

      # Calculate RSI for each timeframe
      self.data_15m['MACD_14_15m'] = calculate_macd(self.data_15m, 14)  # Day trading
      self.data_daily['MACD_14_1d'] = calculate_macd(self.data_daily, 14)  # Swing trading
      self.data_weekly['MACD_21_1w'] = calculate_macd(self.data_weekly, 21)  # Long term trading
      
      day_macd = round(self.data_15m['MACD_14_15m'].dropna().iloc[-1], 2)
      swing_macd = round(self.data_daily['MACD_14_1d'].dropna().iloc[-1], 2)
      long_macd = round(self.data_weekly['MACD_21_1w'].dropna().iloc[-1], 2)
      # Display the most recent RSI values for each timeframe
      macd_values = {
        "Day": day_macd,
        "Swing": swing_macd,
        "Long": long_macd
      }
      # # Determine if the stock is overbought or oversold
      # if macd_diff.iloc[-1] > 0 and macd_diff.diff().iloc[-1] > 0:
      #     return "Overbought"
      # elif macd_diff.iloc[-1] < 0 and macd_diff.diff().iloc[-1] < 0:
      #     return "Oversold"
      # else:
      #     return "Neutral"

      macd_trends = {}
      for key in macd_values:
        if macd_values[key] > 0:
          macd_trends[key] = "overbought"
        elif macd_values[key] < 0:
          macd_trends[key] = "oversold"
        else:
          macd_trends[key] = "neutral"

      return macd_values, macd_trends

stock_news = News("AAPL", 25)
news_text_ls = stock_news.get_news_txt()
sentiment = SentimentAnal()
cum_array = sentiment.get_sentiment_analysis(news_text_ls)
sentiment.analyze_sentiment_score(cum_array)
  

# stock = TechnicalIndicators("RIVN")
# rsi, rsi2 = stock.get_rsi()
# macd, macd2 = stock.get_macd()
# print("RSI")
# print(rsi)
# print(rsi2)
# print("MACD")
# print(macd)
# print(macd2)

