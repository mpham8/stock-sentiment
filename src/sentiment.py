import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import requests
import re

class News:
  def __init__(self, ticker, articles):
    self.ticker = ticker
    self.articles = articles

  def get_news_txt(self):
    """
    This function gets the text of news articles from Finviz of a given ticker
    return: list containing each article's text as a string
    """

    finwiz_url = 'https://finviz.com/quote.ashx?t='
    url = finwiz_url + self.ticker

    #URL request, get html of news table from Finviz
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
      print(f"Failed to retrieve articles for ticker {self.ticker}. HTTP Status Code: {response.status_code}")
      return
    
    soup = BeautifulSoup(response.content, 'html.parser')
    print("Pulling URLs...")
    news_table = soup.find(id='news-table')
    
    if news_table == False:
      print(f"No news table found for ticker {self.ticker}")
      return 

    #get news urls from finwiz news table
    news_table = str(news_table)
    url_pattern = re.compile(r'href="(.*?)"')
    news_urls = url_pattern.findall(news_table)
    news_urls = news_urls[:self.articles]
    print("Found URLS:")
    for url in news_urls:
      print(url)

    for url in news_urls:
      try:
        pass
      except:
        pass


    #pull text from urls
    return

stock_news = News("AAPL", 10)
stock_news.get_news_txt()

#Structure:
#initialize sentiment analysis model locally on website load
#News class (for each ticker) -> return all the news txt
#run in the model

