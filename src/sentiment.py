import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import requests
from newspaper import Article
from transformers import pipeline


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
  def __init__(self, news_text_ls):
    self.news_text_ls = news_text_ls
  def get_sentiment_analysis(self):
    
    #if more 500 tokens
      #split into 500 token chunks
      #run pipeline seperately
      #get average
    
    
    pass
  

stock_news = News("RIVN", 10)
news_text_ls = stock_news.get_news_txt()
# print(news_text_ls)
stock_news = SentimentAnal(news_text_ls)

#Structure:
#initialize sentiment analysis model locally on website load
#News class (for each ticker) -> return all the news txt
#run in the model

