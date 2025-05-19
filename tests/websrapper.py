import requests
from bs4 import BeautifulSoup
import pandas
from datetime import datetime

web = requests.get("https://coinmarketcap.com/")
c = web.content
soup = BeautifulSoup(c, "html.parser")
all = soup.find_all("tr", {"id", ""})
coin = all[0].find("a", {"class", "currency-name-container link-secondary"}).text
l=[]

for page in range(0, 20, 1):
    base_url = "https://coinmarketcap.com/"
    r = requests.get(base_url + str(page))
    c = r.content
    soup = BeautifulSoup(c, "html.parser")
    all = soup.find_all("tr", {"id", ""})
    for item in all:
        d = {}
        d["Coin"] = item.find("a", {"class", "currency-name-container"}).text
        d["ABV"] = item.find("a", {"class", "link-secondary"}).text
        d["Price"] = item.find("a", {"class", "price"}).text
        d["Market Cap"] = item.find("td", {"class", "market-cap"})["data-usd"]
        try:
            d["Percent Change"] = item.find("td", {"class", "percent-change"}).text
        except:
            d["Percent Change"] = None
        l.append(d)
df = pandas.DataFrame(l)
current_time = datetime.now().strftime("%Y-%m-%d")
df.to_csv(current_time + ".csv")