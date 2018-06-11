import csv
import requests
import time
import cPickle as pkl

data_path = "/home/vinay/Documents/quant-works/data/"

nifty500_sec = {}

data = csv.reader(open(data_path+'ind_nifty500list.csv','r'))
for row in data:
	nifty500_sec[row[2]] = row[0]
del nifty500_sec['Symbol']

st = time.time()
price_history = {}
ct = 0 
api_format = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=NSE:%s&outputsize=compact&apikey=795GFKM0NHW7HCCD"
for symbol in nifty500_sec.keys():
	if ct%10 == 0:
		print ct,(time.time()-st)/60
	api_request = api_format%symbol
	content = requests.get(api_request)
	jf = content.json()
	try:
		price_history[symbol] = jf['Time Series (Daily)']
	except KeyError:
		print symbol
		continue
	ct += 1
	time.sleep(1)

print "Collected",len(price_history),"securities data in", (time.time()-st)/60, "mins"
pkl.dump(price_history, open(data_path+'nifty500.pkl','w'))

#missing_securities = ['COX&KINGS', 'MCDOWELL-N', 'L&TFH', 'M&M', 'J&KBANK', 'M&MFIN', 'GET&D', 'SPTL']