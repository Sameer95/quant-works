import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt

def expected_return(seq):
	E_Ret = np.mean(seq)
	return E_Ret

def expected_risk(seq):
	E_risk = np.std(seq)
	return E_risk

def plot_data(risk_data, return_data):
	plt.scatter(risk_data, return_data)
	plt.xlabel('Standard deviation(Risk)')
	plt.ylabel('Expected return')
	plt.show()

def compute_risk_var_nifty(mode):
	global E_ret_risk
	E_ret_risk = {}
	for sym, hist in data.iteritems():
		stock_hist = []
		for day, attr in hist.iteritems():
			op_price = float(attr['1. open'])
			if op_price == 0.0:									# assumption: non trivial trading has non zero opening price
				continue										# i.e if an entry has 0 opening price then the data is considered missing

			if mode == 1:
				cl_price = float(attr['4. close'])
				gain = (cl_price-op_price)
			elif mode == 2:
				adj_cl = float(attr['5. adjusted close'])
				gain = (adj_cl-op_price)

			stock_hist.append( gain*100/op_price)

		stock_hist = np.asarray(stock_hist)
		ret = expected_return(stock_hist)
		risk = expected_risk(stock_hist)
		E_ret_risk[sym] = [ret, risk]

if __name__=="__main__":
	global E_ret_risk, mode, data, data_path, num_combns
	num_combns = 100

	data_path = "/home/vinay/Documents/quant-works/data/"
	data = pkl.load(open(data_path+'nifty500.pkl','r'))

	compute_risk_var_nifty(2)

	return_data = []
	risk_data = []

	for sym, val in E_ret_risk.iteritems():
		return_data.append(val[0])
		risk_data.append(val[1])

	plot_data(risk_data, return_data)
