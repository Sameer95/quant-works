import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import itertools

def expected_return(seq):
	E_Ret = np.mean(seq)
	return E_Ret

def expected_risk(seq):
	E_risk = np.std(seq)
	return E_risk

def expected_covariance(arr):
	E_cov = np.cov(arr)
	return E_cov

def softmax(arr):
	arr_max = np.max(arr)
	arr = arr- arr_max
	exp_arr = np.exp(arr)
	exp_arr_sum = np.sum(exp_arr)
	softmax_arr = exp_arr/exp_arr_sum
	return softmax_arr

def generate_samples(shape, n_samples, distribution = 'uniform'):
	samples = []
	for i in range(n_samples):
		if distribution == 'uniform':
			sample = np.random.uniform(size = shape)
		elif distribution == 'normal':
			#sample = np.random.normal(size = shape, loc = 1.0)
			sample = np.random.randn(shape[0], 1)*100
		elif distribution == 'binomial':
			wt_vec = []
			for wt in range(shape[0]):
				prob = np.random.uniform()/100
				wt = np.random.binomial(n = 10, p = prob)
				wt_vec.append(wt)
			sample = np.asarray(wt_vec)
		sample = softmax(sample)
		sample2 = softmax(sample*-1)
		samples.append(sample)
		samples.append(sample2)
	return samples

def plot_data(risk_data, return_data):
	plt.scatter(risk_data, return_data)
	plt.xlabel('Standard deviation(Risk)')
	plt.ylabel('Expected return')
	plt.show()

def compute_risk_var_nifty(mode):
	global E_ret_risk
	for sym, hist in data.iteritems():
		stock_hist = []
		for day, attr in hist.iteritems():
			op_price = float(attr['1. open'])
			if op_price == 0.0:									# assumption: non trivial trading has non zero opening price
				continue										# i.e if an entry has 0 opening price then the data is considered missing

			if mode == 1:
				cl_price = float(attr['4. close'])
			elif mode == 2:
				 cl_price = float(attr['5. adjusted close'])
			gain = (cl_price-op_price)

			stock_hist.append( gain*100/op_price)

		stock_hist = np.asarray(stock_hist)
		ret = expected_return(stock_hist)
		risk = expected_risk(stock_hist)
		E_ret_risk[sym] = [ret, risk]

def compute_covariance_nifty(mode):
	global E_ret_risk
	global covar_matrix
	securities = data.keys()
	sec_indices = np.arange(len(securities))
	tuples = list(itertools.combinations(sec_indices, 2))
	ct = 0
	for sec1, sec2 in tuples:
		hist1 = data[securities[sec1]]
		hist2 = data[securities[sec2]]
		stock_hist1 = []
		stock_hist2 = []
		for day, attr in hist1.iteritems():
			try:
				attr2 = hist2[day]
			except KeyError:
				continue
			op_price1 = float(attr['1. open'])
			op_price2 = float(attr2['1. open'])
			if op_price1 == 0.0 or op_price2 == 0.0:
				continue

			if mode == 1:
				cl_price1 = float(attr['4. close'])
				cl_price2 = float(attr2['4. close'])
			elif mode == 2:
				cl_price1 = float(attr['5. adjusted close'])
				cl_price2 = float(attr2['5. adjusted close'])				
			gain1 = (cl_price1-op_price1)
			gain2 = (cl_price2-op_price2)

			stock_hist1.append( gain1*100/op_price1)
			stock_hist2.append( gain2*100/op_price2)

		stock_hist = [stock_hist1, stock_hist2]
		stock_hist = np.asarray(stock_hist)
		if stock_hist.shape[1] < 2:
			covar_matrix[sec1][sec2] = 0.0
			covar_matrix[sec2][sec1] = 0.0
			continue
		else:
			cov = expected_covariance(stock_hist)
			assert cov[0][1] == cov[1][0]
			covar_matrix[sec1][sec2] = cov[0][1]

def calc_eff_frontier():
	a = 1
	return a

if __name__=="__main__":
	np.random.seed(10)
	global mode, data, data_path, num_combns
	E_ret_risk = {}
	num_combns = 100000

	data_path = "/home/vinay/Documents/quant-works/data/"
	data = pkl.load(open(data_path+'nifty500.pkl','r'))

	covar_matrix = np.zeros((len(data), len(data)))
	compute_risk_var_nifty(2)

	return_data = []
	risk_data = []

	idx = 0
	for sym, val in E_ret_risk.iteritems():
		return_data.append(val[0])
		risk_data.append(val[1])
		covar_matrix[idx][idx] = val[1]
		idx += 1

	return_data = np.asarray(return_data)[:, None]
	risk_data = np.asarray(risk_data)
	plot_data(risk_data, return_data)
	print np.max(return_data)

	compute_covariance_nifty(2)
	portfolio_return_data = []
	portfolio_risk_data = []
	portfolios = generate_samples(shape = (len(E_ret_risk), 1), n_samples = num_combns, distribution = 'binomial')

	securities = data.keys()
	sec_indices = np.arange(len(securities))
	tuples = list(itertools.combinations(sec_indices, 2))
	for portfolio in portfolios:
		E_ret_p = np.dot(portfolio.T, return_data)
		weight_combns = np.dot(portfolio, portfolio.T)
		weighted_cov = covar_matrix*weight_combns
		E_cov_p = np.sum(weighted_cov)
		E_risk_p = np.sqrt(E_cov_p)
		portfolio_return_data.append(E_ret_p)
		portfolio_risk_data.append(E_risk_p)

	plot_data(portfolio_risk_data, portfolio_return_data)
	print np.max(portfolio_return_data)