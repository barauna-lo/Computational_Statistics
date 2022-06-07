import numpy as np



#def specplus(nomeArquivo):

# ------------------------------------------------------------------------
# SpecplusV01.py
# ------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats, optimize
import numpy as np
import math
import csv

__author__ = 'Paulo Giovani; Updated for Luan Orion Baraúna (2022)'
__copyright__ = 'Copyright 2017, 3DBMO Project INPE'
__credits__ = ['Paulo Giovani', 'Reinaldo Roberto Rosa', 'Murilo da Silva Dantas']
__license__ = 'GPL'
__version__ = '0.1B'
__maintainer__ = 'Paulo Giovani'
__email__ = 'pg_faria@yahoo.com.br'





# FUNÇÃO PARA CONVERSÃO DE DADOS... PRECISA ARRUMAR

# def data(nomeArquivo):

# 	#Alocating `filename` dataset as `df`
# 	df = nomeArquivo
# 	#Saving `df` file as `.csv` extension
# 	df.to_csv("df.csv", index=False, header=False)

# 	#Conververting "df.csv" to a `.txt` file extension
# 	with open("Swind4096.txt", "w") as my_output_file: 
# 		with open("df.csv", "r") as my_input_file: [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
# 		my_output_file.close()

	
# 	print("\nData Analysis for 3DBMO simulations...\n")
	
# 	# Desabilita as mensagens de erro do Numpy (warnings)
# 	old_settings = np.seterr(divide = 'ignore', invalid = 'ignore', over = 'ignore')
	
# 	# Carrega o arquivo de dados
# 	nomeArquivo = 'Swind4096.txt'		
# 	data = np.genfromtxt(nomeArquivo,delimiter = ',',dtype = 'float32',filling_values = 0)

# 	return data
	

#	return data
	





#---------------------------------------------------------------------
# Calcula o PSD da série temporal
#---------------------------------------------------------------------
def psd(data):
	"""Calcula o PSD de uma série temporal."""


	# Define um intervalo para realizar o ajuste da reta
	INICIO = 10
	FIM = len(data)-10
	
	# O vetor com o tempo é o tamanho do número de pontos
	N = len(data)
	tempo = np.arange(len(data))

	# Define a frequência de amostragem
	dt = (tempo[-1] - tempo[0] / (N - 1))
	fs = 1 / dt

	# Calcula o PSD utilizando o MLAB
	power, freqs = mlab.psd(data, Fs = fs, NFFT = N, scale_by_freq = False)

	# Calcula a porcentagem de pontos utilizados na reta de ajuste
	totalFrequencias = len(freqs)
	totalPSD = FIM - INICIO
	porcentagemPSD = int(100 * totalPSD / totalFrequencias)

	# Seleciona os dados dentro do intervalo de seleção
	xdata = freqs[INICIO:FIM]
	ydata = power[INICIO:FIM]

	# Simula o erro
	yerr = 0.2 * ydata

	# Define uma função para calcular a Lei de Potência
	powerlaw = lambda x, amp, index: amp * (x**index)

	# Converte os dados para o formato LOG
	logx = np.log10(xdata)
	logy = np.log10(ydata)

	# Define a função para realizar o ajuste
	fitfunc = lambda p, x: p[0] + p[1] * x
	errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err    
	logyerr = yerr / ydata

	# Calcula a reta de ajuste
	pinit = [1.0, -1.0]
	out = optimize.leastsq(errfunc, pinit, args = (logx, logy, logyerr), full_output = 1)    
	pfinal = out[0]
	covar = out[1]
	index = pfinal[1]
	amp = 10.0 ** pfinal[0]
	indexErr = np.sqrt(covar[0][0])
	ampErr = np.sqrt(covar[1][1]) * amp
	
	# Retorna os valores obtidos
	return freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM
	
#---------------------------------------------------------------------
# Calcula o DFA 1D da série temporal
#---------------------------------------------------------------------
def dfa1d(timeSeries, grau):
	"""Calcula o DFA 1D (adaptado de Physionet), onde a escala cresce
	de acordo com a variável 'Boxratio'. Retorna o array 'vetoutput', 
	onde a primeira coluna é o log da escala S e a segunda coluna é o
	log da função de flutuação."""

	# 1. A série temporal {Xk} com k = 1, ..., N é integrada na chamada função perfil Y(k)
	x = np.mean(timeSeries)
	timeSeries = timeSeries - x
	yk = np.cumsum(timeSeries)
	tam = len(timeSeries)

	# 2. A série (ou perfil) Y(k) é dividida em N intervalos não sobrepostos de tamanho S
	sf = np.ceil(tam / 4).astype(np.int)
	boxratio = np.power(2.0, 1.0 / 8.0)
	vetoutput = np.zeros(shape = (1,2))

	s = 4
	while s <= sf:        
		serie = yk        
		if np.mod(tam, s) != 0:
			l = s * int(np.trunc(tam/s))
			serie = yk[0:l]			
		t = np.arange(s, len(serie), s)
		v = np.array(np.array_split(serie, t))
		l = len(v)
		x = np.arange(1, s + 1)
		
		# 3. Calcula-se a variância para cada segmento v = 1,…, n_s:
		p = np.polynomial.polynomial.polyfit(x, v.T, grau)
		yfit = np.polynomial.polynomial.polyval(x, p)
		vetvar = np.var(v - yfit)
		
		# 4. Calcula-se a função de flutuação DFA como a média das variâncias de cada intervalo
		fs = np.sqrt(np.mean(vetvar))
		vetoutput = np.vstack((vetoutput,[s, fs]))
		
		# A escala S cresce numa série geométrica
		s = np.ceil(s * boxratio).astype(np.int)

	# Array com o log da escala S e o log da função de flutuação   
	vetoutput = np.log10(vetoutput[1::1,:])

	# Separa as colunas do vetor 'vetoutput'
	x = vetoutput[:,0]
	y = vetoutput[:,1]

	# Regressão linear
	slope, intercept, _, _, _ = stats.linregress(x, y)

	# Calcula a reta de inclinação
	predict_y = intercept + slope * x

	# Calcula o erro
	pred_error = y - predict_y

	# Retorna o valor do ALFA, o vetor 'vetoutput', os vetores X e Y,
	# o vetor com os valores da reta de inclinação e o vetor de erros
	return slope, vetoutput, x, y, predict_y, pred_error

#---------------------------------------------------------------------
# Trecho principal
#---------------------------------------------------------------------
def main(nomeArquivo,titulo_da_serie):
	"""Função com o código princiapl do programa."""


	#Alocating `filename` dataset as `df`
	df = nomeArquivo
	#Saving `df` file as `.csv` extension
	df.to_csv("df.csv", index=False, header=False)

	#Conververting "df.csv" to a `.txt` file extension
	with open("Swind4096.txt", "w") as my_output_file: 
		with open("df.csv", "r") as my_input_file: [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
		my_output_file.close()

	
	print("\nData Analysis for 3DBMO simulations...\n")
	
	# Desabilita as mensagens de erro do Numpy (warnings)
	old_settings = np.seterr(divide = 'ignore', invalid = 'ignore', over = 'ignore')
	
	# Carrega o arquivo de dados
	nomeArquivo = 'Swind4096.txt'		
	data = np.genfromtxt(nomeArquivo,delimiter = ',',dtype = 'float32',filling_values = 0)

	


	#data = data(nomeArquivo)

	# Exibe os primeiro N valores do arquivo
	N = 10
	print("Original time series data (%d points): \n" %(len(data)))
	print("First %d points: %s\n" %(N, data[0:10]))
	print()
	
	#-----------------------------------------------------------------
		# Parâmetros gerais de plotagem
	#-----------------------------------------------------------------
	
	# Define os subplots
	fig = plt.figure()
	fig.subplots_adjust(hspace = .3, wspace = .2)
	
	# Tamanho das fontes
	tamanhoFonteEixoX = 16
	tamanhoFonteEixoY = 16
	tamanhoFonteTitulo = 16
	tamanhoFontePrincipal = 25
	
	# Título principal
	tituloPrincipal = 'Spectral Time Series Analysis'+str(titulo_da_serie) 	
	
	#-----------------------------------------------------------------
		# Plotagem da série original
	#-----------------------------------------------------------------
	
	# Define as cores da plotagem
	corSerieOriginal = 'r'
	
	# Título dos eixos da série original
	textoEixoX = 'Tempo'
	textoEixoY = 'Amplitude'
	textoTituloOriginal = 'Original Time Series Data'
	
	print("1. Plotting time series data...")
	
	# Plotagem da série de dados    
	#O = fig.add_subplot(1, 3, 1)    
	O = fig.add_subplot(2, 1, 1)
	O.plot(data, '-', color = corSerieOriginal)
	O.set_title(textoTituloOriginal, fontsize = tamanhoFonteTitulo)
	O.set_xlabel(textoEixoX, fontsize = tamanhoFonteEixoX)
	O.set_ylabel(textoEixoY, fontsize = tamanhoFonteEixoY)
	O.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
	O.grid()
	
	#-----------------------------------------------------------------
	# Cálculo e plotagem do PSD
	#-----------------------------------------------------------------
	
	# Calcula o PSD
	freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM = psd(data)

	# O valor do beta equivale ao index
	b = index

	# Define as cores da plotagem
	corPSD1 = 'k'
	corPSD2 = 'navy'

	# Título dos eixos do PSD
	textoPSDX = 'Frequência (Hz)'
	textoPSDY = 'Potência'
	textoTituloPSD = r'Power Spectrum Density $\beta$ = '
	
	print("2. Plotting Power Spectrum Density...  beta ="+str(b))


	# Plotagem do PSD    
	PSD = fig.add_subplot(2, 2, 3)    
	PSD.plot(freqs, power, '-', color = corPSD1, alpha = 0.7)
	PSD.plot(xdata, ydata, color = corPSD2, alpha = 0.8)
	PSD.axvline(freqs[INICIO], color = corPSD2, linestyle = '--')
	PSD.axvline(freqs[FIM], color = corPSD2, linestyle = '--')    
	PSD.plot(xdata, powerlaw(xdata, amp, index), 'r-', linewidth = 1.5, label = '$%.4f$' %(b))    
	PSD.set_xlabel(textoPSDX, fontsize = tamanhoFonteEixoX)
	PSD.set_ylabel(textoPSDY, fontsize = tamanhoFonteEixoY)
	PSD.set_title(textoTituloPSD + '%.4f' %(b), loc = 'center', fontsize = tamanhoFonteTitulo)
	PSD.set_yscale('log')
	PSD.set_xscale('log')
	PSD.grid() 
	
	#-----------------------------------------------------------------
	# Cálculo e plotagem do DFA
	#-----------------------------------------------------------------
	# Calcula o DFA 1D

	alfa, vetoutput, x, y, reta, erro = dfa1d(data, 1)

	print('alfa = '+str(alfa))
	# Verifica se o DFA possui um valor válido
	# Em caso afirmativo, faz a plotagem
	if not math.isnan(alfa):
		
		# Define as cores da plotagem
		corDFA = 'darkmagenta'

		# Título dos eixos do DFA
		textoDFAX = '$log_{10}$ (s)'
		textoDFAY = '$log_{10}$ F(s)'
		textoTituloDFA = r'Detrended Fluctuation Analysis $\alpha$ = '
		
		print("3. Plotting Detrended Fluctuation Analysis...")
		
		# Plotagem do DFA 
		DFA = fig.add_subplot(2, 2, 4)    
		DFA.plot(x, y, 's', 
				color = corDFA, 
				markersize = 4,
				markeredgecolor = 'r',
				markerfacecolor = 'None',
				alpha = 0.8)				 
		DFA.plot(x, reta, '-', color = corDFA, linewidth = 1.5)
		DFA.set_title(textoTituloDFA + '%.4f' %(alfa), loc = 'center', fontsize = tamanhoFonteTitulo)
		DFA.set_xlabel(textoDFAX, fontsize = tamanhoFonteEixoX)
		DFA.set_ylabel(textoDFAY, fontsize = tamanhoFonteEixoY)
		DFA.grid()

	else:  
		DFA = fig.add_subplot(2, 2, 4)
		DFA.set_title(textoTituloDFA + 'N.A.', loc = 'center', fontsize = tamanhoFonteTitulo)
		DFA.grid()

	#-----------------------------------------------------------------
	# Exibe e salva a figura
	#-----------------------------------------------------------------
	plt.suptitle(tituloPrincipal, fontsize = tamanhoFontePrincipal)
	nomeImagem = '3DBMO_PSD_DFA_2.png'
	fig.set_size_inches(15, 9)
	plt.savefig(nomeImagem, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)	
	plt.show()

	#print('teste')
		
#---------------------------------------------------------------------
# Trecho principal
#---------------------------------------------------------------------
#if __name__ == "__main__":
	#return main()	
#########################################################################################


def gamma1(nomeArquivo):
	"""Função com o código princiapl do programa."""

	#Alocating `filename` dataset as `df`
	df = nomeArquivo
	#Saving `df` file as `.csv` extension
	df.to_csv("df.csv", index=False, header=False)

	#Conververting "df.csv" to a `.txt` file extension
	with open("Swind4096.txt", "w") as my_output_file: 
		with open("df.csv", "r") as my_input_file: [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
		my_output_file.close()
	
	#print("\nData Analysis for 3DBMO simulations...\n")
	#print("α β γ1")
	# Desabilita as mensagens de erro do Numpy (warnings)
	old_settings = np.seterr(divide = 'ignore', invalid = 'ignore', over = 'ignore')
	
	# Carrega o arquivo de dados
	nomeArquivo = 'Swind4096.txt'		
	data = np.genfromtxt(nomeArquivo,delimiter = ',',dtype = 'float32',filling_values = 0)

	# Calcula o PSD
	freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM = psd(data)					
	# Calcula o DFA 1D
	alfa, vetoutput, x, y, reta, erro = dfa1d(data, 1)
	beta = index
	gamma = 2/7*(float(alfa)-float(beta))

	return alfa, beta, gamma

#########################################################################################
def normalizeSerie(s):
  serie = s.copy()
  serie = serie-np.average(serie)
  serie = serie/np.std(serie)
  return serie




def pmodel(noValues=4096, p=0.4999, slope=[]):
    import numpy as np
    noOrders = int(np.ceil(np.log2(noValues)))
    noValuesGenerated = 2**noOrders
    
    y = np.array([1])
    for n in range(noOrders):
        y = next_step_1d(y, p)
    
    if (slope):
        fourierCoeff = fractal_spectrum_1d(noValues, slope/2)
        meanVal = np.mean(y)
        stdy = np.std(y)
        x = np.fft.ifft(y - meanVal)
        phase = np.angle(x)
        x = fourierCoeff*np.exp(1j*phase)
        x = np.fft.fft(x).real
        x *= stdy/np.std(x)
        x += meanVal
    else:
        x = y
    
    return x[0:noValues], y[0:noValues]
 
     
def next_step_1d(y, p):
    import numpy as np
    y2 = np.zeros(y.size*2)
    sign = np.random.rand(1, y.size) - 0.5
    sign /= np.abs(sign)
    y2[0:2*y.size:2] = y + sign*(1-2*p)*y
    y2[1:2*y.size+1:2] = y - sign*(1-2*p)*y
    
    return y2
 
 
def fractal_spectrum_1d(noValues, slope):
    ori_vector_size = noValues
    ori_half_size = ori_vector_size//2
    a = np.zeros(ori_vector_size)
    
    for t2 in range(ori_half_size):
        index = t2
        t4 = 1 + ori_vector_size - t2
        if (t4 >= ori_vector_size):
            t4 = t2
        coeff = (index + 1)**slope
        a[t2] = coeff
        a[t4] = coeff
        
    a[1] = 0
    
    return a



def getSeries(slist, pinterval, betainterval, n):
    pval = np.random.uniform(pinterval[0], pinterval[1], n)
    beta = np.random.uniform(betainterval[0], betainterval[1], n)
    for i in range(0,(n)):
        x, dx = pmodel(4096, pval[i], beta[i])
        slist.append((dx - np.mean(dx)) / np.std(dx)) 


def plotpmodel(A, stype):
  plt.plot(A, color='r', lw=0.1)
  if(stype == 1):
    plt.title("Endogenous Series")
  else:
    plt.title("Exogenous Series")
  plt.show()




def getCFparams(slist):
  #k2list = np.empty((N), dtype=float)
  #sslist = np.empty((N), dtype=float)
  df = pd.DataFrame(np.zeros((N, 2)),columns = ['kurtosis', 'skewness'])
  for i in range(0, (N)):
    s=skew(slist[i])
    #Parâmetros de Cullen-Frey
    df['skewness'][i] = s*s
    df['kurtosis'][i] = kurtosis(slist[i])+3
  return df

#Espaço de C&F para A
#Espaço de C&F para A
def cullenfreyGEV(A):
    # Gaussian x Non-Gaussian (The Kullen-Frey Parameter Space)
    m=A.mean()
    std=A.std()
    s=A.skew()
    k1=A.kurtosis()
    k2=k1+3
    #Parâmetros de Cullen-Frey
    ss=s*s
    k=k2

    # Ploting Cullen-Frey Space
    plt.figure(num=None, figsize=(25, 25), dpi=100, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots()
    maior = np.max([ss,k])

    polyX1 = maior if maior > 4.4 else 4.4
    polyY1 = polyX1 + 1
    polyY2 = 3/2.*polyX1 + 3
    y_lim = polyY2 if polyY2 > 5 else 5
    #y_lim = max(exoParams['kurtosis'])

    x = [0, polyX1, polyX1, 0]
    y = [1, polyY1, polyY2, 3]

    poly2X1 = maior if maior > 2.15 else 2.15
    # EXTRAPLAÇÃO
    poly2Y2 = 2.62*polyX1 + 3
    xGEV = [0, polyX1, polyX1, 0]
    yGEV = [3, polyY2, poly2Y2, 3]    
    
    scale = 1
    poly = Polygon(np.c_[x, y]*scale, facecolor='#1B9AAA', edgecolor='#1B9AAA', alpha=0.5,label='β')
    ax.add_patch(poly)

    polyGEV = Polygon(np.c_[xGEV, yGEV]*scale, facecolor='red', edgecolor='red', alpha=0.5, label='GEV')
    ax.add_patch(polyGEV)

    
    # # poly2X1 = maior if maior > 2.15 else 2.15
    # poly2Y2 = 2.62*polyX1 + 3
    # x2 = [0, polyX1, polyX1, 0]
    # y2 = [3, polyY2, poly2Y2, 3]

    #ax.plot(x=x2, y=y2, fill="toself", name = "GEV")
    ax.plot(0, 3, label="Gaussian", marker='*', c='magenta')
    # ax.plot(ss, k, marker="o", c="green", label="Gaussian", linestyle='')
    ax.plot(ss,k, marker="o", c="blue", label="Observation", linestyle='')
    #ax.plot(exoParams['skewness'], exoParams['kurtosis'], marker="o", c="red", label="Observation", linestyle='')
    ax.plot(0, 4.187999875999753, label="logistic", marker='+', c='black')
    ax.plot(0, 1.7962675925351856, label ="uniform", marker='*', c='black')
    ax.plot(4, 9, label="exponential", marker='s', c='black')
    ax.plot(np.arange(0, polyX1, 0.1), 3/2. * np.arange(0, polyX1, 0.1) + 3, label="gamma", linestyle='-', c='black')
    ax.plot(np.arange(0, polyX1, 0.1), 2 * np.arange(0, polyX1, 0.1) + 3, label="lognormal", linestyle='-.', c='black')
    ax.legend(loc='best')
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), fancybox=True, shadow=True, ncol=3)
    #ax.set_ylim((y_lim +5), .3)
    #ax.set_xlim(-.1, (polyX1 + 0.3))
    ax.set_ylim(ax.get_ylim()[::-1])   #https://stackoverflow.com/questions/2051744/reverse-y-axis-in-pyplot
    plt.xlabel("Skewness²")
    #plt.title("C&F Parameter Space (Green Area for β Function)")
    plt.ylabel("Kurtosis")

    #plt.plot()
