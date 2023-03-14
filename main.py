import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from pyod.models.abod import ABOD
from pyod.utils.data import generate_data, get_outliers_inliers
import matplotlib.font_manager
import csv
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from numpy import unique
from numpy import where
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from bioinfokit.analys import stat
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

df = pd.read_csv(r'Design Applications Dataset.csv', parse_dates=True)

date = df['Date']
date = date[::-1]
date = pd.to_datetime(date)
date2 = pd.to_datetime(date, format='%Y')

USGDP = df['US GDP']
USGDP = USGDP[::-1]
# fig, axusgdp = plt.subplots()
# axusgdp.plot(date,USGDP)
# axusgdp.set_title('US GDP')
# axusgdp.set_xlabel('Date')
# axusgdp.set_ylabel('US GDP')


USInflation = df['US Inflation']
USInflation = USInflation[::-1]
# fig, axusinflation = plt.subplots()
# axusinflation.plot(date,USInflation)
# axusinflation.set_title('US Inflation')
# axusinflation.set_xlabel('Date')
# axusinflation.set_ylabel('US Inflation')

GlobalGDP = df['Global GDP']
GlobalGDP = GlobalGDP[::-1]
# fig, axglobalgdp = plt.subplots()
# axglobalgdp.plot(date,GlobalGDP)
# axglobalgdp.set_title('Global GDP')
# axglobalgdp.set_xlabel('Date')
# axglobalgdp.set_ylabel('Global GDP')

GlobalInflation = df['Global Inflation Rate']
GlobalInflation = GlobalInflation[::-1]
# fig, axglobalinflation = plt.subplots()
# axglobalinflation.plot(date,GlobalInflation)
# axglobalinflation.set_title('Global Inflation Rate')
# axglobalinflation.set_xlabel('Date')
# axglobalinflation.set_ylabel('Global Inflation')


twybpopen = df['2 Year Bond Open']
twybpopen = twybpopen[::-1]

twybphigh = df['2 Year Bond High']
twybphigh = twybphigh[::-1]

twybplow = df['2 Year Bond Low']
twybplow = twybplow[::-1]

twybpchange = df['2 Year Bond Change %']
twybpchange = twybpchange[::-1]

twybp = df['2 Year Bond Price']
twybp = twybp[::-1]
# fig, ax3 = plt.subplots()
# ax3.plot(date,twybp)
# ax3.set_title('2 Year Bond Price')
# ax3.set_xlabel('Date')
# ax3.set_ylabel('2 Year Bond Price')


# fig, axtybpchange = plt.subplots()
# axtybpchange.plot(date,tybpchange)
# tybpchange.set_title('2 Year Bond Change %')
# tybpchange.set_xlabel('Date')
# tybpchange.set_ylabel('2 Year Bond Change %')

tybpopen = df['10 Year Bond Open']
tybpopen = tybpopen[::-1]

tybp = df['10 Year Bond Price']
tybp = tybp[::-1]
# fig, axtybp = plt.subplots()
# axtybp.plot(date, tybp)
# axtybp.set_title('10 Year Bond Price')
# axtybp.set_xlabel('Date')
# axtybp.set_ylabel('10 Year Bond Price')


tybpchange = df['10 Year Bond Change %']
tybpchange = tybpchange[::-1]

tybphigh = df['10 Year Bond High']
tybphigh = tybphigh[::-1]

tybplow = df['10 Year Bond Low']
tybplow = tybplow[::-1]

# fig, axtybpchange = plt.subplots()
# axtybpchange.plot(date,tybpchange)
# tybpchange.set_title('10 Year Bond Change %')
# tybpchange.set_xlabel('Date')
# tybpchange.set_ylabel('10 Year Bond Change %')

snpopen = df['S&P 500 Open']
snpopen = snpopen[::-1]

snphigh = df['S&P 500 High']
snphigh = snphigh[::-1]

snplow = df['S&P 500 Low']
snplow = snplow[::-1]

snpchange = df['S&P 500 Change %']
snpchange = snpchange[::-1]

snp = df['S&P 500 Price']
snp = snp[::-1]
# fig, ax2 = plt.subplots()
# ax2.plot(date,snp)
# ax2.set_title('S&P 500 Price')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('S&P 500 Price')


snpchange = df['S&P 500 Change %']
snpchange = snpchange[::-1]
# fig, axsnpchange = plt.subplots()
# axsnpchange.plot(date,snpchange)
# axsnpchange.set_title('S&P 500 Change %')
# axsnpchange.set_xlabel('Date')
# axsnpchange.set_ylabel('S&P 500 Change %')


USInterest = df['US Insterest']
USInterest = USInterest[::-1]
# fig, axusinterest = plt.subplots()
# axusinterest.plot(date,USInterest)
# axusinterest.set_title('US Insterest')
# axusinterest.set_xlabel('Date')
# axusinterest.set_ylabel('Us Interest')

snphigh_price = df['S&P 500 High - Price']
snphigh_price = snphigh_price[::-1]

snpprice_low = df['S&P 500 Price -Low']
snpprice_low = snpprice_low[::-1]

snphigh_low = df['S&P 500 High- Low']
snphigh_low = snphigh_low[::-1]

snphigh_open = df['S&P 500 High - Open']
snphigh_open = snphigh_open[::-1]

snpopen_low = df['S&P 500 Open - Low']
snpopen_low = snpopen_low[::-1]

tybhigh_price = df['10 Year Bond High - Price']
tybhigh_price = tybhigh_price[::-1]

tybprice_low = df['10 Year Bond Price -Low']
tybprice_low = tybprice_low[::-1]

tybhigh_low = df['10 Year Bond High - Low']
tybhigh_low = tybhigh_low[::-1]

tybhigh_open = df['10 Year Bond High - Open']
tybhigh_open = tybhigh_open[::-1]

tybopen_low = df['10 Year Bond Open - Low']
tybopen_low = tybopen_low[::-1]

twybhigh_price = df['2 Year Bond High - Price']
twybhigh_price = twybhigh_price[::-1]

twybprice_low = df['2 Year Bond Price -Low']
twybprice_low = twybprice_low[::-1]

twybhigh_low = df['2 Year Bond High - Low']
twybhigh_low = twybhigh_low[::-1]

twybhigh_open = df['2 Year Bond High - Open']
twybhigh_open = twybhigh_open[::-1]

twybopen_low = df['2 Year Bond Open - Low']
twybopen_low = twybopen_low[::-1]

# plt.plot(date,USInterest,color="green", label= 'US Interest Rate')
# plt.plot(date,twybp,color="red",label= '2 Year Bond Prices')
# plt.xlabel('Date')
# plt.ylabel('Values')
# plt.legend()
# plt.title('2 Year Bond and US Interest Rate')


#######BONDS AND S&P 500 PIE CHART#######


####### BAR CHARTS#######
# plt.bar(date2, USInterest, color='g')
# plt.title("Date and US Interest")
# plt.xlabel("Date")
# plt.ylabel("US Interest")
# plt.show()

# plt.bar(date2, USGDP, color='g')
# plt.title("Date and US GDP")
# plt.xlabel("Date")
# plt.ylabel("US GDP")
# plt.show()

# plt.bar(date2, USInflation, color='g')
# plt.title("Date and US Inflation")
# plt.xlabel("Date")
# plt.ylabel("US Inflation")
# plt.show()

# plt.bar(date, GlobalInflation, color='g')
# plt.title("Date and Global Inflation")
# plt.xlabel("Date")
# plt.ylabel("Global Inflation")
# plt.show()

#######Multiple Plot#######
# plt.plot(date,GlobalInflation,marker='o', markersize=1.7, color="green", alpha=0.5, label='Global Inflation')
# plt.plot(date,USInflation,marker='o', markersize=1.7, color="red", alpha=0.5, label='US Inflation')
# plt.title('US and Global Inflation')
# plt.legend(markerscale=8)

# plt.plot(date,GlobalGDP,marker='o', markersize=1.7, color="green", alpha=0.5, label='Global GDP')
# plt.plot(date,USGDP,marker='o', markersize=0.8, color="red", alpha=0.5, label='US GDP')
# plt.title('US and Global GDP')
# plt.legend(markerscale=8)

# plt.plot(date,twybp,color="grey", label='2 Year Bond Price')
# plt.plot(date,tybp,label='10 Year Bond Price')
# plt.title('10 and 2 Years Bond Prices')
# plt.legend(markerscale=8)

#######Scatter#######
# sns.kdeplot(data = df, x='US GDP', y='Global GDP', cmap="Reds", shade=True).set_title('US and Global GDP')

# sns.kdeplot(data = df, x='10 Year Bond Price', y='2 Year Bond Price', cmap="Reds", shade=True).set_title('Bond Prices')

# sns.kdeplot(data = df, x='US Inflation', y='Global Inflation Rate', cmap="Reds", shade=True).set_title('US and Global Inflation')

# plt.plot( date,USInflation, marker='o', markersize=1.5, color="grey", alpha=0.3, label='US Inflation')
# plt.plot( date,GlobalInflation, marker='o', markersize=1.5, alpha=0.3, label='Global Inflation')
# plt.plot( date,USGDP/10000, color="green",  label='US GDP')
# plt.plot( date,GlobalGDP/10000, color="red",  label='Global GDP')
# plt.legend(markerscale=8)
# plt.xlabel('Date')
# plt.ylabel('Values')
# plt.title('US and Global GDP with Inflation', loc='left')
# plt.show()

# plt.plot( date,USInflation, marker='o', markersize=1.5, color="grey", alpha=0.3, label='US Inflation')
# plt.plot( date,USInterest, marker='o', markersize=1.5, alpha=0.3, label='Us Interest')
# plt.plot( date,twybp, color="green",  label='2 Year Bond Prices')
# plt.plot( date,tybp, color="red",  label='10 Year Bond Prices')
# plt.legend(markerscale=8)
# plt.xlabel('Date')
# plt.ylabel('Values')
# plt.title('Bond Prices with US Inflation and Interest Rates', loc='left')

# plt.plot( date,twybp, marker='o', markersize=1.5, color="grey", alpha=0.3, label='2 Year Bond Prices')
# plt.plot( date,tybp, marker='o', markersize=1.5, alpha=0.3, label='10 Year Bond Prices')
# plt.plot( date,USGDP/10000, color="green",  label='US GDP')
# plt.plot( date,GlobalGDP/10000, color="red",  label='Global GDP')
# plt.legend(markerscale=8)
# plt.xlabel('Date')
# plt.ylabel('Values')
# plt.title('Bond Prices with Global and US GDP', loc='left')

# plt.boxplot([USInflation,GlobalInflation], labels=['US Inflation','Global Infation'])
# plt.title('US and Global Inflation Box Plot')

# plt.boxplot([USInterest], labels=['US Interest'])
# plt.title('US Interest Box Plot')

# plt.boxplot([USGDP, GlobalGDP], labels=['US GDP', 'Global GDP'])
# plt.title('US and Global GDP Box Plot')

# plt.boxplot([twybp, tybp], labels=['2 Year Bond Price', '10 Year Bond Price'])
# plt.title('2 and 10 Year Bond Price Box Plot')

# plt.boxplot([snp], labels=['S&P 500 Price'])
# plt.title('S&P 500 Price Box Plot')

# plt.boxplot([snpopen],labels=['S&P 500 Open'])
# plt.title('S&P 500 Open Box Plot')

# plt.boxplot([twybpopen],labels=['2 Year Bond Open'])
# plt.title('2 Year Bond Open Box Plot')

# plt.boxplot([twybphigh],labels=['2 Year Bond High'])
# plt.title('2 Year Bond High Box Plot')

# plt.boxplot([twybplow],labels=['2 Year Bond Low'])
# plt.title('2 Year Bond Low Box Plot')

# plt.boxplot([twybpchange],labels=['2 Year Bond Change'])
# plt.title('2 Year Bond Change Box Plot')

# plt.boxplot([tybpopen],labels=['10 Year Bond Open'])
# plt.title('10 Year Bond Open Box Plot')

# plt.boxplot([tybphigh],labels=['10 Year Bond High'])
# plt.title('10 Year Bond High Box Plot')

# plt.boxplot([tybplow],labels=['10 Year Bond Low'])
# plt.title('10 Year Bond Low Box Plot')

# plt.boxplot([snpchange],labels=['S&P 500 Change'])
# plt.title('S&P 500 Change Box Plot')
#
# plt.boxplot([snplow],labels=['S&P 500 Low'])
# plt.title('S&P 500 Low Box Plot')
#
# plt.boxplot([snphigh],labels=['S&P 500 High'])
# plt.title('S&P 500 High Box Plot')

# plt.boxplot([snphigh_price],labels=['S&P 500 High - Price'])
# plt.title('S&P 500 High - Price Box Plot')

# plt.boxplot([snpprice_low],labels=['S&P 500 Price -Low'])
# plt.title('S&P 500 Price -Low Box Plot')

# plt.boxplot([snphigh_low],labels=['S&P 500 High- Low'])
# plt.title('S&P 500 High- Low Box Plot')

# plt.boxplot([snphigh_open],labels=['S&P 500 High - Open'])
# plt.title('S&P 500 High - Open Box Plot')

# plt.boxplot([snpopen_low],labels=['S&P 500 Open - Low'])
# plt.title('S&P 500 Open - Low Box Plot')

# plt.boxplot([tybhigh_price],labels=['10 Year Bond High - Price'])
# plt.title('10 Year Bond High - Price Box Plot')

# plt.boxplot([tybprice_low],labels=['10 Year Bond Price -Low'])
# plt.title('10 Year Bond Price -Low Box Plot')

# plt.boxplot([tybhigh_low],labels=['10 Year Bond High - Low'])
# plt.title('10 Year Bond High - Low Box Plot')

# plt.boxplot([tybhigh_open],labels=['10 Year Bond High - Open'])
# plt.title('10 Year Bond High - Open Box Plot')

# plt.boxplot([tybopen_low],labels=['10 Year Bond Open - Low'])
# plt.title('10 Year Bond Open - Low Box Plot')

# plt.boxplot([twybhigh_price],labels=['2 Year Bond High - Price'])
# plt.title('2 Year Bond High - Price Box Plot')

# plt.boxplot([twybprice_low],labels=['2 Year Bond Price -Low'])
# plt.title('2 Year Bond Price -Low Box Plot')

# plt.boxplot([twybhigh_low],labels=['2 Year Bond High - Low'])
# plt.title('2 Year Bond High - Low Box Plot')

# plt.boxplot([twybhigh_open],labels=['2 Year Bond High - Open'])
# plt.title('2 Year Bond High - Open Box Plot')

# plt.boxplot([twybopen_low],labels=['2 Year Bond Open - Low'])
# plt.title('2 Year Bond Open - Low Box Plot')

#######Data Cleaning#######
# print(df.isnull().sum())
# print(df.columns)
#######Data Outliers#######
stdUSGDP = USGDP.std()
avgUSGDP = USGDP.mean()
# print(stdUSGDP)
# print(avgUSGDP)
three_sigma_plus_USGDP = avgUSGDP + (3 * stdUSGDP)
three_sigma_minus_USGDP = avgUSGDP - (3 * stdUSGDP)
# print(three_sigma_minus_USGDP,three_sigma_plus_USGDP)

# for column in df.columns[1:]:
#    processedcolumn = column
#    std = processedcolumn.std()
#    avg = processedcolumn.mean()
#    three_sigma_plus = avg + (3*std)
#    three_sigma_minus = std - (3*std)
#    print(column,std,avg)
x = 0

for x in range(len(USInflation)):
    USInflation[x] = USInflation[x] + 2.097162

for x in range(len(GlobalInflation)):
    GlobalInflation[x] = GlobalInflation[x] + 0.589829

# for i in range(len(USInflation)):
#    print(USInflation[i] ," - " , i)


plt.show()

# print((USGDP > three_sigma_plus_USGDP) | (USGDP < three_sigma_minus_USGDP))
# no_outlier_USGDP = USGDP[(USGDP < three_sigma_plus_USGDP) & (USGDP > three_sigma_minus_USGDP)]
# print(no_outlier_USGDP)
# print((USGDP.shape[0] - no_outlier_USGDP.shape[0])-len(USGDP))

# plt.boxplot(USGDP)

stdUSInflation = USInflation.std()
avgUSInflation = USInflation.mean()

three_sigma_plus_USInflation = avgUSInflation + (1.289 * stdUSInflation)
three_sigma_minus_USInflation = avgUSInflation - (1.289 * stdUSInflation)
# print(stdUSInflation,avgUSInflation)
# print(three_sigma_plus_USInflation,three_sigma_minus_USInflation)

# no_outlier_USInflation = USInflation[(USInflation < three_sigma_plus_USInflation) & (USInflation > three_sigma_minus_USInflation)]
# print(no_outlier_USInflation)

stdGlobalGDP = GlobalGDP.std()
avgGlobalGDP = GlobalGDP.mean()

three_sigma_plus_GlobalGDP = avgGlobalGDP + (3 * stdGlobalGDP)
three_sigma_minus_GlobalGDP = avgGlobalGDP - (3 * stdGlobalGDP)
# print(three_sigma_plus_GlobalGDP,three_sigma_minus_USGDP)

no_outlier_GlobalGDP = GlobalGDP[(GlobalGDP < three_sigma_plus_GlobalGDP) & (GlobalGDP > three_sigma_minus_GlobalGDP)]
# print(no_outlier_GlobalGDP)
# print((GlobalGDP.shape[0] - no_outlier_GlobalGDP.shape[0])-len(GlobalGDP))


stdGlobalInflation = GlobalInflation.std()
avgGlobalInflation = GlobalInflation.mean()

three_sigma_plus_GlobalInflation = avgGlobalInflation + (0.775 * stdGlobalInflation)
three_sigma_minus_GlobalInflation = avgGlobalInflation - (0.775 * stdGlobalInflation)

no_outlier_GlobalInflation = GlobalInflation[
    (GlobalInflation < three_sigma_plus_GlobalInflation) & (GlobalInflation > three_sigma_minus_GlobalInflation)]
# print(no_outlier_GlobalInflation)

stdtwybp = twybp.std()
avgtwybp = twybp.mean()

three_sigma_plus_twybp = avgtwybp + (2.16 * stdtwybp)
three_sigma_minus_twybp = avgtwybp - (2.16 * stdtwybp)

no_outlier_twybp = twybp[(twybp < three_sigma_plus_twybp) & (twybp > three_sigma_minus_twybp)]
# print(no_outlier_twybp)

stdtybp = tybp.std()
avgtybp = tybp.mean()

three_sigma_plus_tybp = avgtybp + (3 * stdtybp)
three_sigma_minus_tybp = avgtybp - (3 * stdtybp)

# print(three_sigma_minus_tybp, three_sigma_plus_tybp)
no_outlier_tybp = tybp[(tybp < three_sigma_plus_tybp) & (tybp > three_sigma_minus_tybp)]
# print(no_outlier_tybp)
# print((tybp.shape[0] - no_outlier_tybp.shape[0]))

stdsnp = snp.std()
avgsnp = snp.mean()

three_sigma_plus_snp = avgsnp + (3 * stdsnp)
three_sigma_minus_snp = avgsnp - (3 * stdsnp)
no_outlier_snp = snp[(snp < three_sigma_plus_snp) & (snp > three_sigma_minus_snp)]
# print(no_outlier_snp)

stdUSInterest = USInterest.std()
avgUSInterest = USInterest.mean()

three_sigma_plus_USInterest = avgUSInterest + (0.9 * stdUSInterest)
three_sigma_minus_USInterest = avgUSInterest - (0.9 * stdUSInterest)

# no_outlier_USInterest = USInterest[(USInterest < three_sigma_plus_USInterest) & (USInterest > three_sigma_minus_USInterest)]
# print(len(no_outlier_USInterest))
# print(no_outlier_USInterest)

# plt.boxplot([no_outlier_USInflation.flatten(),no_outlier_GlobalInflation.flatten()], labels=['US','Global'])
# plt.show()
outlier = []

# plt.boxplot([USInflation,no_outlier_USInflation.flatten()], labels=['US Inflation', 'No Outlier US Inflation'])
# plt.title('US Inflation With and Without Outliers')
# plt.boxplot([USInterest,no_outlier_USInterest.flatten()], labels=['US Interest', 'No Outlier US Interest'])
# plt.title('US Interest With and Without Outliers')
# plt.boxplot([twybp,no_outlier_twybp.flatten()], labels=['2 Year Bond Price', 'No Outlier 2 Year Bond Price'])
# plt.title('2 Year Bond Price With and Without Outliers')
# plt.boxplot([tybp,no_outlier_tybp.flatten()], labels=['10 Year Bond Price', 'No Outlier 10 Year Bond Price'])
# plt.title('10 Year Bond Price With and Without Outliers')
# plt.boxplot([USInflation,no_outlier_USInflation.flatten()], labels=['US Inflation', 'No Outlier US Inflation'])
# plt.title('US Inflation With and Without Outliers')
# plt.boxplot([GlobalInflation,no_outlier_GlobalInflation.flatten()], labels=['Global Inflation', 'No Outlier Global Inflation'])
# plt.title('Global Inflation With and Without Outliers')
plt.show()
# k=0
# no_outlier_USInflation = USInflation.copy()
# no_outlier_GlobalInflation = GlobalInflation.copy()
# no_outlier_USInterest= USInterest.copy()
# no_outlier_twybp = twybp.copy()
# for x in range(len(no_outlier_USInflation)):
#    z= (no_outlier_USInflation[x]-avgUSInflation)/stdUSInflation
#    if np.abs(z) > 1.289 :
#        outlier.append(no_outlier_USInflation[x])
#        del(no_outlier_USInflation[x])
#        k=k+1
# k=0
# for x in range(len(no_outlier_GlobalInflation)):
#    z= (no_outlier_GlobalInflation[x]-avgGlobalInflation)/stdGlobalInflation
#    if np.abs(z) > 0.778 :
#        outlier.append(no_outlier_GlobalInflation[x])
#        del(no_outlier_GlobalInflation[x])
#        k=k+1
# k=0
# for x in range(len(no_outlier_USInterest)):
#    z= (no_outlier_USInterest[x]-avgUSInterest)/stdUSInterest
#    if np.abs(z) > 0.8996 :
#        outlier.append(no_outlier_USInterest[x])
#        del(no_outlier_USInterest[x])
#        k=k+1
# for x in range(len(no_outlier_twybp)):
#    z= (no_outlier_twybp[x]-avgtwybp)/stdtwybp
#    if np.abs(z) > 2.16:
#        outlier.append(no_outlier_twybp[x])
#        del(no_outlier_twybp[x])
#        k=k+1
# plt.boxplot([USInflation,no_outlier_USInflation], labels=['US Inflation', 'No Outlier US Inflation'])
# plt.title('US Inflation With and Without Outliers')
# plt.boxplot([GlobalInflation,no_outlier_GlobalInflation], labels=['Global Inflation', 'No Outlier Global Inflation'])
# plt.title('Global Inflation With and Without Outliers')
# plt.boxplot([USInterest,no_outlier_USInterest], labels=['US Interest', 'No Outlier US Interest'])
# plt.title('US Interest With and Without Outliers')
# plt.boxplot([twybp,no_outlier_twybp], labels=['2 Year Bond Price', 'No Outlier 2 Year Bond Price'])
# plt.title('2 Year Bond Price With and Without Outliers'),

# print(outlier)
# print(k)


no_outlier_snpopen = snpopen.copy()
no_outlier_snphigh = snphigh.copy()
x = 0
i = 0
no_outlier_USInflation = USInflation.copy()
for x in range(len(no_outlier_USInflation)):
    if (no_outlier_USInflation[x] > 7.5):
        no_outlier_USInflation[x] = 7.5
    if (no_outlier_USInflation[x] < 0.95):
        no_outlier_USInflation[x] = 0.95
# plt.plot(USInflation)
# plt.boxplot([USInflation])
x = 0
no_outlier_twybp = twybp.copy()
for x in range(len(no_outlier_twybp)):
    if (no_outlier_twybp[x] > 3.06):
        no_outlier_twybp[x] = 3.06
x = 0
no_outlier_twybpopen = twybpopen.copy()
for x in range(len(no_outlier_twybpopen)):
    if (no_outlier_twybpopen[x] > 3.05):
        no_outlier_twybpopen[x] = 3.05
x = 0
no_outlier_USInterest = USInterest.copy()
for x in range(len(no_outlier_USInterest)):
    if (no_outlier_USInterest[x] > 4.1):
        no_outlier_USInterest[x] = 4.1
x = 0
no_outlier_GlobalInflation = GlobalInflation.copy()
for x in range(len(no_outlier_GlobalInflation)):
    if (no_outlier_GlobalInflation[x] > 5.29):
        no_outlier_GlobalInflation[x] = 5.29
    if (no_outlier_GlobalInflation[x] < 0.2):
        no_outlier_GlobalInflation[x] = 0.2
x = 0
no_outlier_twybphigh = twybphigh.copy()
for x in range(len(no_outlier_twybphigh)):
    if (no_outlier_twybphigh[x] > 3.09):
        no_outlier_twybphigh[x] = 3.09

x = 0
no_outlier_twybplow = twybplow.copy()
for x in range(len(no_outlier_twybplow)):
    if (no_outlier_twybplow[x] > 3):
        no_outlier_twybplow[x] = 3

x = 0
no_outlier_twybpchange = twybpchange.copy()
for x in range(len(no_outlier_twybpchange)):
    if (no_outlier_twybpchange[x] > 0.07):
        no_outlier_twybpchange[x] = 0.07
    if (no_outlier_twybpchange[x] < -0.07):
        no_outlier_twybpchange[x] = -0.07

x = 0
no_outlier_tybpchange = tybpchange.copy()
for x in range(len(no_outlier_tybpchange)):
    if (no_outlier_tybpchange[x] > 0.05):
        no_outlier_tybpchange[x] = 0.05
    if (no_outlier_tybpchange[x] < -0.05):
        no_outlier_tybpchange[x] = -0.05

x = 0
no_outlier_snpchange = snpchange.copy()
for x in range(len(no_outlier_snpchange)):
    if (no_outlier_snpchange[x] > 0.02):
        no_outlier_snpchange[x] = 0.02
    if (no_outlier_snpchange[x] < -0.01):
        no_outlier_snpchange[x] = -0.01
x = 0
no_outlier_snphigh_price = snphigh_price.copy()
for x in range(len(no_outlier_snphigh_price)):
    if (no_outlier_snphigh_price[x] > 31.6):
        no_outlier_snphigh_price[x] = 31.6

x = 0
no_outlier_snpprice_low = snpprice_low.copy()
for x in range(len(no_outlier_snpprice_low)):
    if (no_outlier_snpprice_low[x] > 34.95):
        no_outlier_snpprice_low[x] = 34.95

x = 0
no_outlier_snphigh_low = snphigh_low.copy()
for x in range(len(no_outlier_snphigh_low)):
    if (no_outlier_snphigh_low[x] > 56.05):
        no_outlier_snphigh_low[x] = 56.05

no_outlier_snphigh_open = snphigh_open.copy()
for x in range(len(no_outlier_snphigh_open)):
    if (no_outlier_snphigh_open[x] > 34.3):
        no_outlier_snphigh_open[x] = 34.3

no_outlier_snpopen_low = snpopen_low.copy()
for x in range(len(no_outlier_snpopen_low)):
    if (no_outlier_snpopen_low[x] > 36.9):
        no_outlier_snpopen_low[x] = 36.9

no_outlier_tybhigh_price = tybhigh_price.copy()
for x in range(len(no_outlier_tybhigh_price)):
    if (no_outlier_tybhigh_price[x] > 0.07):
        no_outlier_tybhigh_price[x] = 0.07

no_outlier_tybprice_low = tybprice_low.copy()
for x in range(len(no_outlier_tybprice_low)):
    if (no_outlier_tybprice_low[x] > 0.07):
        no_outlier_tybprice_low[x] = 0.07

no_outlier_tybhigh_low = tybhigh_low.copy()
for x in range(len(no_outlier_tybhigh_low)):
    if (no_outlier_tybhigh_low[x] > 0.152):
        no_outlier_tybhigh_low[x] = 0.152

no_outlier_tybhigh_open = tybhigh_open.copy()
for x in range(len(no_outlier_tybhigh_open)):
    if (no_outlier_tybhigh_open[x] > 0.0725):
        no_outlier_tybhigh_open[x] = 0.0725

no_outlier_tybopen_low = tybopen_low.copy()
for x in range(len(no_outlier_tybopen_low)):
    if (no_outlier_tybopen_low[x] > 0.0725):
        no_outlier_tybopen_low[x] = 0.0725

no_outlier_twybhigh_price = twybhigh_price.copy()
for x in range(len(no_outlier_twybhigh_price)):
    if (no_outlier_twybhigh_price[x] > 0.0401):
        no_outlier_twybhigh_price[x] = 0.0401

no_outlier_twybprice_low = twybprice_low.copy()
for x in range(len(no_outlier_twybprice_low)):
    if (no_outlier_twybprice_low[x] > 0.0402):
        no_outlier_twybprice_low[x] = 0.0402

no_outlier_twybhigh_low = twybhigh_low.copy()
for x in range(len(no_outlier_twybhigh_low)):
    if (no_outlier_twybhigh_low[x] > 0.0925):
        no_outlier_twybhigh_low[x] = 0.0925

no_outlier_twybhigh_open = twybhigh_open.copy()
for x in range(len(no_outlier_twybhigh_open)):
    if (no_outlier_twybhigh_open[x] > 0.04):
        no_outlier_twybhigh_open[x] = 0.04

no_outlier_twybopen_low = twybopen_low.copy()
for x in range(len(no_outlier_twybopen_low)):
    if (no_outlier_twybopen_low[x] > 0.0399):
        no_outlier_twybopen_low[x] = 0.0399

no_outlier_snphigh = twybphigh.copy()

no_outlier_snplow = snplow.copy()

no_outlier_tybphigh = tybphigh.copy()

no_outlier_tybplow = tybplow.copy()

no_outlier_tybpopen = tybpopen.copy()

# plt.boxplot(no_outlier_snphigh_price)
# plt.boxplot(no_outlier_snpprice_low)
# plt.boxplot(no_outlier_snphigh_low)
# plt.boxplot(no_outlier_snphigh_open)
# plt.boxplot(no_outlier_snpopen_low)
# plt.boxplot(no_outlier_tybhigh_price)
# plt.boxplot(no_outlier_tybprice_low)
# plt.boxplot(no_outlier_tybhigh_low)
# plt.boxplot(no_outlier_tybhigh_open)
# plt.boxplot(no_outlier_tybopen_low)
# plt.boxplot(no_outlier_twybhigh_price)
# plt.boxplot(no_outlier_twybprice_low)
# plt.boxplot(no_outlier_twybhigh_low)
# plt.boxplot(no_outlier_twybhigh_open)
# plt.boxplot(no_outlier_twybopen_low)
# fig, axno_outlier_snpchange = plt.subplots()
# axno_outlier_snpchange.plot(date,no_outlier_snpchange)
# axno_outlier_snpchange.set_title('S&P 500 Change')
# axno_outlier_snpchange.set_xlabel('Date')
# axno_outlier_snpchange.set_ylabel('S&P 500 Change')
# plt.boxplot([snpchange,no_outlier_snpchange],labels= ['S&P 500 Change Change','No Outlier S&P 500 Change'])
# plt.title('S&P 500 Change With and Without Outliers')


# fig, axno_outlier_tybpchange = plt.subplots()
# axno_outlier_tybpchange.plot(date,no_outlier_tybpchange)
# axno_outlier_tybpchange.set_title('10 Year Bond Change')
# axno_outlier_tybpchange.set_xlabel('Date')
# axno_outlier_tybpchange.set_ylabel('10 Year Bond Change')
# plt.boxplot([tybpchange,no_outlier_tybpchange],labels= ['2 Year Bond Change','No Outlier 10 Year Bond Change'])
# plt.title('10 Year Bond Change With and Without Outliers')

# fig, axno_outlier_twybpchange = plt.subplots()
# axno_outlier_twybpchange.plot(date,no_outlier_twybpchange)
# axno_outlier_twybpchange.set_title('2 Year Bond Change')
# axno_outlier_twybpchange.set_xlabel('Date')
# axno_outlier_twybpchange.set_ylabel('2 Year Bond Change')
# plt.boxplot([twybpchange,no_outlier_twybpchange],labels= ['2 Year Bond Change','No Outlier 2 Year Bond Change'])
# plt.title('2 Year Bond Change With and Without Outliers')

# fig, axno_outlier_twybpchange = plt.subplots()
# axno_outlier_twybpchange.plot(date,no_outlier_twybpchange)
# axno_outlier_twybpchange.set_title('2 Year Bond Change')
# axno_outlier_twybpchange.set_xlabel('Date')
# axno_outlier_twybpchange.set_ylabel('2 Year Bond Change')
# plt.boxplot([twybpchange,no_outlier_twybpchange],labels= ['2 Year Bond Change','No Outlier 2 Year Bond Change'])
# plt.title('2 Year Bond Change With and Without Outliers')

# fig, axno_outlier_twybplow = plt.subplots()
# axno_outlier_twybplow.plot(date,no_outlier_twybplow)
# axno_outlier_twybplow.set_title('2 Year Bond Low')
# axno_outlier_twybplow.set_xlabel('Date')
# axno_outlier_twybplow.set_ylabel('2 Year Bond Low')
# plt.boxplot([twybplow,no_outlier_twybplow],labels= ['2 Year Bond Low','No Outlier 2 Year Bond Low'])
# plt.title('2 Year Bond Low With and Without Outliers')

# fig, axno_outlier_twybphigh = plt.subplots()
# axno_outlier_twybphigh.plot(date,no_outlier_twybphigh)
# axno_outlier_twybphigh.set_title('2 Year Bond High')
# axno_outlier_twybphigh.set_xlabel('Date')
# axno_outlier_twybphigh.set_ylabel('2 Year Bond High')
# plt.boxplot([twybphigh,no_outlier_twybphigh],labels= ['2 Year Bond High','No Outlier 2 Year Bond High'])
# plt.title('2 Year Bond High With and Without Outliers')

# fig, axno_outlier_GlobalInflation = plt.subplots()
# axno_outlier_GlobalInflation.plot(date,no_outlier_GlobalInflation)
# axno_outlier_GlobalInflation.set_title('Global Inflation')
# axno_outlier_GlobalInflation.set_xlabel('Date')
# axno_outlier_GlobalInflation.set_ylabel('Global Inflation')
# plt.boxplot([GlobalInflation,no_outlier_GlobalInflation],labels= ['Global Inflation With Outliers','No Outlier Global Inflation'])
# plt.title('Global Infaltion With and Without Outliers Box Plot')


# fig, axno_outlier_USInflation = plt.subplots()
# axno_outlier_USInflation.plot(date,no_outlier_USInflation)
# axno_outlier_USInflation.set_title('US Inflation')
# axno_outlier_USInflation.set_xlabel('Date')
# axno_outlier_USInflation.set_ylabel('US Inflation')
# plt.boxplot([USInflation,no_outlier_USInflation],labels= ['US Inflation','No Outlier US Inflation'])
# plt.title('US Inflation With and Without Outliers')


# fig, axno_outlier_USInterest = plt.subplots()
# axno_outlier_USInterest.plot(date,no_outlier_USInterest)
# axno_outlier_USInterest.set_title('US Interest')
# axno_outlier_USInterest.set_xlabel('Date')
# axno_outlier_USInterest.set_ylabel('US Interest')
# plt.boxplot([USInterest,no_outlier_USInterest],labels=['US Interest','No Outlier US Interest'])
# plt.title('US Interest With and Without Outliers')

# fig, axno_outlier_twybpopen = plt.subplots()
# axno_outlier_twybpopen.plot(date,no_outlier_twybp)
# axno_outlier_twybpopen.set_title('2 Year Bond Open')
# axno_outlier_twybpopen.set_xlabel('Date')
# axno_outlier_twybpopen.set_ylabel('2 Year Bond Open')
# plt.boxplot([twybpopen,no_outlier_twybpopen],labels= ['2 Year Bond Open With Outliers','No Outlier 2 Year Bond Open'])
# plt.title('2 Year Bond Open With and Without Outliers')

# twybp = twybp[::-1]
# fig, axno_outlier_twybpopen = plt.subplots()
# axno_outlier_twybpopen.plot(date,no_outlier_twybpopen)
# axno_outlier_twybpopen.set_title('2 Year Bond Open')
# axno_outlier_twybpopen.set_xlabel('Date')
# axno_outlier_twybpopen.set_ylabel('2 Year Bond Open')
# plt.boxplot([twybpopen,no_outlier_twybpopen],labels= ['2 Year Bond Open With Outliers','No Outlier 2 Year Bond Open'])
# plt.title('2 Year Bond Open With and Without Outliers')

# fig, axno_outlier_twybphigh = plt.subplots()
# axno_outlier_twybphigh.plot(date,no_outlier_twybphigh)
# axno_outlier_twybphigh.set_title('2 Year Bond High')
# axno_outlier_twybphigh.set_xlabel('Date')
# axno_outlier_twybphigh.set_ylabel('2 Year Bond High')
# plt.boxplot([twybphigh,no_outlier_twybphigh],labels= ['2 Year Bond High With Outliers','No Outlier 2 Year Bond High'])
# plt.title('2 Year Bond High With and Without Outliers')

# fig, axno_outlier_twybplow = plt.subplots()
# axno_outlier_twybplow.plot(date,no_outlier_twybplow)
# axno_outlier_twybplow.set_title('2 Year Bond Low')
# axno_outlier_twybplow.set_xlabel('Date')
# axno_outlier_twybplow.set_ylabel('2 Year Bond Low')
# plt.boxplot([twybplow,no_outlier_twybplow],labels= ['2 Year Bond Low With Outliers','No Outlier 2 Year Bond Low'])
# plt.title('2 Year Bond Low With and Without Outliers')

# fig, axno_outlier_twybphigh = plt.subplots()
# axno_outlier_twybphigh.plot(date,no_outlier_twybphigh)
# axno_outlier_twybphigh.set_title('2 Year Bond High')
# axno_outlier_twybphigh.set_xlabel('Date')
# axno_outlier_twybphigh.set_ylabel('2 Year Bond High')
# plt.boxplot([twybhigh_price,no_outlier_twybhigh_price],labels= ['2 Year Bond High - Price With Outliers','No Outlier 2 Year Bond High- Price'])
# plt.title('2 Year Bond High - Price With and Without Outliers')


# plt.boxplot([twybprice_low,no_outlier_twybprice_low],labels= ['2 Year Bond Price - Low With Outliers','No Outlier 2 Year Bond Price - Low'])
# plt.title('2 Year Bond Price - Low With and Without Outliers')

# plt.boxplot([twybhigh_low,no_outlier_twybhigh_low],labels= ['2 Year Bond High - Low With Outliers','No Outlier 2 Year Bond High - Low'])
# plt.title('2 Year Bond High - Low With and Without Outliers')

# plt.boxplot([twybhigh_open,no_outlier_twybhigh_open],labels= ['2 Year Bond High - Open With Outliers','No Outlier 2 Year Bond High - Open'])
# plt.title('2 Year Bond High - Open With and Without Outliers')

# plt.boxplot([twybopen_low,no_outlier_twybopen_low],labels= ['2 Year Bond Open - Low With Outliers','No Outlier 2 Year Bond Open - Low'])
# plt.title('2 Year Bond Open - Low With and Without Outliers')

# plt.boxplot([snphigh_price,no_outlier_snphigh_price],labels= ['S&P 500 High - Price With Outliers','No Outlier S&P 500 High - Price'])
# plt.title('S&P 500 High - Price With and Without Outliers')

# plt.boxplot([snpprice_low,no_outlier_snpprice_low],labels= ['S&P 500 Price - Low With Outliers','No Outlier S&P 500 Price - Low'])
# plt.title('S&P 500 Price - Low With and Without Outliers')

# plt.boxplot([snphigh_low,no_outlier_snphigh_low],labels= ['S&P 500 High - Low With Outliers','No Outlier S&P 500 High - Low'])
# plt.title('S&P 500 High - Low With and Without Outliers')

# plt.boxplot([snphigh_open,no_outlier_snphigh_open],labels= ['S&P 500 High - Open With Outliers','No Outlier S&P 500 High - Open'])
# plt.title('S&P 500 High - Open With and Without Outliers')

# plt.boxplot([snpopen_low,no_outlier_snpopen_low],labels= ['S&P 500 Open - Low With Outliers','No Outlier S&P 500 Open - Low'])
# plt.title('S&P 500 Open - Low With and Without Outliers')

# plt.boxplot([tybhigh_price,no_outlier_tybhigh_price],labels= ['10 Year Bond High - Price With Outliers','No Outlier 10 Year Bond High - Price'])
# plt.title('10 Year Bond High - Price With and Without Outliers')

# plt.boxplot([tybprice_low,no_outlier_tybprice_low],labels= ['10 Year Bond Price - Low With Outliers','No Outlier 10 Year Bond Price - Low'])
# plt.title('10 Year Bond Price - Low With and Without Outliers')

# plt.boxplot([tybhigh_low,no_outlier_tybhigh_low],labels= ['10 Year Bond High - Low With Outliers','No Outlier 10 Year Bond High - Low'])
# plt.title('10 Year Bond High - Low With and Without Outliers')

# plt.boxplot([tybhigh_open,no_outlier_tybhigh_open],labels= ['10 Year Bond High - Open With Outliers','No Outlier 10 Year Bond High - Open'])
# plt.title('10 Year Bond High - Open With and Without Outliers')
#
# plt.boxplot([tybopen_low,no_outlier_tybopen_low],labels= ['10 Year Bond Open - Low With Outliers','No Outlier 10 Year Bond Open - Low'])
# plt.title('10 Year Bond Open - Low With and Without Outliers')


#######Quartile Method#######
# Q1_USInterest = np.percentile(USInterest,25)
# Q3_USInterest = np.percentile(USInterest,75)
# IQR_USInterest = Q3_USInterest - Q1_USInterest
# low_range_USInterest = Q1_USInterest - (1.5*IQR_USInterest)
# up_range_USInterest = Q3_USInterest - (1.5*IQR_USInterest)
# index = []
# no_outlier_USInterest = []
# x=0
# for x in range(len(USInterest)):
#     if((USInterest[x] <low_range_USInterest) | (USInterest[x]>up_range_USInterest)):
#         index.append(x)
#     else:
#         no_outlier_USInterest.append(USInterest[x])

#######Normalization#######
scaler = MinMaxScaler()
no_outlier_USGDP = USGDP.copy()
no_outlier_USGDP = no_outlier_USGDP.values.reshape(-1, 1)
normalised_USGDP = scaler.fit_transform(no_outlier_USGDP)
# print(normalised_USGDP)

no_outlier_GlobalGDP = no_outlier_GlobalGDP.values.reshape(-1, 1)
normalised_GlobalGDP = scaler.fit_transform(no_outlier_GlobalGDP)

no_outlier_USInflation = no_outlier_USInflation.values.reshape(-1, 1)
normalised_USInflation = scaler.fit_transform(no_outlier_USInflation)

no_outlier_GlobalInflation = no_outlier_GlobalInflation.values.reshape(-1, 1)
normalised_GlobalInflation = scaler.fit_transform(no_outlier_GlobalInflation)

no_outlier_USInterest = no_outlier_USInterest.values.reshape(-1, 1)
normalised_USInterest = scaler.fit_transform(no_outlier_USInterest)

no_outlier_snp = no_outlier_snp.values.reshape(-1, 1)
normalised_snp = scaler.fit_transform(no_outlier_snp)

no_outlier_snpopen = no_outlier_snpopen.values.reshape(-1, 1)
normalised_snpopen = scaler.fit_transform(no_outlier_snpopen)

no_outlier_snphigh = no_outlier_snphigh.values.reshape(-1, 1)
normalised_snphigh = scaler.fit_transform(no_outlier_snphigh)

no_outlier_snplow = no_outlier_snplow.values.reshape(-1, 1)
normalised_snplow = scaler.fit_transform(no_outlier_snplow)

no_outlier_snpchange = no_outlier_snpchange.values.reshape(-1, 1)
normalised_snpchange = scaler.fit_transform(no_outlier_snpchange)

no_outlier_tybp = no_outlier_tybp.values.reshape(-1, 1)
normalised_tybp = scaler.fit_transform(no_outlier_tybp)

no_outlier_tybpopen = no_outlier_tybpopen.values.reshape(-1, 1)
normalised_tybpopen = scaler.fit_transform(no_outlier_tybpopen)

no_outlier_tybphigh = no_outlier_tybphigh.values.reshape(-1, 1)
normalised_tybphigh = scaler.fit_transform(no_outlier_tybphigh)

no_outlier_tybplow = no_outlier_tybplow.values.reshape(-1, 1)
normalised_tybplow = scaler.fit_transform(no_outlier_tybplow)

no_outlier_tybpchange = no_outlier_tybpchange.values.reshape(-1, 1)
normalised_tybpchange = scaler.fit_transform(no_outlier_tybpchange)

no_outlier_twybp = no_outlier_twybp.values.reshape(-1, 1)
normalised_twybp = scaler.fit_transform(no_outlier_twybp)

no_outlier_twybpopen = no_outlier_twybpopen.values.reshape(-1, 1)
normalised_twybpopen = scaler.fit_transform(no_outlier_twybpopen)

no_outlier_twybphigh = no_outlier_twybphigh.values.reshape(-1, 1)
normalised_twybphigh = scaler.fit_transform(no_outlier_twybphigh)

no_outlier_twybplow = no_outlier_twybplow.values.reshape(-1, 1)
normalised_twybplow = scaler.fit_transform(no_outlier_twybplow)

no_outlier_twybpchange = no_outlier_twybpchange.values.reshape(-1, 1)
normalised_twybpchange = scaler.fit_transform(no_outlier_twybpchange)

no_outlier_snphigh_price = no_outlier_snphigh_price.values.reshape(-1, 1)
normalised_snphigh_price = scaler.fit_transform(no_outlier_snphigh_price)

no_outlier_snpprice_low = no_outlier_snpprice_low.values.reshape(-1, 1)
normalised_snpprice_low = scaler.fit_transform(no_outlier_snpprice_low)

no_outlier_snphigh_low = no_outlier_snphigh_low.values.reshape(-1, 1)
normalised_snphigh_low = scaler.fit_transform(no_outlier_snphigh_low)

no_outlier_snphigh_open = no_outlier_snphigh_open.values.reshape(-1, 1)
normalised_snphigh_open = scaler.fit_transform(no_outlier_snphigh_open)

no_outlier_snpopen_low = no_outlier_snpopen_low.values.reshape(-1, 1)
normalised_snpopen_low = scaler.fit_transform(no_outlier_snpopen_low)

no_outlier_tybhigh_price = no_outlier_tybhigh_price.values.reshape(-1, 1)
normalised_tybhigh_price = scaler.fit_transform(no_outlier_tybhigh_price)

no_outlier_tybprice_low = no_outlier_tybprice_low.values.reshape(-1, 1)
normalised_tybprice_low = scaler.fit_transform(no_outlier_tybprice_low)

no_outlier_tybhigh_low = no_outlier_tybhigh_low.values.reshape(-1, 1)
normalised_tybhigh_low = scaler.fit_transform(no_outlier_tybhigh_low)

no_outlier_tybhigh_open = no_outlier_tybhigh_open.values.reshape(-1, 1)
normalised_tybhigh_open = scaler.fit_transform(no_outlier_tybhigh_open)

no_outlier_tybopen_low = no_outlier_tybopen_low.values.reshape(-1, 1)
normalised_tybopen_low = scaler.fit_transform(no_outlier_tybopen_low)

no_outlier_twybhigh_price = no_outlier_twybhigh_price.values.reshape(-1, 1)
normalised_twybhigh_price = scaler.fit_transform(no_outlier_twybhigh_price)

no_outlier_twybprice_low = no_outlier_twybprice_low.values.reshape(-1, 1)
normalised_twybprice_low = scaler.fit_transform(no_outlier_twybprice_low)

no_outlier_twybhigh_low = no_outlier_twybhigh_low.values.reshape(-1, 1)
normalised_twybhigh_low = scaler.fit_transform(no_outlier_twybhigh_low)

no_outlier_twybhigh_open = no_outlier_twybhigh_open.values.reshape(-1, 1)
normalised_twybhigh_open = scaler.fit_transform(no_outlier_twybhigh_open)

no_outlier_twybopen_low = no_outlier_twybopen_low.values.reshape(-1, 1)
normalised_twybopen_low = scaler.fit_transform(no_outlier_twybopen_low)
date = date.values.reshape(-1,1)
# print(no_outlier_snp)
# no_outlier_snp = no_outlier_snp.reshape(1,-1)
# print(no_outlier_snp.shape)
# no_outlier_snp=no_outlier_snp.flatten()
# print(no_outlier_snp.shape)
# no_outlier_snpopen = no_outlier_snpopen.flatten()
# no_outlier_snphigh = no_outlier_snphigh.flatten()
# no_outlier_snplow = no_outlier_snplow.flatten()
# no_outlier_snpchange = no_outlier_snpchange.flatten()
# no_outlier_tybp = no_outlier_tybp.flatten()
# no_outlier_tybpopen = no_outlier_tybpopen.flatten()
# no_outlier_tybphigh = no_outlier_tybphigh.flatten()
# no_outlier_twybplow = no_outlier_twybplow.flatten()
# no_outlier_twybpchange = no_outlier_twybpchange.flatten()
# no_outlier_USInflation = no_outlier_USInflation.flatten()
# no_outlier_USGDP = no_outlier_USGDP.flatten()
# no_outlier_USInterest = no_outlier_USInterest.flatten()
# no_outlier_snpprice_low = no_outlier_snpprice_low.flatten()
# no_outlier_snphigh_low = no_outlier_snphigh_low.flatten()
# no_outlier_snphigh_open = no_outlier_snphigh_open.flatten()
# no_outlier_snpopen_low = no_outlier_snpopen_low.flatten()
# no_outlier_tybhigh_price = no_outlier_tybhigh_price.flatten()
# no_outlier_snphigh_price = no_outlier_snphigh_price.flatten()
# no_outlier_GlobalGDP = no_outlier_GlobalGDP.flatten()
# no_outlier_GlobalInflation = no_outlier_GlobalInflation.flatten()
# no_outlier_tybprice_low = no_outlier_tybprice_low.flatten()
# no_outlier_tybhigh_low = no_outlier_tybhigh_low.flatten()
# no_outlier_tybhigh_open = no_outlier_tybhigh_open.flatten()
# no_outlier_tybopen_low = no_outlier_tybopen_low.flatten()
# no_outlier_twybhigh_price = no_outlier_twybhigh_price.flatten()
# no_outlier_twybprice_low= no_outlier_twybprice_low.flatten()
# no_outlier_twybhigh_low = no_outlier_twybhigh_low.flatten()
# no_outlier_twybhigh_open = no_outlier_twybhigh_open.flatten()
# no_outlier_twybopen_low = no_outlier_twybopen_low.flatten()
# no_outlier_data = pd.DataFrame({'S&P 500 Price': no_outlier_snp,
#                         'S&P 500 Open': no_outlier_snpopen,
#                         'S&P 500 High': no_outlier_snphigh,
#                         'S&P 500 Low': no_outlier_snplow,
#                         'S&P 500 Change': no_outlier_snpchange,
#                         '10 Year Bond Price': no_outlier_tybp,
#                         '10 Year Bond Open': no_outlier_tybpopen,
#                         '10 Year Bond High': no_outlier_tybphigh,
#                         '10 Year Bond Low': no_outlier_tybplow,
#                         '10 Year Bond Change': no_outlier_tybpchange,
#                         '2 Year Bond Price': no_outlier_twybp,
#                         '2 Year Bond Open': no_outlier_twybpopen,
#                         '2 Year Bond High': no_outlier_twybphigh,
#                         '2 Year Bond Low': no_outlier_twybplow,
#                         '2 Year Bond Change': no_outlier_twybpchange,
#                         'US Inflation': no_outlier_USInflation,
#                         'US GDP': no_outlier_USGDP,
#                         'US Insterest': no_outlier_USInterest,
#                         'Global Inflation Rate': no_outlier_GlobalInflation,
#                         'Global GDP': no_outlier_GlobalGDP,
#                         'S&P 500 High - Price': no_outlier_snphigh_price,
#                         'S&P 500 Price -Low': no_outlier_snpprice_low,
#                         'S&P 500 High- Low': no_outlier_snphigh_low,
#                         'S&P 500 High - Open': no_outlier_snphigh_open,
#                         'S&P 500 Open - Low': no_outlier_snpopen_low,
#                         '10 Year Bond High - Price': no_outlier_tybhigh_price,
#                         '10 Year Bond Price -Low': no_outlier_tybprice_low,
#                         '10 Year Bond High - Low': no_outlier_tybhigh_low,
#                         '10 Year Bond High - Open': no_outlier_tybhigh_open,
#                         '10 Year Bond Open - Low': no_outlier_tybopen_low,
#                         '2 Year Bond High - Price': no_outlier_twybhigh_price,
#                         '2 Year Bond Price -Low': no_outlier_twybprice_low,
#                         '2 Year Bond High - Low': no_outlier_twybhigh_low,
#                         '2 Year Bond High - Open': no_outlier_twybhigh_open,
#                         '2 Year Bond Open - Low': no_outlier_twybopen_low})
# no_outlier_data_df = pd.DataFrame(data = no_outlier_data, index=[0])
# no_outlier_data.to_excel("C:\\Users\\USER\\Desktop\\NoOutlierData.csv",index=False)
# plt.plot(normalised_tybp)
# plt.show()

# alldataset = np.concatenate((normalised_snp,normalised_twybp))
# x=0
# for x in range(len(alldataset)):
#     print(alldataset[x])

# date = date.transpose()
# normalised_snp = normalised_snp.transpose()
# normalised_twybp = normalised_twybp.transpose()
# normalised_tybp = normalised_tybp.transpose()
# normalised_USGDP = normalised_USGDP.transpose()
# normalised_USInflation = normalised_USInflation.transpose()
# normalised_USInterest = normalised_USInterest.transpose()
# normalised_GlobalInflation = normalised_GlobalInflation.transpose()
# normalised_GlobalGDP = normalised_GlobalGDP.transpose()

# date = date.tolist()
# normalised_snp = normalised_snp.tolist()
# normalised_tybp = normalised_tybp.tolist()
# normalised_twybp = normalised_twybp.tolist()
# normalised_USInflation = normalised_USInflation.tolist()
# normalised_USGDP = normalised_USGDP.tolist()
# normalised_USInterest = normalised_USInterest.tolist()
# normalised_GlobalInflation = normalised_GlobalInflation.tolist()
# normalised_GlobalGDP = normalised_GlobalGDP.tolist()
# normalised_snp = np.array(normalised_snp)
# normalised_tybp = np.array(normalised_tybp)
# normalised_twybp = np.array(normalised_twybp)
# normalised_USInflation = np.array(normalised_USInflation)
# normalised_USGDP = np.array(normalised_USGDP)
# normalised_USInterest = np.array(normalised_USInterest)
# normalised_GlobalInflation = np.array(normalised_GlobalInflation)
# normalised_GlobalGDP = np.array(normalised_GlobalGDP)
# print(normalised_snp.shape,normalised_tybp.shape,normalised_twybp.shape,normalised_USInflation.shape,normalised_USGDP.shape,normalised_USInterest.shape,normalised_GlobalInflation.shape,normalised_GlobalGDP.shape)
# plt.plot(date,normalised_snp,color="green", label= 'S&P 500 Price')
# plt.plot(date,normalised_tybp,color="red", label= '10 Year Bond Price')
# plt.plot(date,normalised_twybp,color="blue", label= '2 Year Bond Price')
# plt.plot(date,normalised_USInflation,color="red",label= 'US Inflation')
# plt.plot(date,normalised_USGDP,color="blue",label= 'US GDP')
# plt.plot(date,normalised_USInterest,color="red",label= 'US Interest')
# plt.plot(date,normalised_GlobalInflation,color="green",label= 'Global Inflation')
# plt.plot(date,normalised_GlobalGDP,color="red",label= 'Global GDP')
# plt.xlabel('Date')
# plt.ylabel('Values')
# plt.legend()
# plt.title('Normalised Market and Bond Prices')
# plt.title('Normalised Inflation Rates')
# plt.title('Normalised US Interest Rate')
# plt.title('Normalised GDP')
plt.show()
# normalised_snp = normalised_snp.reshape(1,-1)
# normalised_twybp = normalised_twybp.reshape(1,-1)
# normalised_tybp = normalised_tybp.reshape(1,-1)
# normalised_USGDP = normalised_USGDP.reshape(1,-1)
# normalised_USInterest = normalised_USInterest.reshape(1,-1)
# normalised_USInflation = normalised_USInterest.reshape(1,-1)
# normalised_GlobalGDP = normalised_GlobalGDP.reshape(1,-1)
# normalised_GlobalInflation = normalised_GlobalInflation.reshape(1,-1)
# normalised_snp = np.array(normalised_snp)
# normalised_tybp = np.array(normalised_tybp)
# normalised_twybp = np.array(normalised_twybp)
# normalised_USGDP = np.array(normalised_USGDP)
# normalised_USInterest = np.array(normalised_USInterest)
# normalised_USInflation = np.array(normalised_USInflation)
# normalised_GlobalGDP = np.array(normalised_GlobalGDP)
# normalised_GlobalInflation = np.array(normalised_GlobalInflation)
# date = date.transpose()

# print(date.shape,normalised_snp.shape,normalised_twybp.shape,normalised_USInflation.shape,normalised_USGDP.shape,normalised_USInterest.shape,normalised_GlobalInflation.shape,normalised_GlobalGDP.shape)
date = date.flatten()
normalised_snp = normalised_snp.flatten()
normalised_tybp = normalised_tybp.flatten()
normalised_twybp = normalised_twybp.flatten()
normalised_USInflation = normalised_USInflation.flatten()
normalised_USInterest = normalised_USInterest.flatten()
normalised_USGDP = normalised_USGDP.flatten()
normalised_GlobalGDP = normalised_GlobalGDP.flatten()
normalised_GlobalInflation = normalised_GlobalInflation.flatten()
normalised_twybpopen = normalised_twybpopen.flatten()
normalised_tybpopen = normalised_tybpopen.flatten()
normalised_snpopen = normalised_snpopen.flatten()
normalised_twybphigh = normalised_twybphigh.flatten()
normalised_snphigh = normalised_snphigh.flatten()
normalised_tybphigh = normalised_tybphigh.flatten()
normalised_twybplow = normalised_twybplow.flatten()
normalised_tybplow = normalised_tybplow.flatten()
normalised_snplow = normalised_snplow.flatten()
normalised_twybpchange = normalised_twybpchange.flatten()
normalised_tybpchange = normalised_tybpchange.flatten()
normalised_snpchange = normalised_snpchange.flatten()
normalised_snphigh_price = normalised_snphigh_price.flatten()
normalised_snphigh_open = normalised_snphigh_open.flatten()
normalised_snpopen_low = normalised_snpopen_low.flatten()
normalised_snpprice_low = normalised_snpprice_low.flatten()
normalised_snphigh_low = normalised_snphigh_low.flatten()
normalised_tybhigh_price = normalised_tybhigh_price.flatten()
normalised_tybhigh_open = normalised_tybhigh_open.flatten()
normalised_tybprice_low = normalised_tybprice_low.flatten()
normalised_tybhigh_low = normalised_tybhigh_low.flatten()
normalised_tybopen_low = normalised_tybopen_low.flatten()
normalised_twybhigh_price = normalised_twybhigh_price.flatten()
normalised_twybprice_low = normalised_twybprice_low.flatten()
normalised_twybhigh_low = normalised_twybhigh_low.flatten()
normalised_twybhigh_open = normalised_twybhigh_open.flatten()
normalised_twybopen_low = normalised_twybopen_low.flatten()
# print(normalised_snp.shape)
# print(len(normalised_snp),len(normalised_tybp),len(normalised_twybp),len(normalised_USInflation),len(normalised_USGDP),len(normalised_USInterest),len(normalised_GlobalInflation),len(normalised_GlobalGDP))
alldata = pd.DataFrame({'Date': date,
                        'S&P 500 Price': normalised_snp,
                        'S&P 500 Open': normalised_snpopen,
                        'S&P 500 High': normalised_snphigh,
                        'S&P 500 Low': normalised_snplow,
                        'S&P 500 Change': normalised_snpchange,
                        '10 Year Bond Price': normalised_tybp,
                        '10 Year Bond Open': normalised_tybpopen,
                        '10 Year Bond High': normalised_tybphigh,
                        '10 Year Bond Low': normalised_tybplow,
                        '10 Year Bond Change': normalised_tybpchange,
                        '2 Year Bond Price': normalised_twybp,
                        '2 Year Bond Open': normalised_twybpopen,
                        '2 Year Bond High': normalised_twybphigh,
                        '2 Year Bond Low': normalised_twybplow,
                        '2 Year Bond Change': normalised_twybpchange,
                        'US Inflation': normalised_USInflation,
                        'US GDP': normalised_USGDP,
                        'US Insterest': normalised_USInterest,
                        'Global Inflation Rate': normalised_GlobalInflation,
                        'Global GDP': normalised_GlobalGDP,
                        'S&P 500 High - Price': normalised_snphigh_price,
                        'S&P 500 Price -Low': normalised_snpprice_low,
                        'S&P 500 High- Low': normalised_snphigh_low,
                        'S&P 500 High - Open': normalised_snphigh_open,
                        'S&P 500 Open - Low': normalised_snpopen_low,
                        '10 Year Bond High - Price': normalised_tybhigh_price,
                        '10 Year Bond Price -Low': normalised_tybprice_low,
                        '10 Year Bond High - Low': normalised_tybhigh_low,
                        '10 Year Bond High - Open': normalised_tybhigh_open,
                        '10 Year Bond Open - Low': normalised_tybopen_low,
                        '2 Year Bond High - Price': normalised_twybhigh_price,
                        '2 Year Bond Price -Low': normalised_twybprice_low,
                        '2 Year Bond High - Low': normalised_twybhigh_low,
                        '2 Year Bond High - Open': normalised_twybhigh_open,
                        '2 Year Bond Open - Low': normalised_twybopen_low})

# pd.describe_option('display')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
# pd.set_option('display.max_colwidth', None)
# print(alldata)
# firsttendataframerow = alldata.head(10)
# print(firsttendataframerow)
# print(alldata.head(10))
# alldata= pd.DataFrame(alldata)
# alldata = alldata.to_numpy()
# print(alldata)
# alldata = np.reshape(np.array(alldata),alldata[0],alldata[1],1)
# array_alldata = pd.DataFrame(alldata)
#########Correaltion Matrix and Variance############
# correlation_matrix = alldata.corr()
# variance_alldata = alldata.var()
# print(variance_alldata)
# print(correlation_matrix)

# sns.heatmap(data=correlation_matrix,annot=True, xticklabels=True, yticklabels=True)

alldata = pd.DataFrame({'Date': date,
                        'S&P 500 Price': normalised_snp,
                        'S&P 500 Change': normalised_snpchange,
                        '10 Year Bond Price': normalised_tybp,
                        '10 Year Bond Change': normalised_tybpchange,
                        '2 Year Bond Price': normalised_twybp,
                        '2 Year Bond Change': normalised_twybpchange,
                        'US Inflation': normalised_USInflation,
                        'US GDP': normalised_USGDP,
                        'US Interest': normalised_USInterest,
                        'Global Inflation Rate': normalised_GlobalInflation,
                        'Global GDP': normalised_GlobalGDP,
                        'S&P 500 High - Price': normalised_snphigh_price,
                        'S&P 500 Price -Low': normalised_snpprice_low,
                        'S&P 500 High- Low': normalised_snphigh_low,
                        'S&P 500 High - Open': normalised_snphigh_open,
                        'S&P 500 Open - Low': normalised_snpopen_low,
                        '10 Year Bond High - Price': normalised_tybhigh_price,
                        '10 Year Bond Price -Low': normalised_tybprice_low,
                        '10 Year Bond High - Low': normalised_tybhigh_low,
                        '10 Year Bond High - Open': normalised_tybhigh_open,
                        '10 Year Bond Open - Low': normalised_tybopen_low,
                        '2 Year Bond High - Price': normalised_twybhigh_price,
                        '2 Year Bond Price -Low': normalised_twybprice_low,
                        '2 Year Bond High - Low': normalised_twybhigh_low,
                        '2 Year Bond High - Open': normalised_twybhigh_open,
                        '2 Year Bond Open - Low': normalised_twybopen_low})
pd.describe_option('display')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_colwidth', None)
print(alldata)
# alldata.to_csv("C:\\Users\\USER\\Desktop\\FeatureSelectedData.csv",index=False)

####### Linear Regression ########
# X=alldata['2 Year Bond High - Low']
# y= alldata['2 Year Bond Price']
# X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=0)
# X_train = X_train.values.reshape(-1,1)
# y_train = y_train.values.reshape(-1,1)
# X_test = X_test.values.reshape(-1,1)
# y_test = y_test.values.reshape(-1,1)
# LR_model = LinearRegression()
# LR_model.fit(X_train,y_train)
# y_pred = LR_model.predict(X_test)
# print('Linear Regression Score', LR_model.score(X_test,y_test))
# CV_score = cross_val_score(LR_model,X_train,y_train, scoring ='r2', cv=10)
# print('Cross Valıdation Score :' )
# for k in range(len(CV_score)):
#     print(CV_score[k])
# print('R2 Score of Linear Regression: ',r2_score(y_test,y_pred))
# print('Average CV Score : ',np.mean(CV_score))
# print('Mean Absolute Error : ', mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error : ',mean_squared_error(y_test,y_pred))
# plt.scatter(X_test,y_test, label='Test Values', color='green')
# plt.plot(X_test,LR_model.predict(X_test),label='Linear Regression Model',color='red')
# plt.legend()
# plt.title('Linear Regression Plot')
# intercept = LR_model.intercept_
# print('Intercept value of Simple Linear Regression: ',intercept)
# coeffs = LR_model.coef_
# print('Coeficient value of Simple Linear Regression: ',coeffs)

# plt.plot(X)
# plt.scatter(X_test,y_test, label= 'Test Data', color= 'green')
# plt.legend()
# plt.show()
# print(LR_model.score(X_test.values.reshape(-1,1),y_test.values))
####### Multivariable Regression ########

X = alldata.drop(columns=['S&P 500 Price', 'Date'])
y = alldata['S&P 500 Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# X_train = X_train.values.reshape(-1,1)
# y_train =y_train.values.reshape(-1,1)
# X_test = X_test.values.reshape(-1,1)
# y_test = y_test.values.reshape(-1,1)
# print(y_train.shape)
# print(y_test.shape)
# print(X_train.shape)
# print(X_test.shape)
LR_model = LinearRegression()
LR_model.fit(X_train,y_train)
y_pred = LR_model.predict(X_test)
plt.plot(date,snp)
plt.xlabel('Date')
plt.ylabel('S&P 500 Price')
plt.title('S&P 500 Price Plot')
plt.show()
# plt.scatter(y_pred, y_test, alpha= 0.6)
# plt.title('Multivariable Regression Scatter')
# plt.xlabel('Actual Data')
# plt.ylabel('Test Data')
# intercept = LR_model.intercept_
# print('Intercept value of Multivariale Regression: ',intercept)
# coeffs = LR_model.coef_
# print('Coeficient values of Multivaraible Regression: ')
# print(coeffs)
# y_pred = y_pred.reshape(-1,1)
# plt.plot(y_pred,label ='Predicted Data',color='red',alpha=0.7,linestyle='dashed')
# plt.plot(y_test,label = 'Actual Data',color='green',alpha =0.7)
# plt.title('Multivariable Regression Plot')
# plt.legend()
# plt.show()
# print('Mean Absolute Error : ',mean_absolute_error(y_test, y_pred))
# print('R2 Score : ',r2_score(y_test,y_pred))
# print('Mean Squared Erorr : ', mean_squared_error(y_test,y_pred))
# print('Multivariable Regression Score', LR_model.score(X_test,y_test))
# intercept = LR_model.intercept_
# print(intercept)
# coeffs = LR_model.coef_
# # print(coeffs)
# r2_score(X_train,X_train_pred)
# plt.show()
# diff = np.subtract(y_pred,y_test)
# plt.plot(diff)
# plt.title('Difference Between Predicted and Actual Data For S&P 500 Price')
# diffperc = diff/y_test
# plt.plot(diffperc)
# plt.title('Percentage of Difference on S&P 500 Price Regression')

plt.show()
# # print(len(y_pred))
# plt.plot(y_test,color='green', alpha=0.6, label='Actual Data')
# plt.title('Actual and Predicted Data for S&P 500 Prediction')
# plt.show()
# Lasso Regression #
# LassoReg = linear_model.Lasso()
# LassoReg.fit(X_train, y_train)
# y_pred = LassoReg.predict(X_test)
# y_test = y_test.values.reshape(-1, 1)
# intercept = LassoReg.intercept_
# print('Intercept value of Lasso Regression: ',intercept)
# coeffs = LassoReg.coef_
# print('Coeficient values of Lasso Regression: ')
# print(coeffs)
# print('Lasso Regression Score', LassoReg.score(X_test, y_test))
# print('Mean Absolute Error : ', mean_absolute_error(y_test, y_pred))
# print('R2 Score : ', r2_score(y_test, y_pred))
# print('Mean Squared Error : ', mean_squared_error(y_test, y_pred))
# CV_score = cross_val_score(LassoReg,X_train,y_train, scoring ='r2', cv=10)
# print('Cross Valıdation Score :' )
# plt.plot(y_pred,label ='Predicted Data',color='red',alpha=0.7,linestyle='dashed')
# plt.plot(y_test,label = 'Actual Data',color='green',alpha =0.7)
# plt.title('Lasso Regression')
# plt.legend()
# plt.show()
# for k in range(len(CV_score)):
#     print(CV_score[k])
# print('Average Cross Validation Score : ',np.mean(CV_score))
# Ridge Regression #
# RidgeReg = Ridge()
# RidgeReg.fit(X_train,y_train)
# y_pred = RidgeReg.predict(X_test)
# print('Mean Absolute Error : ',mean_absolute_error(y_test, y_pred))
# print('R2 Score : ',r2_score(y_test,y_pred))
# print('Mean Squared Error : ', mean_squared_error(y_test,y_pred))
# print('Ridge Regression Score', RidgeReg.score(X_test,y_test))
# y_test = y_test.values.reshape(-1,1)
# intercept = RidgeReg.intercept_
# print('Intercept value of Ridge Regression: ',intercept)
# coeffs = RidgeReg.coef_
# print('Coeficient values of Ridge Regression: ')
# print(coeffs)
# plt.plot(y_pred,label ='Predicted Data',color='red',alpha=0.7,linestyle='dashed')
# plt.plot(y_test,label = 'Actual Data',color='green',alpha =0.7)
# plt.title('Ridge Regression')
# plt.legend()
# plt.show()

###### Performance Metrics #######
# Multivariable Regression#
# print(mean_absolute_error(X_train, X_train_pred))
# print(r2_score(X_train,X_train_pred))
# print(mean_squared_error(X_train,X_train_pred))
# Linear Regression#
# print(mean_absolute_error(X_test, X_test_pred))
# print(r2_score(X_test,X_test_pred))
# print(mean_squared_error(X_test,X_test_pred))
# Cross Validation#
# CV_score = cross_val_score(RidgeReg,y_train,X_train, scoring ='r2', cv=10)
# print('Cross Valıdation Score :' )
# for k in range(len(CV_score)):
#     print(CV_score[k])
# print('Average Cross Validation Score : ',np.mean(CV_score))
