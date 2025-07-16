
# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.preprocessing import MinMaxScaler
import joblib
# Load the dataset (update the filename accordingly)
df = pd.read_csv("irrigation_machine.csv")
# first 5 rows to be printed, df.tail()
df.head()
Unnamed: 0	sensor_0	sensor_1	sensor_2	sensor_3	sensor_4	sensor_5	sensor_6	sensor_7	sensor_8	...	sensor_13	sensor_14	sensor_15	sensor_16	sensor_17	sensor_18	sensor_19	parcel_0	parcel_1	parcel_2
0	0	1.0	2.0	1.0	7.0	0.0	1.0	1.0	4.0	0.0	...	8.0	1.0	0.0	2.0	1.0	9.0	2.0	0	1	0
1	1	5.0	1.0	3.0	5.0	2.0	2.0	1.0	2.0	3.0	...	4.0	5.0	5.0	2.0	2.0	2.0	7.0	0	0	0
2	2	3.0	1.0	4.0	3.0	4.0	0.0	1.0	6.0	0.0	...	3.0	3.0	1.0	0.0	3.0	1.0	0.0	1	1	0
3	3	2.0	2.0	4.0	3.0	5.0	0.0	3.0	2.0	2.0	...	4.0	1.0	1.0	4.0	1.0	3.0	2.0	0	0	0
4	4	4.0	3.0	3.0	2.0	5.0	1.0	3.0	1.0	1.0	...	1.0	3.0	2.0	2.0	1.0	1.0	0.0	1	1	0
5 rows × 24 columns

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 24 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Unnamed: 0  2000 non-null   int64  
 1   sensor_0    2000 non-null   float64
 2   sensor_1    2000 non-null   float64
 3   sensor_2    2000 non-null   float64
 4   sensor_3    2000 non-null   float64
 5   sensor_4    2000 non-null   float64
 6   sensor_5    2000 non-null   float64
 7   sensor_6    2000 non-null   float64
 8   sensor_7    2000 non-null   float64
 9   sensor_8    2000 non-null   float64
 10  sensor_9    2000 non-null   float64
 11  sensor_10   2000 non-null   float64
 12  sensor_11   2000 non-null   float64
 13  sensor_12   2000 non-null   float64
 14  sensor_13   2000 non-null   float64
 15  sensor_14   2000 non-null   float64
 16  sensor_15   2000 non-null   float64
 17  sensor_16   2000 non-null   float64
 18  sensor_17   2000 non-null   float64
 19  sensor_18   2000 non-null   float64
 20  sensor_19   2000 non-null   float64
 21  parcel_0    2000 non-null   int64  
 22  parcel_1    2000 non-null   int64  
 23  parcel_2    2000 non-null   int64  
dtypes: float64(20), int64(4)
memory usage: 375.1 KB
df.columns
Index(['Unnamed: 0', 'sensor_0', 'sensor_1', 'sensor_2', 'sensor_3',
       'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9',
       'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
       'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19',
       'parcel_0', 'parcel_1', 'parcel_2'],
      dtype='object')
df = df.drop('Unnamed: 0', axis=1)
df.head()
sensor_0	sensor_1	sensor_2	sensor_3	sensor_4	sensor_5	sensor_6	sensor_7	sensor_8	sensor_9	...	sensor_13	sensor_14	sensor_15	sensor_16	sensor_17	sensor_18	sensor_19	parcel_0	parcel_1	parcel_2
0	1.0	2.0	1.0	7.0	0.0	1.0	1.0	4.0	0.0	3.0	...	8.0	1.0	0.0	2.0	1.0	9.0	2.0	0	1	0
1	5.0	1.0	3.0	5.0	2.0	2.0	1.0	2.0	3.0	1.0	...	4.0	5.0	5.0	2.0	2.0	2.0	7.0	0	0	0
2	3.0	1.0	4.0	3.0	4.0	0.0	1.0	6.0	0.0	2.0	...	3.0	3.0	1.0	0.0	3.0	1.0	0.0	1	1	0
3	2.0	2.0	4.0	3.0	5.0	0.0	3.0	2.0	2.0	5.0	...	4.0	1.0	1.0	4.0	1.0	3.0	2.0	0	0	0
4	4.0	3.0	3.0	2.0	5.0	1.0	3.0	1.0	1.0	2.0	...	1.0	3.0	2.0	2.0	1.0	1.0	0.0	1	1	0
5 rows × 23 columns

df.describe() # Statistics of the dataset
sensor_0	sensor_1	sensor_2	sensor_3	sensor_4	sensor_5	sensor_6	sensor_7	sensor_8	sensor_9	...	sensor_13	sensor_14	sensor_15	sensor_16	sensor_17	sensor_18	sensor_19	parcel_0	parcel_1	parcel_2
count	2000.000000	2000.000000	2000.000000	2000.000000	2000.000000	2000.000000	2000.000000	2000.000000	2000.000000	2000.000000	...	2000.000000	2000.000000	2000.000000	2000.000000	2000.000000	2000.00000	2000.000000	2000.00000	2000.000000	2000.000000
mean	1.437000	1.659000	2.654500	2.674500	2.887500	1.411000	3.315500	4.201500	1.214000	1.901000	...	2.731500	3.416000	1.206500	2.325000	1.729500	2.27450	1.813500	0.63550	0.730500	0.212000
std	1.321327	1.338512	1.699286	1.855875	1.816451	1.339394	2.206444	2.280241	1.386782	1.518668	...	1.774537	1.960578	1.258034	1.715181	1.561265	1.67169	1.469285	0.48141	0.443811	0.408827
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	...	0.000000	0.000000	0.000000	0.000000	0.000000	0.00000	0.000000	0.00000	0.000000	0.000000
25%	0.000000	1.000000	1.000000	1.000000	2.000000	0.000000	2.000000	3.000000	0.000000	1.000000	...	1.000000	2.000000	0.000000	1.000000	0.000000	1.00000	1.000000	0.00000	0.000000	0.000000
50%	1.000000	1.000000	2.000000	2.000000	3.000000	1.000000	3.000000	4.000000	1.000000	2.000000	...	2.000000	3.000000	1.000000	2.000000	1.000000	2.00000	2.000000	1.00000	1.000000	0.000000
75%	2.000000	2.000000	4.000000	4.000000	4.000000	2.000000	5.000000	6.000000	2.000000	3.000000	...	4.000000	5.000000	2.000000	3.000000	3.000000	3.00000	3.000000	1.00000	1.000000	0.000000
max	8.000000	9.000000	10.000000	11.000000	12.000000	7.000000	13.000000	12.000000	8.000000	9.000000	...	11.000000	11.000000	6.000000	10.000000	11.000000	10.00000	7.000000	1.00000	1.000000	1.000000
8 rows × 23 columns

# -------------------------------
# STEP 2: DEFINE FEATURES AND LABELS
# -------------------------------

X = df.iloc[:, 0:20]   # This gives you columns 0 to 19 (sensor_0 to sensor_19)


y = df.iloc[:, 20:]
X.sample(10)
sensor_0	sensor_1	sensor_2	sensor_3	sensor_4	sensor_5	sensor_6	sensor_7	sensor_8	sensor_9	sensor_10	sensor_11	sensor_12	sensor_13	sensor_14	sensor_15	sensor_16	sensor_17	sensor_18	sensor_19
1368	1.0	3.0	3.0	8.0	1.0	0.0	0.0	3.0	1.0	4.0	2.0	1.0	6.0	2.0	6.0	1.0	1.0	0.0	2.0	0.0
176	0.0	2.0	3.0	1.0	2.0	4.0	2.0	6.0	1.0	1.0	2.0	1.0	1.0	1.0	3.0	2.0	4.0	2.0	2.0	2.0
461	1.0	1.0	2.0	3.0	2.0	0.0	6.0	4.0	0.0	2.0	1.0	4.0	5.0	2.0	2.0	3.0	4.0	3.0	0.0	2.0
1948	2.0	4.0	1.0	0.0	8.0	2.0	4.0	4.0	0.0	0.0	2.0	3.0	5.0	1.0	2.0	4.0	2.0	6.0	1.0	0.0
720	3.0	0.0	2.0	3.0	5.0	0.0	7.0	3.0	2.0	5.0	1.0	3.0	1.0	2.0	1.0	0.0	1.0	2.0	1.0	2.0
870	4.0	0.0	4.0	1.0	1.0	1.0	3.0	2.0	2.0	4.0	2.0	2.0	3.0	0.0	4.0	0.0	2.0	0.0	1.0	3.0
463	3.0	3.0	6.0	1.0	2.0	0.0	4.0	0.0	1.0	2.0	4.0	3.0	3.0	1.0	3.0	1.0	3.0	3.0	1.0	0.0
1106	1.0	0.0	2.0	3.0	1.0	3.0	3.0	8.0	1.0	2.0	1.0	6.0	5.0	4.0	8.0	0.0	5.0	1.0	4.0	1.0
1168	0.0	2.0	1.0	4.0	4.0	1.0	3.0	3.0	2.0	1.0	4.0	4.0	10.0	0.0	8.0	3.0	1.0	0.0	3.0	0.0
1066	1.0	3.0	2.0	3.0	2.0	0.0	1.0	5.0	1.0	3.0	2.0	6.0	3.0	5.0	5.0	0.0	2.0	0.0	5.0	2.0
y.sample(10)
parcel_0	parcel_1	parcel_2
1604	1	1	1
1924	1	0	0
1853	0	0	0
1494	1	1	1
1289	1	0	0
1441	0	1	0
1464	0	1	0
1415	0	0	0
1173	1	1	0
1341	0	1	0
X.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 20 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   sensor_0   2000 non-null   float64
 1   sensor_1   2000 non-null   float64
 2   sensor_2   2000 non-null   float64
 3   sensor_3   2000 non-null   float64
 4   sensor_4   2000 non-null   float64
 5   sensor_5   2000 non-null   float64
 6   sensor_6   2000 non-null   float64
 7   sensor_7   2000 non-null   float64
 8   sensor_8   2000 non-null   float64
 9   sensor_9   2000 non-null   float64
 10  sensor_10  2000 non-null   float64
 11  sensor_11  2000 non-null   float64
 12  sensor_12  2000 non-null   float64
 13  sensor_13  2000 non-null   float64
 14  sensor_14  2000 non-null   float64
 15  sensor_15  2000 non-null   float64
 16  sensor_16  2000 non-null   float64
 17  sensor_17  2000 non-null   float64
 18  sensor_18  2000 non-null   float64
 19  sensor_19  2000 non-null   float64
dtypes: float64(20)
memory usage: 312.6 KB
y.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   parcel_0  2000 non-null   int64
 1   parcel_1  2000 non-null   int64
 2   parcel_2  2000 non-null   int64
dtypes: int64(3)
memory usage: 47.0 KB
X
sensor_0	sensor_1	sensor_2	sensor_3	sensor_4	sensor_5	sensor_6	sensor_7	sensor_8	sensor_9	sensor_10	sensor_11	sensor_12	sensor_13	sensor_14	sensor_15	sensor_16	sensor_17	sensor_18	sensor_19
0	1.0	2.0	1.0	7.0	0.0	1.0	1.0	4.0	0.0	3.0	1.0	3.0	6.0	8.0	1.0	0.0	2.0	1.0	9.0	2.0
1	5.0	1.0	3.0	5.0	2.0	2.0	1.0	2.0	3.0	1.0	3.0	2.0	2.0	4.0	5.0	5.0	2.0	2.0	2.0	7.0
2	3.0	1.0	4.0	3.0	4.0	0.0	1.0	6.0	0.0	2.0	3.0	2.0	4.0	3.0	3.0	1.0	0.0	3.0	1.0	0.0
3	2.0	2.0	4.0	3.0	5.0	0.0	3.0	2.0	2.0	5.0	3.0	1.0	2.0	4.0	1.0	1.0	4.0	1.0	3.0	2.0
4	4.0	3.0	3.0	2.0	5.0	1.0	3.0	1.0	1.0	2.0	4.0	5.0	3.0	1.0	3.0	2.0	2.0	1.0	1.0	0.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1995	4.0	1.0	2.0	2.0	1.0	1.0	1.0	2.0	1.0	2.0	4.0	3.0	3.0	1.0	2.0	3.0	2.0	1.0	1.0	0.0
1996	1.0	3.0	3.0	3.0	2.0	2.0	3.0	3.0	1.0	5.0	2.0	2.0	4.0	3.0	3.0	0.0	1.0	0.0	6.0	2.0
1997	1.0	3.0	3.0	1.0	1.0	4.0	8.0	1.0	0.0	0.0	3.0	2.0	4.0	2.0	3.0	4.0	4.0	4.0	1.0	0.0
1998	2.0	1.0	0.0	2.0	2.0	0.0	1.0	3.0	0.0	0.0	0.0	5.0	2.0	2.0	4.0	0.0	2.0	0.0	3.0	0.0
1999	0.0	1.0	4.0	1.0	2.0	2.0	6.0	8.0	5.0	1.0	2.0	4.0	3.0	2.0	1.0	1.0	0.0	5.0	2.0	1.0
2000 rows × 20 columns

X.shape, y.shape
((2000, 20), (2000, 3))
