# Cryptocurrencies
Use unsupervised machine learning models to reduce data dimensions and cluster data into meaningful groups. 

## Project Overview
A client, Accountability Accounting, wants to offer its customers a new cryptocurrency investment portfolio. However, there are vast selections of cryptocurrencies in the market. I was asked to perform a cluster analysis to build a classification system for this new investment. 

In this analysis, I used PCA to reduce the principal components and the K-Means clustering algorithm to group the cryptocurrencies.  

## Data Preprocessing 
The raw DataFrame has 1252 records with 6 columns. Each record represents a unique cryptocurrency. Among these cryptocurrencies, 1144 cryptocurrencies are being traded. Cryptocurrencies that were not actively traded are not good options for investment. So these cryptocurrencies are removed. Next, I checked the null values in the DataFrame and removed these records. The CoinName column is not useful in the unsupervised learning model. This column was dropped as well. This leaves a crypto_df with 532 rows and 4 columns. 

To perform unsupervised machine learning algorithm with this dataset, I have to transfer all the data in the DataFrame into numerical values. I used pd.get_dummies() to achieve this goal. Next, I scaled the data to bring all the features to the same level of magnitude.  

## Reducing Data Dimensions Using PCA
With PCA, I can reduce the number of dimensions by transforming a large set of variables into a smaller one that contains most of the information in the original large set of variables. In this step, I reduced the data dimensions to 3 Principal Components with PCA and saved the data in the pcs_df. 

## K-Means
With the K-Means algorithm, I created an elbow curve to decide the best value of K, which is the number of clusters for the dataset. Then, I feed the cryptocurrencies DataFrame with reduced data dimensions into the K-Means algorithm. Next, I created a new DataFrame, clustered_df, that concatenates crypto_df, pcs_df, and clusters prediction values. 

## Visualization
In this part, I created a scatter plot with Plotly Express and hvplot. I visualized the 4 distinct groups that correspond to the three principal components with X-axis as "TotalCoinsMined" and the Y-axis as "TotalCoinSupply". 
