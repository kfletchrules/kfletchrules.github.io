# Classifying Healthcare Procurement Data using Machine Learning  
<img align="right" width="300" height="300" src="/assets/IMG/ML class final proj logo.png">

### Introduction
  The primary duty of healthcare providers is first and foremost “to do no harm” according
to the Hippocratic oath, but if we continue to disregard environmental consequences in the
delivery of patient care: are we really living up to this standard? Scope 3 emissions are difficult to quantify, but they “account for about 80 percent of health sector emissions and [are] generated largely from supply chain expenses”(Agbafe et al., 2024). According to the Greenhouse Gas Protocol, scope 1 emissions are classified as direct emissions (i.e., direct emissions from owned or controlled sources) and scope 2 (i.e., indirect emissions from the generation of purchased energy consumed by the reporting company) and 3 as indirect emissions. Scope 3 emissions are more specifically defined as “occur[ring] from sources owned or controlled by other entities in the value chain (e.g., materials suppliers)” (Corporate Value Chain (Scope 3) Accounting and Reporting Standard, n.d.). The current industry standard for measuring scope 3 emissions is through US Environmentally-Extended Input-Output (USEEIO) modeling, which “is an environmental-economic model of US goods and services that can be used for life cycle assessment, footprinting, national prioritization, and related applications” (Ingwersen et al., 2022). USEEIO takes economic information and converts it to Greenhouse Gas (GHG) emissions based on US commodity categories, essentially creating a model that uses dollars as inputs and generates GHG emissions as outputs. This approach is inherently flawed due to the weak correlation between product price and product emissions intensity, but with thousands of products in a businesses supply chain, individual Life Cycle Assessments (LCA) to calculate emissions for each product are extremely cumbersome and unrealistic. 
  The aim of this project is to investigate the use of machine learning models to improve the identification of high emissions intensity products based on product text descriptions and quantities from procurement data. There have been few studies utilizing machine learning for measuring business emissions, even fewer studies targeting scope 3 measurement, and none that have involved the US healthcare industry. Machine learning classification offers a new approach not only for healthcare, but for all businesses that wish to focus their efforts on scope 3 reductions.
  
### Data
The dataset used in this study was provided by University of San Francisco (UCSF) Medical Center, a 600 bed public hospital located in San Francisco, California. They are a level 1 trauma center and provide both tertiary and quaternary care to their patients (UCSF Medical Center at Parnassus, Mount Zion, Mission Bay | Department of Medicine, n.d.). The dataset consists of procurement data from one fiscal year, not inclusive of a year affected by the Covid-19 pandemic. The variables included that are of interest for this study include: item description, yearly total quantity, yearly total spent, and commodity type. The sample size is approximately 25,000 unique products.

<img align="center" width="700" height="300" src="/assets/log10_toqty.png">

Figure 1. log10 scale distribution of product quantities across the 25,000 products.

### Modeling
  The objective of this study was to attempt to predict whether healthcare items were low, medium, or high emissions products based on their text description and annual quantities. The main machine learning approach utilized in this study was K-means (and Minibatch K-Means). Decision Tree and Random Forest Classification were attempted, but due to the diversity and volume of products to be analyzed it was determined to be unfeasible for the timeline of this project.  The workflow of this study began with pre-processing text data, followed by k-means and minibatch k-means clustering, cosine similarity calculation, and concluded with cluster visualization.
  
*Pre-processing*

  After loading the dataset into Google Collab, item descriptions (categorical), yearly total quantity (continuous), department (categorical), and yearly total cost (continuous) were extracted as relevant columns to later be combined as a feature set. Wordtokenizer was utilized to vectorize the item descriptions into tokens, which were then cleaned to convert the text to lowercase, and remove stopwords, spaces, and the catalog number at the end of the description. Empty descriptions were filtered out. The cleaned tokens were used to train the Word2Vec model, which then generated 100 embeddings to represent the text descriptions. A label encoder was used for categorical features and continuous features were normalized with StandardScaler. The categorical and continuous features were then combined into a single feature set array. The output shapes of the combined features were then generated to be, (25959, 102).
  
*K-Means & Minibatch K-Means*

  Both K-Means and Minibatch K-Means were applied to this combined feature set, to group into clusters. The number of clusters that was determined to have the highest silhouette score was 5. The K-Means clustering algorithm iterates over the entire dataset to minimize intra-cluster variance, while MiniBatchKMeans uses smaller random batches (2048 in this case) for faster computation on large datasets. 
  
*Cosine Similarity* 

  Cosine similarity was used to measure the similarity of item descriptions from the Word2Vec embeddings. Cosine similarity min, max, mean, and standard deviation help verify the quality and consistency of the embeddings and their pairwise relationships.
  
*Visualization*

  A 2D t-SNE visualization was generated to show a scatter plot of clusters and their assigned labels. A box plot was also generated to show the distribution of clusters compared to the annual quantity variable. 
  
### Results

### Discussion

### Links
https://colab.research.google.com/drive/1HJcNoHLEc2PlRUwgHd1gAq94qvvUmWHR?usp=sharing

### References
Agbafe, V. C., Singh, H., & Cerceo, E. (2024). Comprehensive SEC Disclosure Rules Can Reduce Health Care Emissions. Health Affairs Forefront. https://doi.org/10.1377/forefront.20240814.180078

Corporate Value Chain (Scope 3) Accounting and Reporting Standard. (n.d.). https://ghgprotocol.org/sites/default/files/standards/Corporate-Value-Chain-Accounting-Reporing-Standard_041613_2.pdf

Ingwersen, W. W., Li, M., Young, B., Vendries, J., & Birney, C. (2022). USEEIO v2.0, The US Environmentally-Extended Input-Output Model v2.0. Scientific Data, 9(1), 194. https://doi.org/10.1038/s41597-022-01293-7

UCSF Medical Center at Parnassus, Mount Zion, Mission Bay | Department of Medicine. (n.d.). Retrieved September 4, 2024, from https://medicine.ucsf.edu/about/locations/ucsf-medical-center-parnassus-mount-zion-mission-bay
