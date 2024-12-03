# Classifying Healthcare Procurement Data using Machine Learning  
<img align="left" width="300" height="300" src="/assets/IMG/ML class final proj logo.png">

### Introduction
  The primary duty of healthcare providers is first and foremost “to do no harm” according
to the Hippocratic oath, but if we continue to disregard environmental consequences in the
delivery of patient care: are we really living up to this standard? Scope 3 emissions are difficult to quantify, but they “account for about 80 percent of health sector emissions and [are] generated largely from supply chain expenses”(Agbafe et al., 2024). According to the Greenhouse Gas Protocol, scope 1 emissions are classified as direct emissions (i.e., direct emissions from owned or controlled sources) and scope 2 (i.e., indirect emissions from the generation of purchased energy consumed by the reporting company) and 3 as indirect emissions. Scope 3 emissions are more specifically defined as “occur[ring] from sources owned or controlled by other entities in the value chain (e.g., materials suppliers)” (Corporate Value Chain (Scope 3) Accounting and Reporting Standard, n.d.). The current industry standard for measuring scope 3 emissions is through US Environmentally-Extended Input-Output (USEEIO) modeling, which “is an environmental-economic model of US goods and services that can be used for life cycle assessment, footprinting, national prioritization, and related applications” (Ingwersen et al., 2022). USEEIO takes economic information and converts it to Greenhouse Gas (GHG) emissions based on US commodity categories, essentially creating a model that uses dollars as inputs and generates GHG emissions as outputs. This approach is inherently flawed due to the weak correlation between product price and product emissions intensity, but with thousands of products in a businesses supply chain, individual Life Cycle Assessments (LCA) to calculate emissions for each product are extremely cumbersome and unrealistic. 
  The aim of this project is to propose the use of machine learning models to improve the identification of high emissions intensity products based on product text descriptions and quantities from procurement data. There have been few studies utilizing machine learning for measuring business emissions, even fewer studies targeting scope 3 measurement, and none that have involved the US healthcare industry. Machine learning classification offers a new approach not only for healthcare, but for all businesses that wish to focus their efforts on scope 3 reductions.
  
### Data
The dataset used in this study was provided by University of San Francisco (UCSF) Medical Center, a 600 bed public hospital located in San Francisco, California. They are a level 1 trauma center and provide both tertiary and quaternary care to their patients (UCSF Medical Center at Parnassus, Mount Zion, Mission Bay | Department of Medicine, n.d.). The dataset consists of procurement data from one fiscal year, not inclusive of a year affected by the Covid-19 pandemic. The variables included that are of interest for this study include: item description, yearly total quantity, yearly total spent, and commodity type.

<img align="center" width="700" height="300" src="/assets/log10_toqty.png">

Figure 1. log10 scale distribution of product quantities across the 25,000 products.

### Modeling

### Results

### Discussion

### References

### Links

### References
