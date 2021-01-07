# Personal Data Science Portfolio

This repository contains my data science projects, some for my post graduate course, some for online courses like Udacity Data Analyst or Alura short courses, others were made for my self learning using some Kaggle problems. Most of the notebooks were made in Python programming language using Jupyter Notebook, just one was made in R using the R Markdown Tool.

## Repositories by subject:

### Machine Learning Tools and Techniques
- [Basic Clustering Techniques](https://github.com/daniel-guarino/ds-portfolio/blob/main/ml-tools/clustering/ClusteringBasico_Alura.ipynb): K-means, BDSCAN and Mean Shift to group unclassified data using Plotly and Scikit Learn.

- [Cross Validation](https://github.com/daniel-guarino/ds-portfolio/blob/main/ml-tools/cross-validation/ML_Valida%C3%A7%C3%A3o_de_Modelos_Alura.ipynb): create a good validation process is one of the key processess to succeed in a Machine Learning project. In this notebook there are examples of use of techniques such as KFold, StratifiedKFold and GroupKFold.

- [Hyperparameter Model Optimization](https://github.com/daniel-guarino/ds-portfolio/blob/main/ml-tools/hyperparameter-optimization/ML_Parte1_Alura.ipynb): Understanding hiperparameters for Decision Tree Models.

- [Dimensionality Reduction](https://github.com/daniel-guarino/ds-portfolio/blob/main/ml-tools/dimensionality-reduction/ReducaoDimensionalidade_Alura.ipynb): Avoiding features with high correlation among themselves, selecting features with visualization and PCA.

### Statistics

- [Simpson's Paradox](https://github.com/daniel-guarino/ds-portfolio/blob/main/statistics/simpsons-paradox/UC%20Berkley%20-%20Admissions%20Case.ipynb): A simple example of how to use this important tool in a Exploratory Analysis, explaning how we can be fooled if don't look in a correct way to our data.

- [Investigate a Dataset](https://github.com/daniel-guarino/project02-investigate-a-dataset/blob/master/investigate-a-dataset-template-DanielGuarino.ipynb): Analyse economical data such as Inflation, GDP and GINI index from Gapminder World website for South American countries. In this notebook are present some data visualization and I also used the Exploratory Analysis to answer questions about the countries. In the end a brief reflection on what we really an answer about these countries using this dataset.

- [Analyse AB Test Results](https://github.com/daniel-guarino/project03-analyze-ab-test-results/blob/master/Analyze_ab_test_results_notebook.ipynb): A/B tests are a important tool for data analysts and data scientists. In this project, I used this techinque to understand the results of an A/B test run by an e-commerce website. The goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.

- [Time Series Decompose](https://github.com/daniel-guarino/ds-portfolio/blob/main/machine-learning/time-series/Time_Series_Alura.ipynb): Decompose time series and generate new features in order to analyse sales behaviour for three different datasets. For this it was used Pandas, Matplotlib and Statsmodels Python libraries.

- [Wrangling and Analyse Twitter Data](https://github.com/daniel-guarino/project04-wrangling-and-analyze-data/blob/master/wrangle_act.ipynb): Gather data from  different sources (Twitter API, csv file), assess and clean dirty data and get some data insights using data visualization.

### Machine Learning

- [Credit Analysis using GCP](https://github.com/daniel-guarino/ds-portfolio/blob/main/machine-learning/credit-analysis-GCP/C%C3%B3pia_de_Concess%C3%A3o_de_Cr%C3%A9dito_com_Machine_Learning.ipynb): A simple classification model using Google Cloud Computing with Docker Container and Postgree as database architeture (the GCP architeture repository can be found in this link: https://github.com/elthonf/plataformas-cognitivas-docker). We access a machine learning model served as an API and then can evaluate if the customer has a hight probability to default and a conclusion if we must loan some money to this customer or not.

- [Housing Prices](https://github.com/daniel-guarino/ds-portfolio/blob/main/machine-learning/housing-prices/Trabalho_EDA_Python.ipynb): To solve this problem (predict house prices in California) it was used data mining techniques such as EDA, Data Visualization and Data Preprocessing, then it was generated a Linear Regression. The Root Mean Square Error - RSME to evaluate the models. 

- [Movies review's sentiment classification](https://github.com/daniel-guarino/ds-portfolio/blob/main/nlp-classifier/Trabalho_NLP_Fiap10IA.ipynb): Problem solved with NLP techniques using NLTK and Spacy, pre processing techniques such as Count Vectorizer and TD-IDF. A first baseline model was generated using Decision Tree keeping the sopt words in order to obtain a baseline F1 Score, then the models were generated using Naïve Bayes, Decision Tree, Logistic Regression and SVM machine learning algorithms using Scikit Learn library. It was used Accuracy and F1 Score to evaluate the models. All the movie reviews are in Portuguese in this dataset.

- [Demand Forecasting](https://github.com/daniel-guarino/ds-portfolio/blob/main/machine-learning/time-series/Previsao_demanda_ML_MarioFilho.ipynb): A self leraning study during a youtube class by the famous brazilian data scientist Mario Filho. The notebook was generate at the same time I watched the class

- [Recommender System](https://github.com/daniel-guarino/ds-portfolio/blob/main/recommender-algorithm/Sistema_de_Recomenda%C3%A7%C3%A3o_de_Filmes_Alura.ipynb): Movies recommender system made without Scikit Learn, just using numpy.  

- [ENEM 2017 score prediction](https://github.com/daniel-guarino/Projeto_Formacao_Machine_Learning/blob/master/Forma%C3%A7%C3%A3o_Machine_Learning_Projeto_Alura.ipynb): ENEM is a pre college test exam for brazilian students (like SAT). In this project was necessary to create predictions of the total score based on the score of 5 subjects. Later it was also implemented a classification model in order to predict if the student is in the highest 25% scores.

### Computer Vision using deep learning

- [Face recognition, Age and sex prediction](https://github.com/daniel-guarino/VisaoComputacional-VideoAudit10IA/blob/master/projeto/object_people_audit_colab.ipynb):

This is a project using Convolutional Neural Network (CNN) transfer learning in order to find men and woman with a certain age in video images, analysing frame by frame. The training of this two neural network models (age and biological sex) were made using Keras library trained using **Google Colab Pro TPU resourses**. Two different techniques of face recognition were tested (Cascade OpenCV and HOG algorithm). To test everything it was used a "The Office" tv show 05 minutes video.

### R

- [Wine Quality](https://github.com/daniel-guarino/ds-portfolio/blob/main/wine-analysis/EDA-Vinhos_Dataset.Rmd): Here it was used a modified version of the wine dataset from UCI data of Portuguese red and white wine. In this work I generated regression models to predict wine's quality, logistic regression models to predict if the wine is good or not and also some unsupervised techniques to obtain more information about the data.