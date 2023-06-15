# Hybrid-system--Monte-Carlo-Simulation-with-Machine-Learning
Hybrid system- Monte Carlo Simulation with Machine Learning

Supply chain management (SCM) deals with the planning and control of materials and information flow in the distribution channel. Efficient SCM calls for data-driven decisions to optimize operations and achieve cost-effectiveness (Gunasekaran et al., 2017). Our study aims to predict late delivery risks using machine learning (ML) techniques.

Ipynb with EDA:
https://drive.google.com/file/d/1m3ke4zB9KB6DYMYPsEyDYlbi2MhqXy6P/view?usp=drive_link

The dataset includes a varity of features pertaining to the optimization of a supply chain. 
We aim to create a hybrid model composed of machine learning algorithms and monte carlo simulations to build upon each other. 
First three machine learning algorithms: Random Forest Classifier, Gradient Boost classifier, and logistic regression are fitted on the dataset and their prediction accuracies are compared. 
The most suitable model is selected
The model.fit from this step is used as the 'transfer function' for simulation modelling
The simulation consists checking the impact of different scenarios keeping the variables whos impact we want to measure constant. 
The fixed feature is judged ont eh impact it has on the late delivery risk.
The features that are judged are: 
- Market
- Customer sales
- Product price
Through running simulations we can check the leaks in the supply chain and target those weak points
This can further be used for supply chain optimization
