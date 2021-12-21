# Regression method comparison for steels alloys


*Comparing multiple linear, decision tree and SVM regression methods for predicting mechanical properties of steel alloys.*


### 1 Introduction

Alloys are metallic materials, which contain two or more chemical elements. Steel alloys are iron and carbon-based alloys, which may contain other elements to improve the qualities of the material. Such useful alloying elements include manganese, chromium, nickel, aluminum, and titanium. (Callister, 2007)

The chemical composition of an alloy has a huge effect on its mechanical properties. For example, the mechanical properties of a steel are sensitive to its carbon content, measured in weight percentages (wt%). High-carbon steels (between 0.6 and 1.4 wt% carbon) are stronger and harder, but less ductile materials. On the other hand, low-carbon steels (less than 0.25 wt% carbon) are soft yet tough and ductile in comparison. (Callister, 2007)

When designing new steel alloys, material designers must select the alloying elements and their amounts carefully, as the chemical composition has such a great effect on mechanical properties. However, material science is an experimental science, and thus, it is hard to predict the exact behaviour of a material – even with exact chemical compositions in mind. Materials are designed mostly by trial-and-error, and while some predictions can be made with the extensive amount of experimental data and knowledge collected throughout human history, discovering an optimal material requires vast amounts of materials created and compared in lab settings.

Therefore, machine learning (ML) methods can be used to assist in this trial-and-error process by predicting suitable chemical compositions for certain preferred mechanical properties. This helps reduce the time spent testing different chemical compositions, as material designers would have better starting points for comparing and optimizing compositions. This approach of using machine learning methods and data science in materials science is called material informatics. (Ramprasad et al., 2017)

In this project, we use the chemical composition of steel alloys to study their effect on their mechanical properties. Specifically, we are interested in the iron, carbon, chromium, aluminum and titanium contents of an alloy to learn or predict its yield strength. Yield strength refers to the maximum amount of pulling tension the material can withstand before deforming permanently or yielding, measured in megapascals (MPa) (Callister, 2007).

This report consists of four sections (excluding the introduction): problem formulation, method, results, and conclusion. First, we formulate the mentioned study topic as a machine learning problem and define the data points and their properties. Second, we introduce selected ML methods, including their underlying models and loss functions, and discuss how we validate our training data. Third, we compare our selected ML methods through their training and validation errors. Then we assess the quality of our derived model by comparing validation and test errors. Last, we summarize our findings and contextualize the results. We also explore avenues for future exploration.


### 2 Problem formulation

We model the effect of the chemical composition of a steel alloy to its yield strength as a ML problem. In our models, an individual steel alloy represents a data point. The chemical composition works as our feature space: the weight percentage of each studied chemical element is a feature of the data point. In this study, we have five elements and thus five features. The yield strength in megapascals is assessed as the single label for each data point. **Table 1** showcases the features (in wt-%) and label (yield strength in MPA) as the variables for each data point.


### 3 Method

This project uses a dataset consisting of 312 steel alloys or data points provided through Figshare (Hacking Materials, 2018). This dataset includes the chemical compositions of an alloy through the weight percentages of carbon, manganese, silicon, chromium, nickel, molybdenum, vanadium, nitrogen, niobium, cobalt, wolfram, aluminum and titanium; and the mechanical properties for each alloy: yield strength, tensile strength, and elongation. It is a cleaned and deduplicated version of a dataset originally provided by Gareth Conduit from Cambridge University and Intellegens of +800 steel alloys (Conduit, 2017; Hacking Materials, 2018).

From this dataset, we exclude unnecessary columns, leaving only the feature and label columns we are interested in, as shown in Table 1. Table 1 also shows the key information for our dataset, including the features (in wt-%) and label (yield strength in MPa), and then the means, standard deviations, minimums and maximums for each column.

| | Fe (wt-%) | C (wt-%) | Cr (wt-%) | Al (wt-%) | Ti (wt-%) | Yield strength (MPa) |
| :---  | ---: | ---: | ---: | ---: | ---: | ---: |
| **mean** | 72.98 | 0.10 | 8.04| 0.24 | 0.31 | 1 421.00 |
| **std** | 5.15 | 0.11 | 5.43 | 0.34 | 0.56 | 301.89 |
| **min** | 62.00 | 0.00 | 0.01 | 0.01 | 0.00 | 1 005.90 |
| **max** | 86.00 | 0.43 | 17.50 | 1.80 | 2.50 | 2 510.30 |

> **Table 1.** Key values for each column of the dataset. Leftenmost columns mark the features (wt-% of chemical elements) and the last column marks the label (yield strength) of the dataset, respectively. Each value has been rounded to the nearest second decimal for displaying purposes.

The 312-row dataset was split into three sets using single splitting to keep the comparison of methods simple. The data was split into training and test sets with a 80-20 split. Then the training set was further split into a training and validation set with a 70-30 split. This resulted in training, validation and test sets with 174, 75, and 63 data points, respectively. We normalize all datasets with a min-max scaler, which scales each value between 0 and 1 in comparison to other values (Pedregosa et al., 2011). The training and validation sets are used to train and validate our models. We then select one model based on training and validation errors we get with the training and validation sets. The test set is used to assess the quality of our selected model.

As we have five numerical features and one numerical label, we use multiple regression methods to create our hypotheses (Jung, 2021). Hypotheses with five weighted variables are difficult to visualize, and thus, we must test and compare a set of ML methods with different hypothesis maps. The selected methods are multiple linear regression, decision tree regression, and support-vector machine regression (SVR) with polynomial and radial basis function (RBF) kernels. As their hypothesis maps, multiple linear regression uses linear hypothesis maps in the form of linear functions (Jung, 2021); decision tree regression uses all hypotheses which can be represented by a collection of decision trees, i.e., flow-charts (Jung, 2021); SVR with a polynomial kernel creates a hyperplane, where it uses nonlinear maps with a polynomial form (Jung, 2021); and SVR with a RBF kernel creates a hyperplane hypothesis map in the form of a radial basis function (Pietersma, 2010).

We do not implement feature selection, as it is outside the scope of this project. Thus, the five features are present in each hypothesis. This simplifies the process of minimizing the training error for multiple linear regression and SVR with the RBF kernel (or SVR-RBF), as these do not use hyperparameters for changing any dimensions for the models. Therefore, the training error can be minimized only with the decision tree regression and SVR with the polynomial kernel (SVR-P) models by the maximum depth of the tree d and maximum polynomial degree r, respectively. We use three decision tree regression models with maximum depth d = 2, 3, 4 and three SVR-P models with maximum polynomial degrees r = 2, 3, 4. Thus, we end up with 8 different candidate models as our hypotheses.

For minimizing the training error with the decision tree regression and SVR-P methods, we use squared error loss or mean squared error (MSE). MSE is a useful loss function as the values of the data points are numerical (Jung, 2021), and we can easily calculate it for each model with the Scikit Learn library (Pedregosa et al., 2011). Therefore, we use MSE also for calculating validation errors for each of the eight models. We then select the most fitting model based on which model has the smallest validation error. Table 2 introduces the training and validations errors for each model. 

| | multiple linear regression | decision tree regression, d=2 | decision tree regression, d=3 | decision tree regression, d=4 | SVR-P, r=2 | SVR-P, r=3 | SVR-P, r=4 | SVR-RBF |
| :---  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **training** | 18.615 | 15.348 | 12.535 | 9.512 | 14.537 | 11.651 | 11.505 | 9.001 |
| **validation** | 20.680 | 11.849 | 12.123 | 15.574 | 19.023 | 20.744 | 32.785 | **11.726** |

> **Table 2**. Training and validation errors for eight studied models: multiple linear regression, decision tree regression with maximum tree depths of 2, 3 and 4, support-vector machine regression with a polynomial kernel (SVR-P) with maximum polynomial degrees of 2, 3 and 4, and support-vector machine regression with a radial basis function kernel (SVR-RBF). The errors have been multiplied by 10³ for displaying and clearer comparison purposes.


### 4 Results

Table 2 shows that the SVR-RBF model produced the smallest validation error. In addition, decision tree regression produces comparable validation errors, with the smallest error being through a maximum tree depth of 2 – it also has a smaller validation error than training error. Based on the differences between training and validation errors, the decision tree regression with d=3, multiple linear regression and SVR-RBF models seem not to be overfitting too much.

The multiple linear regression and SVR-P models have much larger validation errors, which indicates that these models are of lower quality compared to the decision tree regression and SVR-RBF models. On the other hand, we can see that as the hyperparameters d and r grow, the training error gets smaller and the validation error grows for the decision tree regression and SVR-P models, respectively. Thus, we can argue that the hyperparameterized models have a tendency to overfit, as d and r grow (Jung, 2021). 

Our best fitting model and thus hypothesis can be evaluated by comparing the validation and test errors. We calculate the MSE for our hypothesis using the untouched test set. **This results in a test error of 10.834 ⋅ 10-³**, which is quite close to the validation error (11.726 ⋅ 10-³). The test error is 7.6% smaller than the validation error.


### 5 Conclusion

We studied eight quite different models to predict the yield strength of a steel alloy based on its chemical composition. The eight models were based on multiple linear regression, decision tree regression with different maximum tree depths, and support-vector machine regression with polynomial and radial basis function kernels. We minimized training error with the models that use hyperparameters through testing different values for those. We chose the model with the smallest validation error, which was the SVR-RBF model with a validation error of 11.726 ⋅ 10-³. The comparable training error was 9.001 ⋅ 10-³, which indicates that the model overfits slightly. However, we do not have a way of judging whether the error levels are suitable in this problem space without benchmarks or further experimenting and testing predicted materials in lab settings.

For usefulness and quality, our ML method could be improved with more features and more data. In this project, we used only a subset of chemical elements. In addition, we omitted other factors, such as heat treatments, which have a significant impact on the mechanical properties of a material. Our method could be improved by including all collected features, and then investigate which features are important through feature selection. Other research, such as a study by Shiraiwa, Miyazawa and Enoki (2018) for predicting fatigue strength, has done just so. 

Shiraiwa, Miyazawa and Enoki (2018) also saw some success using neural networks in their study. However, neural networks become useful with much more data than our 312-row dataset provides (Ramprasad et al., 2017). Thus, to explore this problem space with neural networks, we would require much more data points to be collected, yet it could be worth the gigantic task of data collection as neural networks could offer more robust models for steel design and material informatics in general.


### References

Callister Jr, W. D. (2007). *Materials science and engineering: an introduction*. 7th edition. John Wiley & Sons. ISBN: 978-0-471-73696-7.

Conduit, G. (2017). *Mechanical properties of some steels*. 3rd version [data file]. Citrination [distributor]. Available at: https://citrination.com/datasets/153092.

Hacking Materials. (2018). *Steel Strength Data*. [data file]. Figshare [distributor]. DOI: doi.org/10.6084/m9.figshare.7250453.v1.

Jung, A. (2021). *Machine Learning: The Basics*. Unpublished manuscript. Available at: http://mlbook.cs.aalto.fi/.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M.,  Perrot, M., Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12. Pp. 2825–2830. DOI: arXiv:1201.0490.

Pietersma, A-D. (2010). *Feature space learning in Support Vector Machines through Dual Objective optimization*. Master’s thesis. Available at: https://www.researchgate.net/publication/46102644_Feature_space_learning_in_Support_Vector_Machines_through_Dual_Objective_optimization.

Ramprasad, R., Batra, R., Pilania, G., Mannodi-Kanakkithodi, A., Kim, C. (2017). Machine learning in materials informatics: recent applications and prospects. *npj Comput Mater*, 3:54. Pp. 1–13. DOI: 10.1038/s41524-017-0056-5.

Shiraiwa, T., Miyazawa, Y., Enoki, M. (2018). Prediction of fatigue strength in steels by linear regression and neural network. *Materials Transactions*, 60:2. Pp. 189–198. DOI: 10.2320/matertrans.ME201714.
