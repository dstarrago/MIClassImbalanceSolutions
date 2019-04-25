# MIClassImbalanceSolutions
Sampling-based class imbalance solutions for multiple-instance classification

In contrasts with regular classification problems, in which each example has a unique description, in multiple-instance classification (MIC) problems, each example has many descriptions. In the same way as regular classification problems, multiple-instances classification problems can suffer from the class imbalance problem. A data set suffer from the class imbalance problem when one or more of its classes are underrepresented, which means that the size of these classes is much smaller than that of the rest of the classes. Underrepresented classes are hard to learn by classification algorithms, and their instances are frequently misclassified in favor of the larger classes.

We explored here several solutions based on classifier ensembles to the class imbalance problem in multiple-instance classification. We introduced sampling steps in bagging and boosting algorithms to create classifier ensembles that dynamically adapt to class imbalanced data sets. Sampling techniques included are random undersampling and synthetic oversampling. More details can be found in 

- Calderón Muro, C.C., Sanchez Tarrago, D.: Development of robust ensemble classifiers to deal with the class imbalance problem in multiple instance classification. Central University Marta Abreu de Las Villas, Santa Clara, Cuba (2015). <a href="https://www.researchgate.net/publication/332593863_Development_of_robust_ensemble_classifiers_to_deal_with_the_class_imbalance_problem_in_multiple_instance_classification" target="_blank">(text)</a>

Developed with:
- Java 1.8
- NetBeans IDE 8.2

Dependencies:
- Weka 3.7
- Weka package citationKNN 1.0.1
- Weka package multiInstanceLearning 1.0.10
- Weka package multiInstanceFilters 1.0.10
- JExcelApi 2.6
