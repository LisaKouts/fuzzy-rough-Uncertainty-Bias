# Fuzzy-rough-Uncertainty-Bias
This repository contains:
- the source code related to the conference paper Bias Quantification for Protected Features in Pattern Classification Problems published at the 25th Iberoamerican Congress on Pattern Recognition in March 2021. The paper can be found here: https://www.researchgate.net/publication/353098759_Bias_Quantification_for_Protected_Features_in_Pattern_Classification_Problems

- the Master Thesis with the initial experiments that led to the paper above. The thesis was written in framework of the program Data Science and Society in Tilburg University. It was submitted on January 15, 2021. It contains our initial attempts to create a fuzzy-rough set-based bias quantification measure. After submission, we developed a stronger measure which is presented in the conference paper Bias Quantication for Protected Features in Pattern Classication Problems (Koutsoviti Koumeri & Napoles, 2021)

**Description**

We propose a new bias quantification measure that relies on the fuzzy-rough set theory. The intuition of our measure is that protected features should not change the fuzzy-rough set boundary regions signiﬁcantly. The extent to which this happens can be understood as a proxy for bias quantiﬁcation. Our measure can be categorized as an individual fairness measure sincethe fuzzy-rough regions are computed using instance-based information pieces. 

**Experimental setup**

We tested our measure on the German Credit data set (retrieved from UCI Machine Learning Repository) which is widely used in the context of AI Fairness. The source code can be found in the 'fuzzy-rough uncertainty.ipynb' file. Then, we compare our measure to popular literature measures using the AIF360 (see https://aif360.mybluemix.net) toolkit. The source code for this can be found in the 'baselines.ipynb' file.

We look forward to any comments or feedback. 
Feel free to send your questions to lisa.koutsoviti@uhasselt.be

An extention of the paper can be found here: https://www.researchgate.net/publication/354065853_A_fuzzy-rough_uncertainty_measure_to_discover_bias_encoded_explicitly_or_implicitly_in_features_of_structured_pattern_classification_datasets. The source code for the extention will be released soon. 
