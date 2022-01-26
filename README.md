# Fuzzy-rough-Uncertainty-Bias
This repository contains:
- the Python notebook with the methodology and experimental setup of the conference paper *Bias Quantification for Protected Features in Pattern Classification Problems published at the 25th Iberoamerican Congress on Pattern Recognition in March 2021* which can be found <a href="https://www.researchgate.net/publication/353098759_Bias_Quantification_for_Protected_Features_in_Pattern_Classification_Problems">here</a>.

- the Python notebook with the methodology and experimental setup of the journal paper *A fuzzy-rough uncertainty measure to discover bias encoded explicitly or implicitly in features of structured pattern classification datasets* which can be found <a href="https://www.sciencedirect.com/science/article/pii/S0167865522000058?via%3Dihub">here</a>.

- the master thesis with the initial experiments that led to the paper above. The thesis was written in framework of the program Data Science and Society in Tilburg University. It was submitted on January 15, 2021. It contains our initial attempts to create a fuzzy-rough set-based bias quantification measure. After the thesis submission, we developed a stronger measure which is presented in the above mentioned conference paper.

**Description**

We propose a new bias quantification measure that relies on the fuzzy-rough set theory. 

The intuition of our measure is that protected features should not change the fuzzy-rough set boundary regions signiﬁcantly. The extent to which this happens can be understood as a proxy for bias quantiﬁcation. Our measure can be categorized as an individual fairness measure since the fuzzy-rough regions are computed using instance-based information pieces. 

**Experimental setup**

We tested our measure on the German Credit data set (retrieved from UCI Machine Learning Repository). The source code can be found in the 'fuzzy-rough uncertainty.ipynb' file. Then, we compare our measure to popular literature measures using the AIF360 (see https://aif360.mybluemix.net) toolkit. The source code for this can be found in the 'baselines.ipynb' file.

We look forward to any comments or feedback. 
Feel free to send your questions to lisa.koutsoviti@uhasselt.be
