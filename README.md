# Applied-Predictive-Analytics
**Topic**
###
Federated Machine Learning and Applications in Marketing
#
**Summary**
###
The goal is to design a Federated Machine Learning model on marketing data to demonstrate its privacypreserving capabilities
#
**Course Overview**
###
The seminar facilitates following up on our introductory module Business Analytics and Data Science (BADS). Working in a team of two to three students, you will research a real-life modeling problem and develop a machine learning (ML)-based solution. Seminar topics vary from year to year but are always aim at extending your knowledge of and experience with ML methods. More specifically, the application of ML in industry, for example, to support some decision process, will typically involve more than training an ML model and checking its (predictive) performance on a test set. Remember that this was pretty much the setting in BADS. So the goal of APA is to take things a step further. You will learn about new, more specialized ML methods and processes, and experience at least some of the challenges, requirements, and peculiarities that we have to account for when applying analytical models to real-world problems.
#
**Assignment Description**
###
Machine learning (ML) relies on data. More data is always better. Sharing data (e.g., between companies) could be a way to raise the amount of available data and build better ML models. However, sharing data is not trivial and can easily infringe privacy especially if the data relates to people. Federated learning is an approach to train ML models in a decentralized manner. This means that the entire training data does not need to be available on a single computer. Model training works without actually sharing the data between different entities. [The Wikipedia page](https://en.wikipedia.org/wiki/Federated_learning) provides a useful introduction, and so does this entry from [Google's AI blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html).
Promising a balance between enterprises' desire for more data and better models and consumers' preference for higher degrees of privacy, federated learning is a hot topic in industry. The overall goal of the topic is to give an overview of the state-of-the-art in federated learning.
###
To the best of our knowledge, available research papers focus largely on applications in health care and medical data analysis (e.g., Vaid et al. 2020). Applications in the management sciences are yet scarce. Therefore, a second objective of the topic is to provide empirical evidence of the potential of federated learning (or lack thereof) in the scope of marketing analytics. To that end, students will prototype a federated learning system and try to train ML models in a decentralized manner. At least of the available papers on federated learning have made their codes available, ensuring the feasibility of this task.
###
Data: E-commerce couponing data
###
References:
* Vaid, A., Jaladanki, S. K., Xu, J., Teng, S., Kumar, A., Lee, S., . . . Glicksberg, B. S. (2020). Federated Learning of Electronic Health Records Improves Mortality Prediction in Patients Hospitalized with COVID-19. medRxiv, doi:10.1101/2020.08.11.20172809.
Code available at: https://github.com/HPIMS/CovidFederatedMortality
* Rieke, N., Hancox, J., Li, W., Milletarì, F., Roth, H. R., Albarqouni, S., . . . Cardoso, M. J. (2020). The future of digital health with federated learning. npj Digital Medicine, 3(1), 119.
* Choudhury, O., Park, Y., Salonidis, T., Gkoulalas-Divanis, A., Sylla, I., & Das, A. K. (2020). Predicting Adverse Drug Reactions on Distributed Health Data using Federated Learning. AMIA Annual Symposium proceedings. AMIA Symposium, 2019, 313-322.
