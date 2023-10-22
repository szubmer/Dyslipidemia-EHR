# Dyslipidemia-EHR
# Electronic health records database for developing dyslipidemia prediction algorithms

Dyslipidemia has emerged as a significant clinical risk, with its associated complications, including atherosclerosis and ischemic cerebrovascular disease, presenting a grave threat to human well-being. Hence, it holds paramount importance to precisely predict the onset of dyslipidemia and associated risk factors. To advance research in dyslipidemia management, we have curated a database for dyslipidemia prediction, which is now accessible to the research community. This database will establish a robust foundation for the creation of data-driven algorithms and models for dyslipidemia prediction, fostering further exploration into dyslipidemia monitoring and management technologies.

## Repository Structure

- **data_processing.py**: This file is responsible for data preprocessing. It includes functions for cleaning, transforming, and preparing the input data for modeling.

- **feature_extra.py**: This file is dedicated to feature extraction. It contains methods for extracting relevant features from the preprocessed data. These features are crucial for building the anomaly detection model.

- **modeling.py**: In this file, you will find the implementation of the dyslipidemia predict models. We use various machine learning and statistical techniques to develop models that can predict dyslipidemia.

## Python Version and Dependencies

- **Python Version**: Python 3.9

- **Dependencies**:
  - **numpy**: Version 1.23.5
  - **pandas**: Version 2.1.1
  - **matplotlib**: Version 3.8.0
  - **GPy**: Version 1.12.0
  - **scipy**: Version 1.11.3
  - **sklearn**: Version 1.3.1
  - **tensorflow**: Version 2.14.0
  - **xgboost**: Version 2.0.0
  - **imblearn**: Version 0.11.0

