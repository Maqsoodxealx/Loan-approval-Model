#  Loan Approval Prediction System
A Machine Learning-powered Loan Approval Prediction System with an interactive Gradio Web Application developed in Google Colab
## Project Overview
The “Loan Approval Prediction System” is a Machine Learning application that predicts whether a customer's loan application is likely to be approved based on demographic, financial, and loan-related information.
The project demonstrates the complete Machine Learning workflow including:                              
•	Data Cleaning                                                                                    
•	Exploratory Data Analysis (EDA)                                                                  
•	Feature Engineering                                                                                  
•	Data Encoding                                                                                    
•	Model Training                                                                                    
•	Model Comparison                                                                                    
•	Model Evaluation                                                                                    
•	Model Serialization                                                                              
•	Web Application Development using Gradio                                                            
The application was developed entirely in Google Colab and deployed as a Gradio interface for real-time predictions.
## Objectives
•	Predict loan approval using applicant information.                                                   
•	Compare multiple Machine Learning algorithms.                                                      
•	Select the best-performing model using evaluation metrics.                                          
•	Deploy the trained model as an easy-to-use web application.                                          
•	Demonstrate a complete end-to-end Machine Learning project.                                          
## Repository Structure            
•	Loan-Approval-Prediction                                                                             
•	LoanApprovalDataset.csv                                                                              
•	Loanapprovalmodel.py                                                                              
•	Apploanapprovalmodel.py                                                                              
•	LoanAapprovalTrainedModel.joblib                                                                  
•	onehotencoder.joblib                                                                              
•	ordinalencoder.joblib                                                                              
•	README.md                                                                                          
## Dataset
The project uses a publicly available “Loan Approval Prediction Dataset” containing approximately “52,000 loan applications” collected from financial institutions.                                                
The dataset contains applicant information such as:                                                      
•	Gender                                                                                              
•	Age                                                                                                  
•	Marital Status                                                                                    
•	Education                                                                                          
•	Employment Status                                                                                    
•	Occupation                                                                                          
•	Annual Income                                                                                        
•	Monthly Expenses                                                                                    
•	Existing Loan Amount                                                                                 
•	Outstanding Debt                                                                                    
•	Loan History                                                                                         
•	Loan Purpose                                                                                    
•	Loan Type                                                                                          
•	Interest Rate                                                                                    
•	Loan Amount Requested                                                                              
•	Loan Duration                                                                                    
•	Loan Approval Status                                                                              
The dataset is suitable for:                                                                              
o	Credit Risk Analysis                                                                              
o	Predictive Analytics                                                                              
o	Financial Modeling                                                                                   
o	Classification Problems                                                                              
## Data Preprocessing
Several preprocessing steps were performed before model training.
### Data Cleaning
o	Removed unnecessary columns                                                                        
o	Removed missing values                                                                              
o	Removed duplicate records                                                                        
o	Cleaned column names                                                                              
o	Checked data consistency                                                                             
### Exploratory Data Analysis (EDA)
The project includes multiple visualizations such as:                                                      
o	Gender Distribution                                                                              
o	Age Distribution                                                                                    
o	Loan Approval Distribution                                                                        
o	Marital Status Analysis                                                                              
o	Outlier Detection using Boxplots                                                                  
o	Feature Statistics                                                                              
o	Approval Rate Comparisons                                                                            
Key observations include:                                                                              
o	Dataset contains approximately  “52,000 records”                                                     
o	Gender distribution is nearly balanced                                                            
o	Around 64% of applications are approved                                                          
o	Working-age applicants dominate the dataset                                                      
o	Income and requested loan amount show high variability                                          
o	Gender and marital status have minimal influence on approval outcomes                              
### Feature Engineering
Two encoding techniques were used.                                                                        
### Ordinal Encoding      
Applied to:                                                                                          
- Education                                                                                                
Order used:                                                                                                
High School                                                                                                
↓                                                                                                       
Graduate                                                                                                
↓                                                                                                      
Postgraduate                                                                                          
### One-Hot Encoding
Applied to nominal categorical variables including:                                                      
o	Gender                                                                                          
o	Marital Status                                                                                    
o	Employment Status                                                                              
o	Occupation Type                                                                                    
o	Residential Status                                                                                   
o	City/Town                                                                                          
o	Loan Purpose                                                                                    
o	Loan Type                                                                                          
o	Co-Applicant                                                                                    
### Machine Learning Models Evaluated
The following algorithms were trained and compared:                                                      
o	Logistic Regression                                                                              
o	Support Vector Machine (SVC)                                                                        
o	Linear SVC                                                                                          
o	Decision Tree                                                                                    
o	Random Forest                                                                                        
o	Gradient Boosting                                                                                    
o	Extra Trees                                                                                          
o	XGBoost                                                                                              
o	LightGBM                                                                                          
### Model Evaluation Metrics
Models were evaluated using:                                                                              
o	Accuracy                                                                                          
o	Precision                                                                                          
o	Recall                                                                                          
o	F1 Score                                                                                          
Since loan approval is a classification problem where both false approvals and false rejections are important, “F1 Score” was selected as the primary evaluation metric.
### Best Performing Model
After comparing all algorithms, “LightGBM” achieved the highest overall performance.                       
Metric ------------------------------------- Score                                                       
Accuracy---------------------------------  0.855                                                       
Precision --------------------------------- 0.857                                                       
Recall ------------------------------------- 0.935                                                       
F1 Score---------------------------------  0.894                                                           
Therefore, “LightGBM” was selected as the final prediction model.                                          
### Saved Model Files
The trained components were saved using Joblib.                                                      
o	LoanAapprovalTrainedModel.joblib                                                                  
o	onehotencoder.joblib                                                                              
o	ordinalencoder.joblib                                                                              
These files are loaded by the application during prediction.                                               
### Gradio Web Application
A user-friendly Gradio interface was developed for real-time loan approval prediction.                  
#### User Inputs
#### Personal Information
o	Gender                                                                                          
o	Age                                                                                                
o	Marital Status                                                                                    
o	Education                                                                                          
#### Employment Information
o	Employment Status                                                                                    
o	Occupation                                                                                          
o	Residential Status                                                                                   
o	Region                                                                                               
#### Loan Information
o	Loan Type                                                                                          
o	Loan Purpose                                                                                         
o	Loan History                                                                                    
o	Co-Applicant                                                                                         
o	Interest Rate                                                                                    
o	Loan Duration                                                                                    
#### Financial Information
o	Annual Income                                                                                    
o	Monthly Expenses                                                                                    
o	Existing Loan Amount                                                                              
o	Outstanding Debt                                                                                    
o	Requested Loan Amount                                                                              
###  Prediction Workflow      
User Input                                                                                                
      │                                                                                                
      ▼                                                                                                
Input Validation                                                                                          
      │                                                                                                
      ▼                                                                                                
Ordinal Encoding                                                                                          
      │                                                                                                
      ▼                                                                                                
One-Hot Encoding                                                                                    
      │                                                                                                
      ▼                                                                                                
Feature Combination                                                                                    
      │                                                                                                
      ▼                                                                                                
LightGBM Model                                                                                          
      │                                                                                                
      ▼                                                                                                
Prediction                                                                                                
      │                                                                                                
      ▼                                                                                                
Loan Approved                                                                                          
or                                                                                                      
Loan Not Approved                                                                                    
### Running the Project
### Clone the Repository
git clone https://github.com/Maqsoodxealx/Loan-Approval-Prediction.git
### Install Dependencies
•	pip install pandas                                                                              
•	pip install numpy                                                                                    
•	pip install scikit-learn                                                                             
•	pip install lightgbm                                                                              
•	pip install xgboost                                                                                 
•	pip install gradio                                                                                   
•	pip install joblib                                                                              
### Run the Application
python Apploanapprovalmodel.py                                                                             
Gradio will generate a local (and optionally public) URL where the application can be accessed.            
### Technologies Used
o	Python                                                                                          
o	Pandas                                                                                               
o	NumPy                                                                                                
o	Scikit-Learn                                                                                    
o	LightGBM                                                                                          
o	XGBoost                                                                                          
o	Joblib                                                                                          
o	Gradio                                                                                          
o	Google Colab                                                                                         
### Learning Outcomes
This project demonstrates practical knowledge of:                                                      
o	Data Cleaning                                                                                    
o	Data Visualization                                                                              
o	Feature Engineering                                                                              
o	Machine Learning Classification                                                                  
o	Model Evaluation                                                                                    
o	Model Serialization                                                                              
o	Model Deployment                                                                                    
o	Interactive AI Applications                                                                        
### Future Improvements      
•	Potential enhancements include:                                                                  
o	Hyperparameter Optimization                                                                        
o	Cross-Validation                                                                                    
o	Explainable AI (SHAP/LIME)                                                                           
o	Streamlit or Flask Deployment                                                                        
o	Cloud Deployment (Hugging Face Spaces, Render, Railway)                                          
o	REST API Integration                                                                              
o	Database Support                                                                                    
o	User Authentication                                                                              
### License
This project is intended for educational and academic purposes.                                          
### Author
Maqsood Ahmad                                                                                    
MA Econonomics, AI Engineering                                                                             
Machine Learning Project                                                                              
Developed using “Python, Google Colab, Scikit-Learn, LightGBM, and Gradio”.                                
## ⭐ If you found this project useful, please consider giving it a Star on GitHub!
