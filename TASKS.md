# Telco Churn Project Tasks

## Data Cleaning

- [x] **Optimize Data Types**

  - [x] `Gender`: Convert to `Category`
  - [x] `Senior Citizen`: Map 0,1 to No, Yes -> Convert to `Category`
  - [x] `Partner`: Convert to `Category`
  - [x] `Dependents`: Convert to `Category`
  - [x] `Tenure`: Convert to `Int8`
  - [x] `Phone Service`: Convert to `Category`
  - [x] `MultipleLines`: Convert to `Category`
  - [x] `InternetService`: Convert to `Category`
  - [x] `Online Security`: Convert to `Category`
  - [x] `DeviceProtection`: Convert to `Category`
  - [x] `TechSupport`: Convert to `Category`
  - [x] `StreamingTV`: Convert to `Category`
  - [x] `StreamingMovies`: Convert to `Category`
  - [x] `Contract`: Convert to `Category`
  - [x] `PaperlessBilling`: Convert to `Category`
  - [x] `PaymentMethod`: Convert to `Category`
  - [x] `MonthlyCharges`: Convert to `float16`
  - [x] `TotalCharges`: Drop rows with " " and
  - [x] `Churn`: Convert to `Category`

- [x] **Simplify Multi-class Categoricals**

  - [x] Replace `No internet service` with `No` across all 6 service columns:
        (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`)
  - [x] Replace `No phone service` with `No` in the `MultipleLines` column.

- [x] **Binary Encoding**

  - [x] Map 0, 1 to `No`, `Yes` in the `SeniorCitizen` column

- [x] **Feature Creation**
  - [x] Create a `TotalServices` column that counts number of services
