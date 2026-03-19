# Flight Delay & Cancellation Analysis (2019–2023)

## Overview
This project analyzes **U.S. domestic flight delays and cancellations from 2019 to 2023** using a large-scale aviation dataset sourced from the **U.S. Department of Transportation (BTS)** via Kaggle.

Our goal is to identify patterns in delays and cancellations across **time, airlines, and airports**, and to build a scalable workflow for exploratory data analysis, dashboarding, and future predictive modeling.

## Project Question
**What factors drive flight delays and cancellations, and how do these patterns vary across time, airlines, and airports?**

---

## Team
- **Danish**
- **Denisha**
- **Shruthi** 
- **Kavan**

---

## Dataset
**Source:** U.S. Department of Transportation (BTS), via Kaggle  
**Time Range:** 2019–2023  
**Working Sample:** ~3 million flight records  
**Scalable Full Dataset:** ~29 million+ rows  

### Key Variables
- Airline / carrier
- Origin and destination airports
- Departure and arrival times
- Departure and arrival delays
- Cancellation flag and cancellation reason
- Distance
- Delay cause fields:
  - Carrier
  - Weather
  - NAS
  - Security
  - Late aircraft

---

## Why This Dataset
This dataset is strong for data visualization and analytics because it is:

- **Large-scale** and operationally realistic
- **Multi-dimensional**, covering time, routes, carriers, and disruption causes
- **Noisy**, with missing values caused by cancellations and diversions
- Suitable for:
  - data cleaning
  - exploratory analysis
  - interactive dashboards
  - machine learning direction

---

## Data Cleaning & Processing
We used **NVIDIA RAPIDS (cuDF) in Google Colab** to accelerate preprocessing on the GPU.

### Cleaning Steps
- Handled missing values from cancelled and diverted flights
- Converted `HHMM` time fields into total minutes
- Built consistent datetime features
- Derived scheduled vs actual block time
- Cleaned extreme or unrealistic delay values
- Created delay buckets for more stable analysis

### Feature Engineering
- Extracted temporal features such as:
  - month
  - day
  - hour
- Created cleaned delay variables
- Built categories aligned with visualization and machine learning tasks

### Why RAPIDS
Using RAPIDS allowed us to:
- process millions of rows efficiently
- accelerate filtering and transformations
- scale preprocessing beyond standard CPU-based workflows

---

## Exploratory Data Analysis (EDA)
We developed **6+ meaningful visualizations** across Plotly and Tableau.

### Main EDA Findings
1. **Flight delays show strong seasonal and temporal patterns**
   - Delays rise during summer and year-end travel periods
   - Day-by-month analysis shows recurring congestion windows

2. **Most flights are on time or only slightly delayed**
   - Severe delays are less frequent
   - However, they have a disproportionate operational impact

3. **Delay causes are concentrated**
   - Late aircraft and carrier-related issues account for a large share of delays

4. **Airline performance differs significantly**
   - Some carriers experience consistently higher delays
   - Recovery behavior varies across airlines

5. **Cancellation rates vary across airlines**
   - Some carriers show materially higher cancellation risk than others

6. **Operational composition varies by airline**
   - Taxi time, air time, and delay structure suggest differences in operational efficiency

---

## Visualizations Used

### Tableau Visuals
- **Delay Recovery Quadrant**
- **Flight Delay Patterns Over Time (Heatmap)**
- **Flight Delay Composition by Airline**

### Plotly Visuals
- **Monthly Average Delay Trend**
- **Departure Delay Distribution**
- **Cancellation Reasons Donut**
- **Top Airlines by Average Departure Delay**
- **Top Airlines by Cancellation Rate**

---

## Dashboards

### 1. Plotly Dashboard
Built in **Python using Plotly + Dash**

Includes:
- monthly delay trends
- delay distribution
- cancellation reasons
- airline delay rankings
- cancellation rate rankings
- interactive filters and KPI cards

### 2. Tableau Dashboard
Built in **Tableau**

Includes:
- delay recovery quadrant
- temporal heatmap
- flight delay composition by airline
- interactive filtering and comparative analysis

---

## Machine Learning Direction
We propose a **binary classification task**:

**Predict whether a scheduled flight will have a departure delay greater than 15 minutes.**

### Planned Features
- airline carrier
- origin and destination routes
- month and hour
- flight distance

### Why This Makes Sense
Our EDA revealed strong and consistent delay patterns across:
- time
- airlines
- routes

This suggests that delay risk can be predicted from **pre-departure operational features**.

---

## Tools & Technologies
- **Python**
- **Pandas**
- **Plotly**
- **Dash**
- **Tableau**
- **NVIDIA RAPIDS / cuDF**
- **Google Colab**
