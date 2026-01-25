- Business Objective: Predict the future federal reserve interest rate in 2027 based on the current and previous interest rates
to make decisions on investments. 

- The current situation is unknown so this would be a use case to get informed using a machine learning model. 

- This is a supervised learning multi- regression problem using batch processing. 

- Will use root mean squared error as the metric to evaluate the model. 

- The Effective Federal Reserve Interest Rate is the average of the federal reserve interest rate over the past 12 months. 

- I assumed I had all the data I needed to make a prediction and train the model, but maybe I need more data. 

- These are the fields I will need to make a prediction:

    -   Inflation Rate (specifically the Personal Consumption Expenditures (PCE) index, which is the Fed's preferred gauge)( This is blank in some rows in the dataset)
    -   Unemployment Rate (a key component of the Fed's "dual mandate" for maximum employment)
    -   Real GDP (Percent Change) (used to assess economic growth and potential overheating)
    -   Effective Federal Funds Rate (provides the historical baseline for modeling future shifts)
    -   Federal Funds Target Rate (the primary policy tool being predicted)
    -   Federal Funds Upper and Lower Target (the boundaries set by the Fed for the effective rate)
    -   Time Period (Year, Month, and Day for time-series analysis)







