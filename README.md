# Predictor

The purpose of this model is to make Return on Investment (ROI) predictions for a beauty retailer's customer sampling program of different beauty brands using random forest modeling and multilinear regression techniques with Python’s SciPy, Pandas, and Scikit-learn libraries. The goal is to predict future ROIs for a brand's samples based off of previous data of that brand's sampling success, then predict future ROIs for brands that do NOT have any previous sampling data (explained in a little bit). 

To predict the future ROI success of a brand's samples, we first train a model on the based on several categories: the brand (brand), that brand's product department (world), type of sampling program (program), and that brand's product category (brandBucketCategory). We will then create predictions for the brands that DO have past sampling data. To create predictions for brands WITHOUT previous sampling data, we will train another model based off of the same training categories, but this time we will take out consideration of the "brand" category - hence only training the ROIs based on the other categories. We will then create predictions for the brands that DON'T have past ROI data based off of the qualities they have regarding program, world, etc...

These predictions are outputted to a predictions folder, and the model code also has the capability to display a visualization of the results as well as the raw data.

### How the ROI Model Works:
Code: Predictive_Model/prediction_BROI_2.0.py
1) Import data from /input_data
	- Brands with previous ROI data: data_ROI_brand_generic.json
	- List of all brands, including brands with NO ROI data: no_ROI_brand_generic.json
IN main() FUNCTION
2) USER: Define columns to optimize on:
	- columns_to_keep = ["brand", "brandBucketCategory", "program", "brandROI", "world"]
	- no_data_columns_to_keep = ["brandBucketCategory", "program", "brandROI", "world"]
		- Don't use "brand" column
3) USER: Define model to use for optimization
	- Options: ridge_model, linear_model, randomForest_model, optimizedRF_model, extraForest_model
	- Current (Best?): optimizedRF_model
	- Can modify parameters further up in code
	- Add new models as new functions
4) Train model for brands with ROI data
5) Train model for brands without ROI data
6) Predict ROI output for brands with ROI data using model from step (4) (unneeded if using ALL columns to optimize on)
	- Predictions done because we used less columns than the original data
7) Predict ROI output for brands without ROI data using model from step (5)
8) Append ROI predictions for brands with and without ROI data into the same dataframe
9) Output said dataframe to /predictions:
	-  predicted_brandROI_v*.json (currently v2)

### ADDITIONAL FUNCTIONALITIES:
1) Optimize parameters for randomForest model
	- Uncomment randomForest_optimizer
	- Prints out estimation of better parameters for model
2) Visualize data in 2 or 3 dimenstions
	- Uncomment visualizer(...)
	- df_clean_data: visualize original data
	- df_output: visualize output data
	- Axes (2 or 3): Showcases brandROI for each axis
		eg: [axis_1, axis_2, axis_3]
		    ["program", "brandBucketCategory", "world"]
		- Creates different graphs for each axis_1 (program)
		- x-axis: Each axis_2 (brandBucketCategory)
		- y-axis: Aggregated ROIs for each axis_2 in axis_1
		- (optional) z-axis: Color coded data points for each axis_3 item in axis_2
			- Color legend is also created


