import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from flask import Flask, jsonify, request, render_template, request

app = Flask(__name__)

model = pickle.load(open('xgb_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# def apply_preprocessing(input_df, train=False, return_mapping=False):
#     input_df['Gender'] = input_df['Gender'].astype('category')
#     input_df['Gender'] = input_df['Gender'].cat.codes

#     input_df['Active Member'] = input_df['Active Member'].astype('category')
#     input_df['Active Member'] = input_df['Active Member'].cat.codes

#     input_df['Credit Card'] = input_df['Credit Card'].astype('category')
#     input_df['Credit Card'] = input_df['Credit Card'].cat.codes

#     input_df['Balance'] = input_df['Balance'].astype('int64')
#     input_df['EstimatedSalary'] = input_df['EstimatedSalary'].astype('int64')

#     if train:
#         input_df = input_df[(input_df['Age'] >= 18) & (input_df['Age'] <= 70)]

#     if not is_single_input:
#         input_df = pd.get_dummies(input_df, columns=['Geography'])

#     # input_df = pd.get_dummies(input_df, columns=['Geography'])

#     num_features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Credit Card', 'Active Member', 'Geography_France', 'Geography_Germany', 'Geography_Spain']
#     input_df[num_features] = scaler.transform(input_df[num_features])

#     if return_mapping:
#         original_indices = input_df.index
#         return input_df, original_indices
#     else:
#         return input_df


def apply_preprocessing(input_df, train=False, return_mapping=False, is_single_input=False):
    input_df['Gender'] = input_df['Gender'].astype('category')
    input_df['Gender'] = input_df['Gender'].cat.codes

    input_df['Active Member'] = input_df['Active Member'].astype('category')
    input_df['Active Member'] = input_df['Active Member'].cat.codes

    input_df['Credit Card'] = input_df['Credit Card'].astype('category')
    input_df['Credit Card'] = input_df['Credit Card'].cat.codes

    input_df['Balance'] = input_df['Balance'].astype('int64')
    input_df['EstimatedSalary'] = input_df['EstimatedSalary'].astype('int64')

    if train:
        input_df = input_df[(input_df['Age'] >= 18) & (input_df['Age'] <= 70)]

    if not is_single_input:
        input_df = pd.get_dummies(input_df, columns=['Geography'])

    num_features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                    'Credit Card', 'Active Member', 'Geography_France', 'Geography_Germany', 'Geography_Spain']
    input_df[num_features] = scaler.transform(input_df[num_features])

    if return_mapping:
        original_indices = input_df.index
        return input_df, original_indices
    else:
        return input_df


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'csvfile' not in request.files:
        return "No file part", 400

    csvfile = request.files['csvfile']
    if csvfile.filename == '':
        return "No selected file", 400

    # Read the uploaded CSV file
    input_df = pd.read_csv(csvfile.stream)
    # Create a copy of the original input DataFrame
    input_df_original = input_df.copy()

    # Preprocess the input data
    preprocessed_data, original_indices = apply_preprocessing(
        input_df, train=False, return_mapping=True)

    # Make sure to remove the 'Name' column before passing the data to the model
    preprocessed_data = preprocessed_data.drop('Name', axis=1)  # type: ignore
    predictions = model.predict(preprocessed_data)

    # Replace 0 with 'Not Churned' and 1 with 'Churned'
    predictions = np.where(predictions == 0, 'Not Churned', 'Churned')

    # Add a new column named 'Predicted Exited' to the input DataFrame
    input_df['Predicted Exited'] = pd.Series(
        predictions, index=original_indices)

    # Insert the 'Name' column from the original DataFrame to the end
    input_df['Name'] = input_df_original['Name']

    # Convert the updated input DataFrame to an HTML table
    result_table = input_df.to_html(index=False, border=0, classes=[
                                    "table", "table-striped", "table-bordered", "table-hover"])

    return render_template('index.html', result_table=result_table)


@app.route('/predict_single', methods=['POST'])
def predict_single():
    float_features = [
        float(request.form['CreditScore']),
        float(request.form['Gender']),
        float(request.form['Age']),
        float(request.form['Tenure']),
        float(request.form['Balance']),
        float(request.form['NumOfProducts']),
        float(request.form['EstimatedSalary']),
        float(request.form['Credit Card']),
        float(request.form['Active Member']),
        float(request.form['Geography_France']),
        float(request.form['Geography_Germany']),
        float(request.form['Geography_Spain'])
    ]
    data = [float_features]

    # Preprocess the input data
    num_features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                    'Credit Card', 'Active Member', 'Geography_France', 'Geography_Germany', 'Geography_Spain']
    # preprocessed_data = apply_preprocessing(pd.DataFrame(data, columns=num_features), train=False)  # Replace YOUR_COLUMNS_LIST with the list of your column names
    # Replace YOUR_COLUMNS_LIST with the list of your column names
    preprocessed_data = apply_preprocessing(pd.DataFrame(
        data, columns=num_features), train=False, is_single_input=True)

    prediction = model.predict(preprocessed_data)
    output = round(prediction[0], 2)

    if output == 0:
        prediction_text = 'Customer will not churn'
    else:
        prediction_text = 'Customer will churn'

    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
