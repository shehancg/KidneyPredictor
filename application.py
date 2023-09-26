from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import csv
import boto3
import dotenv
from io import StringIO
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

dotenv.load_dotenv()

application = Flask(__name__)

s3 = boto3.client('s3')

bucket_name = 'flaskapp1'
file_name = 'clean_data2.csv'

response = s3.get_object(Bucket=bucket_name, Key=file_name)
kidney_data = pd.read_csv(response['Body'])

# Encode categorical variables (Label Encoding)
label_encoder = LabelEncoder()
kidney_data['race'] = label_encoder.fit_transform(kidney_data['race'])
kidney_data['sex'] = label_encoder.fit_transform(kidney_data['sex'])
kidney_data['Blood_type'] = label_encoder.fit_transform(kidney_data['Blood_type'])

# Split data into features (X) and target (y)
X = kidney_data.drop(columns=['Id'])  # Exclude 'Id' and 'compatibility_score'
y = kidney_data['Id']

# Standardize numeric features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit the k-Nearest Neighbors model
k = 5  # Number of nearest neighbors to consider
model = NearestNeighbors(n_neighbors=k, metric='euclidean')  # You can choose different metrics
model.fit(X)


@application.route('/predict', methods=['POST'])
def get_kidney_recommendations():
    if request.method == 'POST':
        # Retrieve user-entered donor kidney information from JSON
        user_input = request.json

        # Prepare the user input data
        user_input_features = np.array([[
            user_input['age'],
            user_input['race'],
            user_input['sex'],
            user_input['Blood_type'],
            user_input['HLA_A1'],
            user_input['HLA_A2'],
            user_input['HLA_B1'],
            user_input['HLA_B2'],
            user_input['HLA_DR1'],
            user_input['HLA_DR2'],
            user_input['anti_HBc'],
            user_input['anti_HCV'],
            user_input['agHBs']
        ]])
        user_input_features = scaler.transform(user_input_features)

        # Find the k nearest neighbors to the user input
        distances, indices = model.kneighbors(user_input_features)

        # Get the IDs and compatibility scores of the nearest kidney recipients
        nearest_neighbor_ids = kidney_data.iloc[indices[0]]['Id'].values

        compatibility_scores = distances[0]

        # Combine IDs and compatibility scores into a list of dictionaries
        results = []
        for i in range(len(nearest_neighbor_ids)):
            result = {
                'id': int(nearest_neighbor_ids[i]),
                'compatibility_score': round(float(compatibility_scores[i]), 4)
            }
            results.append(result)

        # Convert the results list to a JSON response
        result_json = {
            'kidney_recipients': results
        }

        return jsonify(result_json)

    # Render the initial form page
    return render_template('index.html')

@application.route('/')
def hello_world():
    return 'Hello,'

# Load your dataset from S3
response = s3.get_object(Bucket=bucket_name, Key=file_name)
csv_data = response['Body'].read().decode('utf-8')
kidney_data = pd.read_csv(StringIO(csv_data))

@application.route('/addData', methods=['POST'])
def add_data_new():
    global kidney_data
    if request.method == 'POST':
        # Retrieve the data sent in the request
        data = request.json  # Assuming the data is sent as JSON

        # Generate a new unique ID for the added data (you may want to implement this differently)
        new_id = max(kidney_data['Id']) + 1

        # Create a new DataFrame for the data to be added
        new_data = pd.DataFrame(data, index=[0])
        new_data['Id'] = new_id

        # Concatenate the new data with the existing kidney_data DataFrame
        kidney_data = pd.concat([kidney_data, new_data], ignore_index=True)

        # Convert the updated DataFrame to CSV format
        updated_csv_data = kidney_data.to_csv(index=False)
        
        # Upload the updated CSV data back to S3
        s3.put_object(Body=updated_csv_data, Bucket=bucket_name, Key=file_name)

        # Return a JSON response indicating success
        response_data = {'message': 'Data Added successfully', 'id': new_id}
        return jsonify(response_data), 200  # 200 indicates success

    # Return a JSON response for invalid requests
    response_data = {'message': 'Invalid request'}
    return jsonify(response_data), 400  # 400 indicates a bad request


if __name__ == '__main__':
    application.run(debug=True)
