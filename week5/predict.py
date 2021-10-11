from flask import Flask, request, jsonify
import pickle

app = Flask('churn')

dv_file = "dv.bin"
model_file = "model1.bin"
with open(dv_file, 'rb') as dv_in:
	dv = pickle.load(dv_in)
		
with open(model_file, 'rb') as model_in:
	model = pickle.load(model_in)

@app.route('/predict', methods=['POST'])
def predict():
	customer = request.get_json()
	y_pred =  model.predict_proba(dv.transform(customer))[0, 1]
	churn = y_pred >= 0.5

	result ={
		'churn_probability': y_pred,
		#'churn': churn
	}
	return jsonify(result)
	
if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=9696)