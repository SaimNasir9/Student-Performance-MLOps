from flask import Flask, render_template, request
from scr.pipelines.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_prep'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            df = data.get_data_as_dataframe()
            pipeline = PredictPipeline()
            results = pipeline.predict(df)
            return render_template('predict.html', results=round(results[0], 2))
        
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)