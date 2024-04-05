from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import joblib
# from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__) # Initiate the web app
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app) # Initiate the database, which is necessary to store data

# Load the models and encoders
decision_tree_model = joblib.load('static/models/decision_tree_model.joblib')
port_arv_encoder = joblib.load('static/models/port_arv_encoder.joblib')
sex_encoder = joblib.load('static/models/sex_encoder.joblib')

# class Todo(db.Model):
#     '''
#     A database model for todo items. Include 2 fields:
#     id: Integer, primary field
#     title: String, the contents of the todo

#     '''
#     id = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(100))

@app.route('/')
def index():
    # todo_list = Todo.query.all()
    return render_template('index.html')


# @app.route('/add', methods=['POST'])
# def add():
#     title = request.form.get("title")
#     db.session.add(Todo(title=title))
#     db.session.commit()
#     return redirect(url_for("index")) # refresh the page after an addition


# @app.route('/delete/<int:todo_id>')
# def delete(todo_id):
#     todo = Todo.query.filter_by(id=todo_id).first()
#     db.session.delete(todo)
#     db.session.commit()
#     return redirect(url_for("index")) # refresh the page after a deletion

@app.route('/project-overview')
def project_overview():
    return render_template('project-overview.html')

@app.route('/kelly-oneill-bio')
def kelly_oneill_bio():
    return render_template('kelly-oneill-bio.html')

@app.route('/vivian-wei-bio')
def vivian_wei_bio():
    return render_template('vivian-wei-bio.html')

@app.route('/original-dataset')
def original_dataset():
    return render_template('original-dataset.html')

@app.route('/pivotal-phases')
def pivotal_phases():
    return render_template('pivotal-phases.html')

@app.route('/current-progress')
def current_progress():
    return render_template('current-progress.html')

@app.route('/exploratory-findings')
def exploratory_findings():
    return render_template('exploratory-findings.html')

@app.route('/relational-database')
def relational_database():
    return render_template('relational-database.html')

@app.route('/ml-model')
def ml_model():
    return render_template('ml-model.html')

@app.route('/publication-dissemination')
def publication_dissemination():
    return render_template('publication-dissemination.html')

@app.route('/final-deliverables')
def final_deliverables():

    # Load the CSV files into pandas DataFrames

    occupations_df = pd.read_csv(os.path.join(app.static_folder, 'dataverse_files', 'ttav_occupations.csv'))
    occ_nm_to_id = occupations_df.set_index('occ_nm')['occID'].to_dict()
    occ_names = list(occ_nm_to_id.keys())
    # # Render your template, passing the items list
    # return render_template('your-template.html', items=items)
    return render_template('final-deliverables.html', occ_names=occ_names)

@app.route('/predict_port', methods=['GET', 'POST'])
def predict_port():
    port_arv = None  # Initialize the variable to None
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        occ_nm = request.form['occ']

        # Encode sex and occupation
        encoded_sex = sex_encoder.transform([[sex]])[0]
        occupations_df = pd.read_csv(os.path.join(app.static_folder, 'dataverse_files', 'ttav_occupations.csv'))
        occ_nm_to_id = occupations_df.set_index('occ_nm')['occID'].to_dict()
        occ_names = list(occ_nm_to_id.keys())
        occID = occ_nm_to_id[occ_nm]

        # Predict using the model
        prediction = decision_tree_model.predict(pd.DataFrame([[age, encoded_sex, occID]], columns=['age', 'encoded_sex', 'occID']))

        # Decode the port_arv prediction
        decoded_port_arv = port_arv_encoder.inverse_transform(prediction.reshape(-1, 1))[0]

        port_arv = decoded_port_arv  # Set the prediction result
        # Render the form page again if the method is GET
    return render_template('final-deliverables.html', port_arv=port_arv, occ_names=occ_names)



if __name__ == "__main__":
    # db.drop_all() # Make sure initial db is clean
    # db.create_all() # Initialize all tables
    # app.run(debug=True) # development mode
    app.run(debug=True)
