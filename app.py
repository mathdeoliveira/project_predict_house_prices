import flask
import pandas as pd 
import pickle

# importar modelo e feature names
with open('model/modelo.pkl', 'rb') as file:
  model = pickle.load(file)
with open('model/features.names', 'rb') as file:
  features_names = pickle.load(file)
with open('model/district_encoder.pkl', 'rb') as file:
  district_encoder = pickle.load(file)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET','POST'])

def main():
    if flask.request.method=='GET':
        return flask.render_template('home.html')
    
    if flask.request.method=='POST':
        user_inputs = {
            'condo' : flask.request.form['condo'],
            'size' : flask.request.form['size'],
            'rooms' : flask.request.form['rooms'],
            'toilets' : flask.request.form['toilets'],
            'suites' : flask.request.form['suites'],
            'parking' : flask.request.form['parking'],
            'district' : district_encoder.transform([flask.request.form['district']])[0]
        }

        df = pd.DataFrame(index=[0], columns=features_names)
        df = df.fillna(value=0)
        for i in user_inputs.items():
                df[i[0]] = i[1]
        df = df.astype(float)

        y_pred = model.predict(df)[0] 

        return flask.render_template('home.html', valor_venda=y_pred)

if __name__ == '__main__':
        app.run()
