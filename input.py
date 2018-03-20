import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def normalize(data):

    for i in range(4):
        norm = np.linalg.norm(data[:, i])
        if norm == 0:
            norm_data[:, i] = data[:, i]
        norm_data = np.array((data[:, i])/norm)
    return norm_data


inputData = pd.read_excel('./IrisDataTrain.xls', header=None)   # loads data
inputData = inputData.sample(frac=1)                            # shuffles data

features = inputData.iloc[1:, 0:4].values                       # extract features from data
labels = inputData.iloc[1:, 4].values                           # extract class labels from data
print(type(features))

from sklearn.preprocessing import normalize
#norm_features = np.array([features[:, i]/features[:, i].max() for i in range(len(features[1,:]))])             # amplitude normalization
norm_features = normalize(features, axis=0, norm='max')

'''
plt.figure(1)
plt.scatter(norm_features[:, 0], norm_features[:, 1], color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.draw()

plt.figure(2)
plt.scatter(norm_features[:, 2], norm_features[:, 3], color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.draw()
#print([norm_features.max(), norm_features.min()])

print('W tym momencie wykresy powinny być gotowe')
plt.show()'''

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_labels = LabelEncoder()
labels = labelencoder_labels.fit_transform(labels)
labels = np.reshape(labels,(-1, 1))
onehotencoder = OneHotEncoder(categorical_features = [0])
labels = onehotencoder.fit_transform(labels).toarray()

import neuron_network as NN

myNN = NN.NeuronNetwork()
myNN.create_network(2, [4, 3])

print('Utworzono siec')

#myNN.training_input = norm_features
#myNN.training_output = labels

print('dodano dane')

myNN.initialize_network(norm_features, labels)
print(str(len(myNN.training_input[:, 1])))
print('Initialized the network for training')
# WOWOWOWOWOWOW IT IS WORKING SO FAR, DAMNNN
#
# tbh, I am genuinely surprised
# lets go further then, to the moon and beyond!

myNN.train()

print("Trained the network, wohooooo")

'''
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go


app = dash.Dash()

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        id='year-slider',
        min=df['year'].min(),
        max=df['year'].max(),
        value=df['year'].min(),
        step=None,
        marks={str(year): str(year) for year in df['year'].unique()}
    )
])


@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
    [dash.dependencies.Input('year-slider', 'value')])
def update_figure(selected_year):
    traces = []
    traces.append(go.Scatter(
        x=features[:,1],
        y=features[:, 2],
        mode='markers',
        opacity=0.7,
        marker={
            'size': 15,
            'line': {'width': 0.5, 'color': 'white'}
        },
        name=1
    ))
    return {
        'data': traces,
        'layout': go.Layout(
            hovermode='closest'
        )
    }'''


'''if __name__ == '__main__':
    app.run_server()'''




