import pandas as pd
import json

# read the CSV file into a Pandas dataframe
data = pd.read_csv('data.csv')

# infer the data types of each column
data_types = data.dtypes.apply(lambda x: x.name).to_dict()

# generate a Vega-Lite specification based on the data types
vega_spec = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "data": {"values": data.to_dict(orient='records')},
    "mark": "point",
    "encoding": {
        "x": {"field": data.columns[0], "type": data_types[data.columns[0]]},
        "y": {"field": data.columns[1], "type": data_types[data.columns[1]]}
    }
}

# output the Vega-Lite specification as JSON
vega_json = json.dumps(vega_spec)
with open('vega.json', 'w') as f:
    json.dump(vega_spec, f)