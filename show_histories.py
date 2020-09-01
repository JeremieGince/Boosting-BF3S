import json, os
from modules.util import plotHistory

filename: str = "training_data/Gen0-conv-4-64_5tway5tshot_97/cp-history.json"
model_name = os.path.basename(os.path.dirname(filename))

try:
    with open(os.path.dirname(filename)+"/test_results.json") as test_file:
        print(json.load(test_file).get("pprint"))
except FileNotFoundError:
    print(f"No test has been executed")

with open(filename, 'r') as file:
    plotHistory(
        json.load(file),
        savefig=True,
        savename=f"training_curve_{model_name}"
    )



