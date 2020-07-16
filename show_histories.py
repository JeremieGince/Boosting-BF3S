import json, os
from modules.util import plotHistory

filename: str = "training_data/prototypical_few_shot_learner-conv-4-64_backbone_30way5shot_5tway5tshot_085_c_67acc/cp-history.json"
model_name = os.path.basename(os.path.dirname(filename))

with open(filename, 'r') as file:
    plotHistory(
        json.load(file),
        savefig=False,
        savename=f"training_curve_{model_name}"
    )

