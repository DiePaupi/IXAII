# IXAII
The interactive XAI interface (IXAII) was developed as part of my research project regarding "Interactive Explainable Intelligent Systems".
It is an interface utilizing multiple XAI methods (SHAP, LIME, Anchors, and DiCE) and presenting several interactivity options (such as selecting audience views, the reference methods to show, and the format of the explanations).
As this is only a prototype, it is only adapted for the IRIS dataset.

## Loading IXAII
Note that you'll need to install the following packages to run IXAII (e.g., via "pip install <package_name>):
- anchor-exp
- base64
- dash
- dash_bootstrap_components
- dice-ml
- pandas
- lime
- math
- matplotlib
- numpy
- plotly
- re
- shap
- sklearn
- time
- xgboost

Further note that IXAII currently requires dash 2.18.2 and dash-bootstrap-components 1.7.1 to run.
For specifications please see the requirements.txt