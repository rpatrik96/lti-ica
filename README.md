---

<div align="center">    
 
# An Interventional Perspective on Identifiability in Gaussian LTI Systems with Independent Component Analysis

[![Paper](http://img.shields.io/badge/arxiv-cs.LG:2311.18048-B31B1B.svg)](https://arxiv.org/abs/2311.18048)

[![Conference](http://img.shields.io/badge/CI4TS@UAI-2023.svg)](https://sites.google.com/view/ci4ts2023/accepted-papers?authuser=0)

![CI testing](https://github.com/rpatrik96/lti-ica/workflows/CI%20testing/badge.svg?branch=main&event=push)

</div>
 
## Description   
We connect the ICA and dynamical systems perspectives on identifiability by showing that for Gaussian Linear Time-Invariant (LTI) systems, experiment design (i.e., introducing agency via prescribing sufficiently varying control signals/interventions) enables identifiability in an active manner.

## How to run   
First, install dependencies   
```bash
# clone lti-ica   
git clone --recurse-submodules https://github.com/rpatrik96/lti-ica


# install lti-ica   
cd lti-ica
pip install -e .   
pip install -r requirements.txt



# install submodule requirements
pip install --requirement tests/requirements.txt --quiet

# install pre-commit hooks (only necessary for development)
pre-commit install
 ```   



## Citation   

```

@inproceedings{
 rajendran2023interventional,
 title={An Interventional Perspective on Identifiability in Gaussian {LTI} Systems with Independent Component Analysis},
 author={Goutham Rajendran and Patrik Reizinger and Wieland Brendel and Pradeep Kumar Ravikumar},
 booktitle={UAI 2023 Workshop on Causal inference for time series data},
 year={2023},
}

```   
