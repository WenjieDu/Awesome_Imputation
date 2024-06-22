<p align="center">
    <a id="AwesomeImputation" href="#AwesomeImputation">
        <img src="https://pypots.com/figs/pypots_logos/AwesomeImputation/banner.jpg"
            alt="Time Series Imputation Survey and Benchmark"
            title="Time Series Imputation Survey and Benchmark"
            width="80%"
        />
    </a>
</p>

The repository for the paper [**TSI-Bench: Benchmarking Time Series Imputation**](https://arxiv.org/abs/2406.12747) 
from <a href="https://pypots.com" target="_blank"><img src="https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg" width="30px" align="center"/> PyPOTS Research</a>.
The code and configurations for reproducing the experimental results in the paper are available under the folder `benchmark_code`.
The README file here maintains a list of must-read papers on time-series imputation, and a collection of time-series imputation toolkits and resources.

🤗 Contributions to update new resources and articles are very welcome!

## ❖ Time-Series Imputation Toolkits
### `Datasets`
[TSDB (Time Series Data Beans)](https://github.com/WenjieDu/TSDB): a Python toolkit can load 170 public time-series datasets with a single line of code.
<img src="https://img.shields.io/github/last-commit/WenjieDu/TSDB" align="center">

[BenchPOTS](https://github.com/WenjieDu/BenchPOTS): a Python suite provides standard preprocessing pipelines of 170 public datasets for benchmarking machine learning on POTS (Partially-Observed Time Series).
<img src="https://img.shields.io/github/last-commit/WenjieDu/BenchPOTS" align="center">

### `Missingness`
[PyGrinder](https://github.com/WenjieDu/PyGrinder): a Python library grinds data beans into the incomplete by introducing missing values with different missing patterns.
<img src="https://img.shields.io/github/last-commit/WenjieDu/PyGrinder" align="center">

### `Algorithms`
[PyPOTS](https://github.com/WenjieDu/PyPOTS): a Python toolbox for data mining on POTS (Partially-Observed Time Series)
<img src="https://img.shields.io/github/last-commit/WenjieDu/PyPOTS" align="center">

[MICE](https://github.com/amices/mice): Multivariate Imputation by Chained Equations
<img src="https://img.shields.io/github/last-commit/amices/mice" align="center">

[AutoImpute](https://github.com/kearnz/autoimpute): a Python package for Imputation Methods
<img src="https://img.shields.io/github/last-commit/kearnz/autoimpute" align="center">

[Impyute](https://github.com/eltonlaw/impyute): a library of missing data imputation algorithms
<img src="https://img.shields.io/github/last-commit/eltonlaw/impyute" align="center">


## ❖ Must-Read Papers on Time-Series Imputation
The papers listed here may be not from top publications, some of them even are not deep-learning methods,
but are all interesting papers related to time-series imputation that deserve reading to 
researchers and practitioners who are interested in this field.

### `Year 2024`

[ICML] **BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition**
[[paper](https://arxiv.org/abs/2308.14906)]

[ICLR] **Conditional Information Bottleneck Approach for Time Series Imputation**
[[paper](https://openreview.net/pdf?id=K1mcPiDdOJ)]
[[official code](https://github.com/Chemgyu/TimeCIB)]

[AISTATS] **SADI: Similarity-Aware Diffusion Model-Based Imputation for Incomplete Temporal EHR Data**
[[paper](https://proceedings.mlr.press/v238/dai24c/dai24c.pdf)]
[[official code](https://github.com/bestadcarry/SADI-Similarity-Aware-Diffusion-Model-Based-Imputation-for-Incomplete-Temporal-EHR-Data)]


### `Year 2023`

[ICLR] **Multivariate Time-series Imputation with Disentangled Temporal Representations**
[[paper](https://openreview.net/forum?id=rdjeCNUS6TG)]
[[official code](https://github.com/liuwj2000/TIDER)]

[ICDE] **PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation**
[[paper](https://arxiv.org/abs/2302.09746)]
[[official code](https://github.com/LMZZML/PriSTI)]

[ESWA] **SAITS: Self-Attention-based Imputation for Time Series**
[[paper](https://arxiv.org/abs/2202.08516)]
[[official code](https://github.com/WenjieDu/SAITS)]

[TMLR] **Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models**
[[paper](https://openreview.net/forum?id=hHiIbk7ApW)]
[[official code](https://github.com/AI4HealthUOL/SSSD)]

[ICML] **Modeling Temporal Data as Continuous Functions with Stochastic Process Diffusion**
[[paper](https://proceedings.mlr.press/v202/bilos23a.html)]
[[official code](https://github.com/morganstanley/MSML/tree/main/papers/Stochastic_Process_Diffusion)]

[ICML] **Provably Convergent Schrödinger Bridge with Applications to Probabilistic Time Series Imputation**
[[paper](https://proceedings.mlr.press/v202/chen23f.html)]
[[official code](https://github.com/morganstanley/MSML/tree/main/papers/Conditional_Schrodinger_Bridge_Imputation)]

[ICML] **Modeling Temporal Data as Continuous Functions with Stochastic Process Diffusion**
[[paper](https://proceedings.mlr.press/v202/bilos23a)]

[ICML] **Probabilistic Imputation for Time-series Classification with Missing Data**
[[paper](https://proceedings.mlr.press/v202/kim23m)]

[KDD] **Source-Free Domain Adaptation with Temporal Imputation for Time Series Data**
[[paper](https://arxiv.org/abs/2307.07542)]
[[official code](https://github.com/mohamedr002/MAPU_SFDA_TS)]

[KDD] **Networked Time Series Imputation via Position-aware Graph Enhanced Variational Autoencoders**
[[paper](https://arxiv.org/abs/2305.18612)]

[KDD] **An Observed Value Consistent Diffusion Model for Imputing Missing Values in Multivariate Time Series**
[[paper](https://dl.acm.org/doi/10.1145/3580305.3599257)]

[TKDE] **Selective Imputation for Multivariate Time Series Datasets With Missing Values**
[[paper](https://drive.google.com/file/d/1ZLDI5NAn5cQhKbWJ7If4tN_pUrMkLgK9/view)]
[[official code](https://github.com/ablazquezg/Selective-imputation)]

[TKDE] **PATNet- Propensity-Adjusted Temporal Network for Joint Imputation and Prediction using Binary EHRs with Observation Bias**
[[paper](https://ieeexplore.ieee.org/document/10285044/)]

[TKDD] **Multiple Imputation Ensembles for Time Series (MIE-TS)**
[[paper](https://drive.google.com/file/d/1jFTOW6L2-pOFY5R2ogLMR0_ZtIovQEp-/view)]

[CIKM] **Density-Aware Temporal Attentive Step-wise Diffusion Model For Medical Time Series Imputation**
[[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614840)]


### `Year 2022`

[ICLR] **Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks**
[[paper](https://arxiv.org/abs/2108.00298)]
[[official code](https://github.com/Graph-Machine-Learning-Group/grin)]

[AAAI] **Online Missing Value Imputation and Change Point Detection with the Gaussian Copula**
[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20906)]
[[official code](https://github.com/yuxuanzhao2295/Online-Missing-Value-Imputation-and-Change-Point-Detection-with-the-Gaussian-Copula)]

[AAAI] **Dynamic Nonlinear Matrix Completion for Time-Varying Data Imputation**
[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20612)]

[AAAI] **Fairness without Imputation: A Decision Tree Approach for Fair Prediction with Missing Values**
[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21189)]


### `Year 2021`

[NeurIPS] **CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation**
[[paper](https://openreview.net/forum?id=VzuIzbRDrum)]
[[official code](https://github.com/ermongroup/CSDI)]

[AAAI] **Generative Semi-supervised Learning for Multivariate Time Series Imputation**
[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17086)]

[VLDB] **Missing Value Imputation on Multidimensional Time Series**
[[paper](https://arxiv.org/abs/2103.01600)]

[ICDM] **STING: Self-attention based Time-series Imputation Networks using GAN**
[[paper](https://arxiv.org/abs/2209.10801)]


### `Year 2020`

[AISTATS] **GP-VAE: Deep Probabilistic Time Series Imputation**
[[paper](https://arxiv.org/abs/1907.04155)]
[[official code](https://github.com/ratschlab/GP-VAE)]

[CVPR] **Imitative Non-Autoregressive Modeling for Trajectory Forecasting and Imputation**
[[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Qi_Imitative_Non-Autoregressive_Modeling_for_Trajectory_Forecasting_and_Imputation_CVPR_2020_paper.html)]

[ICLR] **Why Not to Use Zero Imputation? Correcting Sparsity Bias in Training Neural Networks**
[[paper](https://openreview.net/forum?id=BylsKkHYvH)]

[TNNLS] **Adversarial Recurrent Time Series Imputation**
[[paper](https://drive.google.com/file/d/1AkWlqjYJ1PNgnu5apOx2dow_vgmqViQG/view)]


### `Year 2019`

[NeurIPS] **NAOMI: Non-Autoregressive Multiresolution Sequence Imputation**
[[paper](https://arxiv.org/abs/1901.10946)]
[[official code](https://github.com/felixykliu/NAOMI)]

[IJCAI] **E²GAN: End-to-End Generative Adversarial Network for Multivariate Time Series Imputation**
[[paper](https://www.ijcai.org/proceedings/2019/429)]
[[official code](https://github.com/Luoyonghong/E2EGAN)]

[WWW] **How Do Your Neighbors Disclose Your Information: Social-Aware Time Series Imputation**
[[paper](https://drive.google.com/file/d/1AEzswot_htpwhU4KVt0E7gZW75XKTdy7/view)]
[[official code](https://github.com/tomstream/STI)]


### `Year 2018`

[NeurIPS] **BRITS: Bidirectional Recurrent Imputation for Time Series**
[[paper](https://arxiv.org/abs/1805.10572)]
[[official code](https://github.com/caow13/BRITS)]

[Scientific Reports] **Recurrent Neural Networks for Multivariate Time Series with Missing Values**
[[paper](https://www.nature.com/articles/s41598-018-24271-9)]
[[official code](https://github.com/PeterChe1990/GRU-D)]

[NeurIPS] **Multivariate Time Series Imputation with Generative Adversarial Networks**
[[paper](https://papers.nips.cc/paper_files/paper/2018/hash/96b9bff013acedfb1d140579e2fbeb63-Abstract.html)]
[[official code](https://github.com/Luoyonghong/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks)]


### `Year 2017`

[IEEE Transactions on Biomedical Engineering] **Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks**
[[paper](https://arxiv.org/abs/1711.08742)]
[[official code](https://github.com/jsyoon0823/MRNN)]


### `Year 2016`

[IJCAI] **ST-MVL: Filling Missing Values in Geo-sensory Time Series Data**
[[paper](https://www.ijcai.org/Proceedings/16/Papers/384.pdf)]
[[official code](https://www.microsoft.com/en-us/research/uploads/prod/2016/06/STMVL-Release.zip)]


## ❖ Other Resources
### `Articles about General Missingness and Imputation`
[blog] [**Data Imputation: An essential yet overlooked problem in machine learning**](https://www.vanderschaar-lab.com/data-imputation-an-essential-yet-overlooked-problem-in-machine-learning/)

[Journal of Big Data] **A survey on missing data in machine learning** 
[[paper](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00516-9)]


### `Repos about General Time Series`
[Transformers in Time Series](https://github.com/qingsongedu/time-series-transformers-review)

[LLMs and Foundation Models for Time Series and Spatio-Temporal Data](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)

[AI for Time Series (AI4TS) Papers, Tutorials, and Surveys](https://github.com/qingsongedu/awesome-AI-for-time-series-papers)

## ❖ Citing This Work
If you find this repository helpful to your work, please kindly star it and cite our benchmark paper, survey paper, and PyPOTS as follows:

```bibtex
@article{du2024tsibench,
title={TSI-Bench: Benchmarking Time Series Imputation},
author={Wenjie Du and Jun Wang and Linglong Qian and Yiyuan Yang and Fanxing Liu and Zepu Wang and Zina Ibrahim and Haoxin Liu and Zhiyuan Zhao and Yingjie Zhou and Wenjia Wang and Kaize Ding and Yuxuan Liang and B. Aditya Prakash and Qingsong Wen},
journal={arXiv preprint arXiv:2406.12747},
year={2024}
}
```

```bibtex
@article{wang2024deep,
title={Deep Learning for Multivariate Time Series Imputation: A Survey},
author={Jun Wang and Wenjie Du and Wei Cao and Keli Zhang and Wenjia Wang and Yuxuan Liang and Qingsong Wen},
journal={arXiv preprint arXiv:2402.04059},
year={2024}
}
```

```bibtex
@article{du2023pypots,
title={{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
author={Wenjie Du},
journal={arXiv preprint arXiv:2305.18811},
year={2023},
}
```

<details>
<summary>🏠 Visits</summary>
<a href="https://github.com/WenjieDu/Awesome_Imputation">
    <img alt="Awesome_Imputation visits" align="left" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWenjieDu%2FAwesome_Imputation&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits%20since%20Jan%202023&edge_flat=false">
</a>
</details>
<br>
