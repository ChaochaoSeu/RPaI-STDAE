# Ramp Flow Prediction at Highway Interchanges
This project addresses the challenge of **ramp flow prediction** at highway interchanges, especially in environments where ramp detectors are absent. The framework is built on top of **BasicTS**, a general-purpose time-series forecasting library.  
> ⚙️ BasicTS (by GestaltCogTeam): [https://github.com/GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS) 
> Our paper is at https://arxiv.org/abs/2510.03381
## 📌 Overview

- **Goal**: Predict ramp turning traffic flows using only mainline ETC (Electronic Toll Collection) data.  
- **Challenge**: Ramp sensors are often missing, creating a “real-time data blind spot.” Models must infer ramp dynamics from mainline data alone.  
- **Approach**: Introduce a two-stage model, STDAE (Spatio-Temporal Decoupled Autoencoder), to learn latent representations of ramp behavior from mainline signals, and then feed those embeddings into a downstream predictor (e.g., Graph WaveNet) for final forecasting.  
- **Framework Foundation**: The pipeline is implemented by extending **BasicTS** for data handling, model training, experiment configuration, and evaluation.  

## 🚀 Project Architecture

### 1. Base Framework: BasicTS  
We leverage BasicTS for:

- Modular data loading, normalization, and transformation  
- Experiment management and configuration  
- Training loops, checkpoints, early stopping  
- Metric computation (MAE, RMSE, MAPE, etc.)  
- Model integration (plug-in modules)  

By building on BasicTS, our project focuses on **model innovation and domain-specific logic**, while reusing robust time-series infrastructure.

### 2. Our Model: STDAE + Predictor  
- **Pretraining stage (STDAE)**  
  We use mainline ETC sequences to **reconstruct historical ramp flows** in a proxy task. STDAE comprises two parallel autoencoders:  
  - **SAE** (Spatial Autoencoder)  
  - **TAE** (Temporal Autoencoder)  
  By decoupling spatial and temporal patterns, the model learns the intrinsic mapping between mainline and ramp flows.  
- **Forecasting stage**  
  The learned embeddings from STDAE are fused into the hidden states of the downstream model (e.g., Graph WaveNet) to improve predictive accuracy.

## 📊 Dataset & Preprocessing

- **Data Source**: ETC gantry data from multiple real-world cloverleaf interchanges.  
- **Features**: Ramp turning volume, upstream & downstream gantry volume & speed, mainline width/lane count, ramp lane number, and temporal features (time-of-day, day-of-week).  
- **Sampling Granularity**: Configurable (in our experiments,3-minute intervals, 5-minute intervals, 10-minute intervals).  
- **Data Split**: Train / Val / Test = 17 : 3 : 3 (or as defined in config).  
- **Preprocessing Steps**:  
  1. Load and select target channels  
  2. Add temporal features  
  3. Save processed arrays & adjacency matrices  
  4. Generate experiment description (JSON)  

## 🧪 Experiments & Results

- Evaluated over 3 interchange datasets with multiple sampling granularities  
- Baselines: 13+ state-of-art models (e.g. STGNNs, classical methods)  
- Our model, dubbed **STDAEGWNET**, achieves top performance on MAE, MAPE, RMSE.  
- Notably, even though it relies solely on mainline data, it matches or surpasses models using ramp historical data.

## ✅ Key Contributions & Strengths

- **No reliance on ramp detectors** — only mainline ETC data is required  
- **Cross-modal proxy-task pretraining** to bridge the gap between mainline and ramp domains  
- **Architecture-agnostic embeddings** that can be plugged into different predictors  
- **Demonstrated robustness** under missing data scenarios  
- **Easy integration** built on BasicTS for reproducibility and modularity  

## 📂 Project Structure (example)
├── datasets/
│ ├── QiLin_10min/
│ │ ├── data.dat
│ │ ├── adj_mx.pkl
│ │ └── desc.json
│ └── …
├── baselines
│ ├──models/
│ │ ├── predictor.py
│ │ ├── config.py
│ │ └── …
├── basicts/
│ ├── data/
│ ├── runners/
│ └── ...
├── experiments/
│ ├── evaluate/
│ ├── train/
└── README.md
## 📌 Usage

1. Place your dataset in the `datasets/` folder and process it using `scripts/data_preparation`, configuring the necessary parameters.  
2. Implement your model code in the `baselines/` directory and set up configuration files for each dataset.  
3. Conduct training and testing in the `experiments/` directory.  
