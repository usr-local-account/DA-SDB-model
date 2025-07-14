# DA-SDB-model
# DA-SDB: domain adaptation-based deep learning model for satellite-derived bathymetry

[![Paper](https://img.shields.io/badge/Paper-ISPRS-blue)](https://www.isprs.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Sentinel--2-green)](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
[![ICESat-2](https://img.shields.io/badge/ICESat--2-NASA-red)](https://icesat-2.gsfc.nasa.gov/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

DA-SDB (domain adaptation-based deep learning model for satellite-derived bathymetry) is a deep learning model designed to address the generalization challenges in satellite-derived bathymetry (SDB) across spatial and temporal domains. Traditional empirical models exhibit limited transferability due to domain shifts caused by variations in water quality, substrate types, and atmospheric conditions. Our proposed DA-SDB model leverages domain adaptation techniques to significantly enhance generalization capability, providing an efficient and cost-effective solution for bathymetric mapping in remote areas and long-term monitoring of critical regions.

This research conducted extensive spatial and temporal transfer experiments across five diverse study areas (Dongsha Atoll, Bimini Island, South Warden Reef, Hadrah Island, and Mubarraz and Bu Tinah Islands), demonstrating the significant performance advantages of the DA-SDB model compared to existing methods, with stable performance in cross-regional and cross-temporal application scenarios.

## Key Features

- **Robust Generalization**: Employs domain adaptation to effectively apply models trained on source domains to target domains without requiring labeled target data
- **End-to-End Architecture**: Integrates three components—feature extractor, bathymetry predictor, and domain aligner
- **Innovative Feature Extraction Modules**:
  - **Pyramid-Like Block (PLB)**: Multi-channel neural network structure that effectively extracts multi-dimensional features
  - **Physical-Assisted Block (PAB)**: Utilizes spectral slope information to enhance the model's ability to extract wavelength-related features
- **Robust Domain Alignment**: Aligns the pseudo-inverse Gram matrices to mitigate distribution differences between source and target domains
- **Direct Use of TOA Reflectance**: Operates without complex atmospheric correction, simplifying the data processing pipeline

## Datasets

This research utilizes the following data sources:

- **Sentinel-2 L1C Data**: Provides multispectral remote sensing imagery
- **ICESat-2 ATL03 Data**: Provides high-precision bathymetric reference data
- **Study Areas Covering Five Locations**:
  - Dongsha Atoll (South China Sea)
  - Bimini Island (Great Bahama Bank)
  - South Warden Reef (Great Barrier Reef)
  - Hadrah Island (Southern Red Sea)
  - Mubarraz and Bu Tinah Islands (Persian Gulf)

## Installation

### Requirements

```bash
# Install required dependencies
pip install -r requirements.txt
```

### Quick Start

1. **Data Preparation**

```python
# Prepare dataset
python CustomDataset.py
```

2. **Model Training**

```python
# Train the model
python TransferDepthMain.py -train_source_root [source_domain_data_path] -train_target_root [target_domain_data_path] -val_root [validation_data_path]
```

3. **Model Evaluation**

```python
# Load trained model and evaluate
python 0612_load_model.py
```

4. **Ablation Studies**

```python
# Conduct ablation experiments to verify the contribution of different modules
python AlbrationMainpy.py
```

### Parameter Settings

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| -train_source_root | Source domain training data path | - |
| -train_target_root | Target domain training data path | - |
| -val_root | Validation data path | - |
| -input_features | Input feature dimensions | 14 |
| -batch_size | Batch size | 1024 |
| -epochs | Total training epochs | - |
| -domain_epochs | Domain adaptation training epochs | 500 |
| -hidden_dim | Hidden layer dimensions | 64 |
| --trans_loss | Transfer loss function type | gram |

## Experimental Results

This study conducted spatial and temporal transfer experiments, showing that the DA-SDB model achieved significant performance improvements compared to baseline methods (MBLA, RF, DNN):

- **Spatial Transfer**: Achieved the best results in 6 out of 6 spatial transfer experiments, reducing average RMSE and MAPE by 0.27m and 21.51%, respectively
- **Temporal Transfer**: Achieved the best results in 5 out of 6 temporal transfer experiments, with RMSE as low as 0.37m in Bimini Island and 0.86m in Hadrah Island
- **Model Stability**: Achieved the best results in 11 out of 12 experiments, demonstrating excellent stability and generalization capabilities
- **Depth Adaptability**: Maintained good accuracy across different depth ranges, particularly in regions where traditional methods perform poorly


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please contact: huanxie@tongji.edu.cn or cdliu@tongji.edu.cn
