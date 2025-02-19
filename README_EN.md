# UV-Segformer
Project Introduction: By employing a feature fusion strategy, we have achieved the best accuracy on our self - made CUGUV Dataset, which covers urban villages in 15 Chinese cities, enhancing the robustness of models in cross - city urban village mapping.

CUGUV Dataset Link: https://doi.org/10.6084/m9.figshare.26198093
![Figure 5](https://github.com/user-attachments/assets/3ad6d348-9adc-4b79-8ef1-d9eee5352ba2)

## Performance Comparison
Taking CUGUV as the benchmark, we compared the performance of UV - Segformer and other semantic segmentation models in urban village identification.
| Models | OA(%) | mIoU(%) | Precision(%) | F1-scores(%) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| UNet | 87.32 | 77.87 | 87.10 | 87.21 |
| DeepLabV3plus | 87.95 | 78.16 | 86.94 | 87.42 |
| PSPNet | 88.75 | 79.98 | 88.32 | 88.53 |
| SWin Transformer | 87.26 | 78.82 | 88.27 | 87.75 |
| KNet | 88.48 | 79.53 | 88.01 | 88.24 |
| Segformer | 90.26 | 82.05 | 89.50 | 89.87 |
| **UV-Segformer** | **92.82** | **86.80** | **92.72** | **92.77** |

![Performance Comparison](https://github.com/user-attachments/assets/7b1b1c30-330d-4d43-85d3-6ff5284f3a5e)


