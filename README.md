# 🌿 Plant Disease Classification using MobileNetV2
This project focuses on **classifying plant leaf diseases** using deep learning and transfer learning with the **MobileNetV2** architecture. The model is trained on the **PlantVillage dataset**, which contains images of healthy and diseased plant leaves from multiple species.
---
## 📂 Dataset
**Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Total images: ~38,000  
- Classes: 15 (including healthy and diseased leaves)  
- Categories: Pepper, Potato, Tomato  
- Example classes:  
  - `Pepper__bell___Bacterial_spot`  
  - `Potato___Early_blight`  
  - `Tomato_Leaf_Mold`  
  - `Tomato_healthy`
---
## 🧠 Model Architecture
The model is based on **MobileNetV2**, a lightweight CNN architecture pre-trained on ImageNet. The base layers were frozen initially, then fine-tuned to improve classification performance.
**Model Structure:**
Input (224x224x3)
→ MobileNetV2 (pre-trained, partially frozen)
→ GlobalAveragePooling2D
→ Dropout (rate = 0.3)
→ Dense (15, activation='softmax')
---
## ⚙️ Training Phases
1. **Phase 1 – Frozen Base Training:**  
   - Only the top layers were trained.  
   - Achieved validation accuracy: ~0.89  
2. **Phase 2 – Fine-Tuning:**  
   - Unfroze top MobileNetV2 layers and retrained with a lower learning rate.  
   - Final validation accuracy: **0.98**  
   - Macro F1-score: **0.979**
---
## 📊 Evaluation
- **Accuracy:** 98.0%  
- **Macro Precision:** 0.9797  
- **Macro Recall:** 0.9788  
- **Macro F1:** 0.9790  
**Confusion Matrix (Normalized):**  
Shows strong diagonal dominance, indicating consistent class-level accuracy.
**Example Prediction:**  
Pred: Pepper__bell___Bacterial_spot (100.00%)
True: Pepper__bell___Bacterial_spot
---
## 💾 Model Saving & Inference
- Best model saved as: `mobilenetv2_finetune_best.keras`  
- Inference demo tested on validation images with correct predictions.
---
## 🧰 Tech Stack
| Tool / Library | Purpose |
|:--|:--|
| TensorFlow / Keras | Model building and training |
| Matplotlib, Seaborn | Visualization |
| Scikit-learn | Metrics & confusion matrix |
| KaggleHub | Dataset download |
| Google Colab | Model training environment |
---
## 📈 Results Summary
| Metric | Value |
|:--|:--|
| Validation Accuracy | **0.9803** |
| Macro Precision | **0.9797** |
| Macro Recall | **0.9788** |
| Macro F1-score | **0.9790** |
---
## ✍️ Author
**Name:** Batuhan TUNALI  
**Institution:** Berlin School of Business and Innovation (BSBI)  
**Module:** Practical Skills Assessment (Deep Learning Project)  
**Year:** October 2025  
---
## 📜 License
This project is for **academic and research purposes only.**  
Dataset © Kaggle / PlantVillage, used under public license.
