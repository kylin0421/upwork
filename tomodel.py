from mlp3 import ClassificationModel
import tensorflow as tf
import coremltools as ct


model = ClassificationModel.build_model()
model.load_weights("model_checkpoint.weights.h5")  # 或 model_epoch_0.weights.h5

# 转换为 Core ML 格式
mlmodel = ct.convert(model, source='tensorflow')  # 对于 keras model，用 source='tensorflow' 通常更兼容
mlmodel.save("converted_model.mlmodel")
