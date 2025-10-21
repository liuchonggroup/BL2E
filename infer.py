import os.path as osp

import numpy as np
import onnxruntime as rt
from sklearn.preprocessing import OneHotEncoder

# 创建一个推理会话 (InferenceSession)
onnx_path = osp.join(osp.dirname(__file__), 'onnx', 'BL2Emodel.onnx')
sess_options = rt.SessionOptions()
# 使用 Intel Extension for Scikit-learn 优化后的模型需要注册自定义操作
# sess_options.register_custom_ops_library("libskl2onnx_runtime.so")
sess = rt.InferenceSession(onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"])

onehot_encoder = OneHotEncoder(categories=[np.array(['Zr', 'Sc', 'Sr', 'Cu', 'Ga'])], sparse_output=False)
def infer_model(bl: float, metal_sym: str):
    if metal_sym not in ['Zr', 'Sc', 'Sr', 'Cu', 'Ga']:
        raise ValueError(f'The infer model only supports [Zr, Sc, Sr, Cu, Ga], got {metal_sym}')
    if not (0.7 < bl < 2.1):
        raise ValueError(f'The bond length only allows ranges from 0.7 to 2.1, got {bl}')

    bl = np.array([[bl]], dtype=np.float32)
    sym_vec = onehot_encoder.fit_transform([[metal_sym]])

    x = np.hstack((bl, sym_vec), dtype=np.float32)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    results = sess.run([output_name], {input_name: x})[0]
    print(f"预测能量值为： {results.item()} eV")
    return results


if __name__ == "__main__":
    infer_model(0.75, 'Sr')