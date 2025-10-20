import os.path as osp

import numpy as np
import onnxruntime as rt

def run_onnx_inference(bl_length: float):
    """
    加载 ONNX 模型并用它来进行推理。
    """
    bl_length = np.array([[bl_length]], dtype=np.float32)
    onnx_path = osp.join(osp.dirname(__file__), 'onnx', 'BL2Emodel.onnx')
    print("\n--- 步骤 3: 使用 ONNX Runtime 进行推理 ---")


    # 创建一个推理会话 (InferenceSession)
    sess_options = rt.SessionOptions()
    # 使用 Intel Extension for Scikit-learn 优化后的模型需要注册自定义操作
    # sess_options.register_custom_ops_library("libskl2onnx_runtime.so")
    sess = rt.InferenceSession(onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"])

    # 获取模型的输入和输出名称
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"ONNX 模型输入名: '{input_name}'")
    print(f"ONNX 模型输出名: '{output_name}'")

    # 使用 ONNX 模型进行预测
    # 注意：输入必须是一个字典，键是输入名，值是 numpy 数组
    onnx_predictions = sess.run([output_name], {input_name: bl_length})[0]

    print(f"预测能量值为： {onnx_predictions.item()} eV")

    return onnx_predictions.item()


if __name__ == "__main__":
    print(run_onnx_inference(1.0))