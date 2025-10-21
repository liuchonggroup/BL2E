# 键长能量估算模型
本Repo用于公开PET核药项目的部分技术细节--利用已知配位键长和金属离子估计可能的配位势能ΔE。
调用本Repo中部署好的的onnx/model.onnx，输入已知的平均配位键长和金属离子（'Zr', 'Sc', 'Sr', 'Cu', 'Ga'）即可进行估算。
![workflow](./pictures/Workflow.png)
***图1:** PET核药的筛选和模型的训练工作流*
