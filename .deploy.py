import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


def draw():
    # You can also use it with a range of values, for example for plotting
    import matplotlib.pyplot as plt

    # Generate a range of 't' values from -2 to 2
    t_range = np.linspace(0.7, 2.1, 500)
    z_range = complex_mapping(t_range)

    modulus= np.abs(z_range)
    scaler = MinMaxScaler((-5, 70))
    y = scaler.fit_transform(np.log(modulus).reshape(-1, 1))
    y = 65 - y

    # Plot the result on the complex plane
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(z_range.real, z_range.imag)
    axs[0].set_title("Plot of g(t) in the Complex Plane")
    axs[0].set_xlabel("Real Part")
    axs[0].set_ylabel("Imaginary Part")
    axs[0].grid(True)

    axs[1].plot(t_range, modulus)
    axs[1].set_title("Plot of t to modules")
    axs[1].set_xlabel("input")
    axs[1].set_ylabel("Modules")
    axs[1].grid(True)

    axs[2].plot(t_range, y)
    axs[2].set_title("Plot of t to modules")
    axs[2].set_xlabel("input")
    axs[2].set_ylabel("Modules")
    axs[2].grid(True)

    fig.show()


def complex_mapping(t: np.ndarray) -> float:
    """"""
    t_squared = np.power(t, 5)

    # Calculate the real part: sin(t^2) * cosh(t)
    real_part = np.sin(t_squared) * np.cosh(t)

    # Calculate the imaginary part: cos(t^2) * sinh(t)
    imag_part = np.cos(t_squared) * np.sinh(t)

    # Return the complex number(s)
    return real_part + 1j * imag_part


def gen_dataset():
    x = np.random.uniform(0.7, 2.1, size=100).reshape(-1, 1)
    y = np.log(np.abs(complex_mapping(x)))

    scaler = MinMaxScaler((-5, 70))
    y = scaler.fit_transform(y)

    return x, -(65 - y) + np.random.normal(size=y.shape)

def train():
    x, y = gen_dataset()
    model = RandomForestRegressor()
    model.fit(x, y)

    xt, yt = gen_dataset()

    print(model.score(xt, yt))
    print(model.predict(xt))

    # 导入 onnx 转换库
    import os.path as osp
    import skl2onnx
    from skl2onnx.common.data_types import FloatTensorType

    # 定义模型的输入类型。ONNX 需要知道输入的形状和数据类型。
    # [None, 1] 表示输入可以有任意数量的行（None），但必须有 1 列。
    initial_type = [('float_input', FloatTensorType([None, 1]))]

    # 执行转换
    onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)

    # 将转换后的模型保存到文件

    onnx_filename = osp.join(osp.dirname(__file__), "onnx", "random_forest_model.onnx")
    with open(onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())

    return onnx_filename


if __name__ == '__main__':
    train()
    # draw()
