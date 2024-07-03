import os
import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group

@ray.remote
def load_input(input_path):
    return np.load(input_path)

@ray.remote
class Worker:
    def __init__(self, weight_path):
        self.weight = np.load(weight_path)

    def relu(self, x):
        input = ray.get(x)
        return np.maximum(0, np.dot(input, self.weight))


        

if __name__ == "__main__":
    # 初始化ray环境
    ray.init(address=f"{os.environ['RAY_CLUSTER_ADDR']}")

    # 初始化ray资源束
    pg = placement_group([{"CPU": 4}, {"CPU": 4}, {"CPU": 4}, {"CPU": 4}], strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    # 加载权重矩阵
    weight_paths = [f'weights/weights_{i}.npy' for i in range(4)]
    actors = [Worker.options(placement_group=pg, placement_group_bundle_index=i).remote(weight_paths[i]) for i in range(4)]

    # 流水线并行
    futures = []
    for i in range(100):
        # 加载输入
        input_path = f'inputs/input_{i}.npy'
        x0 = ray.put(load_input.remote(input_path))
        # 提交task
        y1 = actors[0].relu.remote(x0)
        y2 = actors[1].relu.remote(y1)
        y3 = actors[2].relu.remote(y2)
        y4 = actors[3].relu.remote(y3)
        futures.append(y4)
    
    for i, future in enumerate(futures):
        output = ray.get(future)
        # 保存结果
        output_path = f'outputs/output_{i}.npy'
        np.save(output_path, output)