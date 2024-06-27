from models.MCFNet import MCFNet
import os
import torch
import time

iterations = 100   # 重复计算的轮次
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = MCFNet().eval()

model.cuda()

random_input1 = torch.randn(1, 3, 384, 384).cuda()
random_input2 = torch.randn(1, 3, 384, 384).cuda()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input1,random_input2)

# 测速
times = 0    # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        start = time.time()
        model(random_input1,random_input2)
        times += time.time()- start
        # print(curr_time)

print("Inference time: {:.6f}, FPS: {} ".format(times, 100/times *1))
