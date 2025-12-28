## Orchard

代码仓库包含了各种常用工具。比如，请求 LLM (VLM)，日志和处理图像。

## 规范实验内容

### 实验结果和日志管理

#### 规范命名和目录组织

Tmux 会话命名参考：log_结构_参数_名字

规范日志和记录，构建专门记录实验结果的目录，方便参考。命名 logs/experiments，子目录按照任务特征名、实验名，结构、参数等命名。

使用 Docker 虚拟化，管理每个人的环境。

#### Docker 镜像管理

映射实验结果的目录到共享目录到容器，统一管理输出。

使用 Nvidia 的镜像管理作为基础镜像。比如：
- nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
- nvcr.io/nvidia/pytorch:25.03-py3
