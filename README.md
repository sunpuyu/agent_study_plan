# 这是一个菜鸟想成为agent开发工程师的梦想记录

## 激活虚拟环境
### 1、uv run
```
uv run python main.py
uv run pytest
uv run ruff check
```
### 2、手动激活
```
source .venv/bin/activate
# 激活后，直接 python / pip，这里的pip就是uv管理的虚拟环境内pip
python main.py
# 退出环境
deactivate
```