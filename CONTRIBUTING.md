# 贡献指南

感谢您对 DCT-SVD 数字水印项目的关注！我们欢迎所有形式的贡献。

## 📋 如何贡献

### 报告问题 (Bug Reports)

如果您发现了任何bug或有功能建议，请：

1. 使用Issue模板创建新的issue
2. 详细描述问题
3. 提供重现步骤
4. 包含相关的错误信息或截图

### 功能请求 (Feature Requests)

我们欢迎新的功能建议，特别是在以下方面：

- 新的攻击类型实现
- 额外的评估指标
- 可视化改进
- 性能优化
- 其他水印算法的实现

### 代码贡献 (Code Contributions)

#### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/your-username/watermarking_dct_svd.git
cd watermarking_dct_svd

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 代码规范

1. **代码风格**：遵循PEP 8规范
2. **函数文档**：所有函数必须有docstring
3. **类型提示**：尽可能使用类型注解
4. **测试**：新功能应包含测试用例

#### 提交Pull Request

1. Fork项目到您的账户
2. 创建特性分支：`git checkout -b feature/your-feature-name`
3. 提交更改：`git commit -m "Add your feature description"`
4. 推送到您的fork：`git push origin feature/your-feature-name`
5. 创建Pull Request

### 文档贡献

文档改进同样重要！您可以：

- 修复拼写错误和语法问题
- 改进文档清晰度
- 添加更多示例
- 翻译文档到其他语言

## 🏗️ 项目架构

主要模块：

- `main.py` - 主程序入口
- `watermark_algorithm.py` - DCT-SVD水印算法
- `baseline_dct.py` - 纯DCT基线算法
- `attacks.py` - 攻击方法实现
- `metrics.py` - 评估指标
- `visualization.py` - 可视化模块
- `config.py` - 配置文件
- `utils.py` - 工具函数

## 🧪 测试

运行测试（如果有）：
```bash
python -m pytest tests/
```

## 📝 提交信息格式

使用以下格式提交信息：

```
类型(范围): 简短描述

详细描述（可选）

Closes #123
```

类型：
- `feat` - 新功能
- `fix` - 修复bug
- `docs` - 文档更新
- `style` - 代码格式
- `refactor` - 重构
- `test` - 测试
- `chore` - 构建/工具更改

## 🤝 行为准则

请保持友善和专业的交流。对于任何形式的骚扰、不适当或其他不可接受的行为，维护者有权删除相关内容并采取措施。

## 📄 许可证

通过贡献代码，您同意您的贡献将在项目的MIT许可证下发布。