# Test Results 文件夹

这个文件夹用于存放 OpenManus 的测试结果和生成的文件。

## 文件夹结构

```
test_results/
├── README.md           # 本说明文件
├── portfolio.html      # 作品集网站测试结果
└── ...                # 其他测试结果文件
```

## 使用方法

1. **运行测试**：
   ```bash
   python main.py --prompt "你的提示词"
   ```

2. **移动结果**：
   ```bash
   # 将生成的文件移动到测试结果文件夹
   mv workspace/生成的文件名 test_results/
   ```

3. **查看结果**：
   ```bash
   # 查看所有测试结果
   ls -la test_results/

   # 查看特定文件
   cat test_results/文件名
   ```

## 文件命名建议

为了便于管理，建议使用以下命名格式：
- `YYYY-MM-DD_测试描述_结果类型.扩展名`
- 例如：`2024-08-02_portfolio_website.html`

## 注意事项

- 每次运行测试前，建议清理 workspace 文件夹
- 重要的测试结果请及时移动到 test_results 文件夹
- 可以创建子文件夹来组织不同类型的测试结果
