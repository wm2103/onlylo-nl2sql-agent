# 文档导航

本项目是面向电商分析场景的 Text-to-SQL 工程原型。阅读顺序：

1. [主 README](../README.md)：项目定位、技术栈和完整最小启动链路。
2. [架构导航](architecture/index.md)：当前正式图与中文备用草稿的区别。
3. [SQL 行范围契约](sql-row-scope.md)：部署方类型声明、字节精确比较、验证范围和限制。
4. [评测说明](../eval/README.md)：离线替身评测、真实服务诊断和证据解释。
5. [基线测试](../tests/README.md)：固定请求、适配器记录和回归入口。
6. [前端配置](../frontend/README.md)：开发代理与 API 地址。

## 运行与评测边界

离线测试通过不意味着真实模型、MySQL、检索服务或端到端链路已验证。
真实服务评测使用独立输出目录，保留失败与未知状态，不把诊断样本的结果当作正式准确率。
本地凭证、运行日志、数据库文件、模型和生成报告不属于发布源码。

## 来源

项目基于 [didilili/shopkeeper-agent](https://github.com/didilili/shopkeeper-agent) 演进，
保留原始 [MIT LICENSE](../LICENSE) 与版权声明。详见 [来源说明](../README.md#source-and-license)。
