<!--
  前端项目说明文档
  记录启动方式、代理配置和部署时的环境变量
-->

# onlylo-nl2sql-agent 前端

运行 `pnpm test` 执行 SSE 消费回归（使用 Node.js 内置测试器与现有 TypeScript 依赖），`pnpm build` 执行类型检查和生产构建。测试使用替身事件驱动真实 App 回调，不是实际模型/API 联调证明。

匹配本项目后端的问数聊天前端，技术栈为 React + Vite + Tailwind CSS + pnpm。

## 启动

```bash
cd frontend
pnpm install
pnpm dev
```

默认开发代理会把 `/api` 转发到 `http://127.0.0.1:8000`，对应后端的 `POST /api/query` SSE 接口。

如需修改后端地址：

```bash
cp .env.example .env
```

然后调整：

```bash
VITE_DEV_PROXY_TARGET=http://127.0.0.1:8000
```

推荐通过同源反向代理将 `/api` 转发到后端。如果前端直接跨域访问后端，可设置：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

仅设置此变量不足以启用跨域：当前后端未配置 CORS，JSON POST 会触发预检。直接跨域部署时须另行配置后端或网关 CORS，显式允许前端来源、POST、OPTIONS 与 Content-Type；不要将任意来源开放作为默认。Vite 开发代理不受此跨域限制。Vite 环境变量在构建时注入，修改后需重新启动开发服务或重新构建。
