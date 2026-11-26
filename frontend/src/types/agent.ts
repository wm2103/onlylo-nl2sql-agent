/**
 * 智能体类型定义
 * 定义问数智能体前端使用的 SSE 事件、流程步骤和聊天消息类型
 */
export type ProgressStatus = "running" | "success" | "error" | "empty" | "timeout" | "degraded";

export type RecallReasonCode =
  | "recall_started"
  | "candidates_found"
  | "no_candidates"
  | "retrieval_error"
  | "retrieval_timeout"
  | "branch_budget_exhausted"
  | "request_budget_exhausted"
  | "keyword_expansion_error"
  | "keyword_expansion_timeout";

export type ProgressEvent = {
  type: "progress";
  step: string;
  status: ProgressStatus;
  reason_code?: RecallReasonCode;
  request_id?: string;
};

export type ResultEvent = {
  type: "result";
  data: unknown;
};

export type ErrorEvent = {
  type: "error";
  message: string;
};

export type TerminalEvent = {
  type: "terminal";
  status: "success" | "blocked" | "error";
  reason_code: string;
};

export type RecallDecisionEvent = {
  type: "recall_decision";
  request_id: string;
  required_sources: string[];
  missing: string[];
  status: "success" | "degraded" | "blocked";
  reason_code: string;
  user_message: string;
};

export type AgentEvent = ProgressEvent | ResultEvent | ErrorEvent | TerminalEvent | RecallDecisionEvent;

export type StepState = {
  step: string;
  status: ProgressStatus;
  updatedAt: number;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  status?: "streaming" | "done" | "error";
  steps?: StepState[];
  result?: unknown;
  error?: string;
};
