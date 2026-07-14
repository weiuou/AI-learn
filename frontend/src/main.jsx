import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./styles.css";

const EVENT_TYPES = [
  "run_started", "model_call", "agent_message", "tool_call", "tool_output",
  "tool_result", "file_changed", "error", "run_completed", "run_cancelled",
  "files_applied", "files_discarded", "context_compressed", "loop_detected",
  "context_pack_built", "task_state_updated", "checkpoint_saved",
];

const EVENT_META = {
  run_started: ["运行开始", "play", "blue"],
  model_call: ["模型调用", "sparkles", "violet"],
  agent_message: ["Agent 消息", "bot", "blue"],
  tool_call: ["工具调用", "wrench", "violet"],
  tool_output: ["实时输出", "terminal", "slate"],
  tool_result: ["工具结果", "check", "green"],
  file_changed: ["文件变更", "fileEdit", "amber"],
  error: ["执行错误", "alert", "red"],
  run_completed: ["运行完成", "check", "blue"],
  run_cancelled: ["运行取消", "stop", "red"],
  files_applied: ["变更已应用", "check", "green"],
  files_discarded: ["变更已丢弃", "trash", "slate"],
  context_compressed: ["上下文压缩", "layers", "slate"],
  loop_detected: ["检测到循环", "alert", "red"],
  context_pack_built: ["上下文已准备", "layers", "slate"],
  task_state_updated: ["任务状态已保存", "save", "slate"],
  checkpoint_saved: ["检查点已保存", "save", "slate"],
};

const STATUS_META = {
  created: ["准备中", "neutral"],
  running: ["运行中", "running"],
  waiting_user: ["待审核", "waiting"],
  completed: ["已完成", "success"],
  failed: ["失败", "danger"],
  cancelled: ["已取消", "danger"],
  idle: ["空闲", "neutral"],
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try { message = (await response.json()).detail || message; } catch { /* noop */ }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function Icon({ name, size = 18, strokeWidth = 1.8 }) {
  const paths = {
    workspace: <><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M8 4V2m8 2V2M8 9h8m-4-4v8"/></>,
    history: <><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5m4-1v5l3 2"/></>,
    files: <><path d="M4 4h6l2 2h8v14H4z"/><path d="M4 9h16"/></>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-1.6v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"/></>,
    plus: <path d="M12 5v14M5 12h14"/>,
    search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    file: <><path d="M6 2h8l4 4v16H6z"/><path d="M14 2v5h5"/></>,
    folder: <path d="M3 6h7l2 2h9v11H3z"/>,
    play: <path d="m8 5 11 7-11 7Z"/>,
    stop: <rect x="6" y="6" width="12" height="12" rx="2"/>,
    send: <><path d="m22 2-8 20-4-8-8-4Z"/><path d="M22 2 10 14"/></>,
    bot: <><rect x="4" y="7" width="16" height="13" rx="4"/><path d="M12 3v4M8 12h.01M16 12h.01M8 16h8"/></>,
    wrench: <><path d="M14.7 6.3a4 4 0 0 0-5-5L12 3.6 9.6 6 7.3 3.7a4 4 0 0 0 5 5L5 16l3 3 7.3-7.3a4 4 0 0 0 5-5L18 9l-2.4-2.4 2.3-2.3a4 4 0 0 0-3.2 2Z"/></>,
    terminal: <><path d="m5 7 4 4-4 4m7 0h7"/><rect x="2" y="3" width="20" height="18" rx="3"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    alert: <><path d="M12 3 2.5 20h19Z"/><path d="M12 9v4m0 3h.01"/></>,
    fileEdit: <><path d="M6 2h8l4 4v7M14 2v5h5"/><path d="m14 19 5-5 2 2-5 5-3 1Z"/></>,
    trash: <><path d="M4 7h16M9 7V4h6v3m3 0-1 14H7L6 7m4 4v6m4-6v6"/></>,
    layers: <><path d="m12 2 9 5-9 5-9-5Z"/><path d="m3 12 9 5 9-5M3 17l9 5 9-5"/></>,
    save: <><path d="M4 3h13l3 3v15H4z"/><path d="M8 3v6h8V3M8 21v-7h8v7"/></>,
    sparkles: <><path d="m12 3 1.2 3.8L17 8l-3.8 1.2L12 13l-1.2-3.8L7 8l3.8-1.2ZM5 14l.8 2.2L8 17l-2.2.8L5 20l-.8-2.2L2 17l2.2-.8Z"/></>,
    logout: <><path d="M10 4H4v16h6m5-4 4-4-4-4m4 4H9"/></>,
    close: <path d="M6 6l12 12M18 6 6 18"/>,
    refresh: <><path d="M20 7v5h-5"/><path d="M18.2 17A8 8 0 1 1 20 12"/></>,
    copy: <><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/></>,
    thumbsUp: <><path d="M7 10v11H3V10zm0 10h10a2 2 0 0 0 2-1.6l1.6-7A2 2 0 0 0 18.7 9H14l.8-4a2 2 0 0 0-3.5-1.7L7 10"/></>,
    thumbsDown: <><path d="M7 14V3H3v11zm0-10h10a2 2 0 0 1 2 1.6l1.6 7a2 2 0 0 1-1.9 2.4H14l.8 4a2 2 0 0 1-3.5 1.7L7 14"/></>,
  };
  return <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.file}</svg>;
}

function Brand({ compact = false }) {
  return <div className={`brand ${compact ? "brand-compact" : ""}`}>
    <span className="brand-gem"><i /><b /></span>
    <strong>Agent Workspace</strong>
  </div>;
}

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function submit(event) {
    event.preventDefault(); setBusy(true); setError("");
    try {
      const user = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ username, password }) });
      onLogin(user);
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  return <div className="login-page">
    <header className="login-header"><Brand /><span className="beta-badge"><i /> 公测中 <b>Beta</b></span></header>
    <main className="login-content">
      <section className="login-story">
        <p className="kicker">BUILD WITH CONFIDENCE</p>
        <h1>让 Agent 在独立<br />工作区完成真实任务</h1>
        <p className="login-lead">从任务输入到文件审核，每一步都清晰、隔离、可恢复。</p>
        <div className="benefit-list">
          <div><span><Icon name="workspace" /></span><p><strong>独立工作区</strong><small>每个项目拥有隔离文件空间，互不干扰</small></p></div>
          <div><span><Icon name="sparkles" /></span><p><strong>自主执行</strong><small>Agent 读取项目、调用工具并产出真实结果</small></p></div>
          <div><span><Icon name="check" /></span><p><strong>安全可控</strong><small>完整 Trace 与 Diff 审核，确认后才写回</small></p></div>
        </div>
        <div className="workspace-illustration" aria-hidden="true">
          <div className="code-window back"><i /><i /><i /><span /></div>
          <div className="code-window front"><i /><i /><i /><span /><span /><span /></div>
          <div className="bot-orb"><b /><b /></div>
          <div className="platform"><span /></div>
        </div>
      </section>
      <section className="login-card">
        <div className="login-tabs"><button className="active">登录</button><span>邀请账号</span></div>
        <div className="invite-note"><Icon name="alert" size={17} /> 公测期间仅支持邀请账号登录</div>
        <form onSubmit={submit}>
          <label>账号<div className="input-wrap"><Icon name="bot" size={17} /><input value={username} onChange={e => setUsername(e.target.value)} placeholder="请输入邀请账号" autoComplete="username" required /></div></label>
          <label>密码<div className="input-wrap"><Icon name="save" size={17} /><input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="请输入密码" autoComplete="current-password" required /></div></label>
          {error && <p className="form-error"><Icon name="alert" size={15} />{error}</p>}
          <button className="primary login-submit" disabled={busy}>{busy ? "登录中…" : "登录"}</button>
        </form>
        <div className="login-security"><Icon name="check" size={15} /> 登录状态使用安全 Cookie 保存</div>
      </section>
    </main>
    <footer className="login-footer">Agent Workspace · 独立、可审查、可恢复的 Agent 执行环境</footer>
  </div>;
}

function ProductNav({ runCount, fileCount }) {
  const scrollTo = selector => document.querySelector(selector)?.scrollIntoView({ behavior: "smooth", block: "start" });
  return <nav className="product-nav" aria-label="主导航">
    <button className="active"><Icon name="workspace" /><span>工作区</span></button>
    <button onClick={() => scrollTo(".run-select")}><Icon name="history" /><span>运行记录</span><small>{runCount || ""}</small></button>
    <button onClick={() => scrollTo(".files-section")}><Icon name="files" /><span>文件</span><small>{fileCount || ""}</small></button>
    <div className="nav-spacer" />
    <div className="isolation-note"><Icon name="layers" /><strong>Workspace 隔离</strong><p>文件、Run 与执行环境彼此独立。</p></div>
  </nav>;
}

function WorkspaceSidebar({ workspaces, selectedId, onSelect, onCreate, onDelete, files, selectedFile, onOpenFile, busy }) {
  const [name, setName] = useState("");
  const [query, setQuery] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const filtered = workspaces.filter(item => item.name.toLowerCase().includes(query.trim().toLowerCase()));
  async function create(event) {
    event.preventDefault();
    if (!name.trim()) return;
    await onCreate(name.trim()); setName(""); setShowCreate(false);
  }
  return <aside className="workspace-sidebar">
    <section className="workspace-section">
      <div className="section-heading"><strong>工作区</strong><button className="icon-button labeled" onClick={() => setShowCreate(value => !value)}><Icon name="plus" size={16} /> 新建</button></div>
      {showCreate && <form className="inline-create" onSubmit={create}><input autoFocus placeholder="Workspace 名称" maxLength={120} value={name} onChange={e => setName(e.target.value)} /><button className="primary">创建</button></form>}
      <div className="search-box"><Icon name="search" size={17} /><input value={query} onChange={e => setQuery(e.target.value)} placeholder="搜索工作区" /></div>
      <div className="workspace-list">
        {filtered.map((ws, index) => <button key={ws.id} className={`workspace-item ${selectedId === ws.id ? "active" : ""}`} onClick={() => onSelect(ws.id)}>
          <span className={`workspace-avatar tone-${index % 4}`}><Icon name={index % 2 ? "sparkles" : "bot"} size={18} /></span>
          <span><strong>{ws.name}</strong><small>{selectedId === ws.id && busy ? "Agent 正在执行" : `更新于 ${relativeTime(ws.updatedAt)}`}</small></span>
          {selectedId === ws.id && <i className={`presence-dot ${busy ? "busy" : ""}`} />}
        </button>)}
        {!filtered.length && <div className="sidebar-empty">没有匹配的工作区</div>}
      </div>
    </section>
    <section className="files-section">
      <div className="section-heading"><strong>文件</strong><span className="count-badge">{files.filter(file => file.type === "file").length}</span></div>
      <FileTree files={files} selected={selectedFile} onOpen={onOpenFile} />
    </section>
    {selectedId && <button className="delete-workspace" onClick={() => onDelete(selectedId)}><Icon name="trash" size={15} /> 删除当前 Workspace</button>}
  </aside>;
}

function FileTree({ files, selected, onOpen }) {
  if (!files.length) return <div className="sidebar-empty files-empty">暂无文件<br /><small>运行 Agent 创建第一个文件</small></div>;
  return <div className="file-tree">{files.map(file => {
    const depth = file.path.split("/").length - 1;
    return <button key={file.path} className={selected === file.path ? "selected" : ""} disabled={file.type === "directory"} onClick={() => file.type === "file" && onOpen(file.path)} style={{ paddingLeft: `${14 + depth * 14}px` }}>
      {file.type === "directory" ? <Icon name="folder" size={16} /> : <Icon name="file" size={15} />}<span>{file.name}</span>
    </button>;
  })}</div>;
}

function MarkdownView({ children, compact = false }) {
  return <div className={`markdown-body ${compact ? "compact" : ""}`}>
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: props => <a {...props} target="_blank" rel="noreferrer" />,
      }}
    >{String(children || "")}</ReactMarkdown>
  </div>;
}

function DisplayValue({ value }) {
  if (value == null || value === "") return <span className="result-empty">无返回内容</span>;
  if (typeof value === "string") return <pre className="result-text">{value}</pre>;
  if (typeof value === "number" || typeof value === "boolean") return <strong className="result-scalar">{String(value)}</strong>;
  if (Array.isArray(value)) return <ul className="result-list">{value.map((item, index) => <li key={item?.path || item?.name || index}>
    {typeof item === "object" && item ? <><Icon name={item.type === "directory" ? "folder" : "file"} size={14} /><span>{item.path || item.name || `结果 ${index + 1}`}</span>{item.size != null && <small>{formatBytes(item.size)}</small>}</> : <span>{String(item)}</span>}
  </li>)}</ul>;
  return <dl className="result-fields">{Object.entries(value).filter(([, item]) => item != null).map(([key, item]) => <div key={key}><dt>{friendlyField(key)}</dt><dd>{typeof item === "object" ? <DisplayValue value={item} /> : String(item)}</dd></div>)}</dl>;
}

function ToolCallView({ payload }) {
  const name = payload["tool_call.name"] || payload.toolName || "tool";
  const args = payload["tool_call.arguments"] || payload.arguments || {};
  return <div className="tool-event-body">
    <div className="tool-summary"><span>{friendlyTool(name)}</span><code>{name}</code></div>
    {!!Object.keys(args).length && <dl className="tool-arguments">{Object.entries(args).map(([key, value]) => <div key={key}><dt>{friendlyField(key)}</dt><dd>{key === "content" ? `${String(value).length} 个字符` : String(value)}</dd></div>)}</dl>}
  </div>;
}

function ToolResultView({ payload }) {
  const observation = parsePossibleJson(payload.observation);
  const failed = observation && typeof observation === "object" && observation.ok === false;
  const value = observation && typeof observation === "object" && "result" in observation ? observation.result : observation;
  const message = observation && typeof observation === "object" ? observation.message : null;
  return <div className={`tool-result-view ${failed ? "failed" : ""}`}>
    <div className="result-status"><Icon name={failed ? "alert" : "check"} size={15} />{failed ? (message || "工具执行失败") : "工具执行完成"}</div>
    {!failed && <DisplayValue value={value} />}
  </div>;
}

function SmartEventBody({ event, payload }) {
  const content = stripThinking(String(payload.content || payload.answer || payload.message || payload.data || ""));
  if (event.type === "agent_message") return <MarkdownView compact>{content || "Agent 正在规划下一步…"}</MarkdownView>;
  if (event.type === "tool_call") return <ToolCallView payload={payload} />;
  if (event.type === "tool_result") return <ToolResultView payload={payload} />;
  if (event.type === "tool_output") return <pre className="console-output">{String(payload.data || payload.output || "命令正在执行…")}</pre>;
  if (event.type === "run_started") return <p className="event-message">开始处理任务：<strong>{payload.user_goal || payload.task || "当前任务"}</strong></p>;
  if (event.type === "file_changed") return <p className="file-change-message"><code>{payload.path || "文件"}</code><span>{friendlyChange(payload.changeType)}</span></p>;
  if (event.type === "run_completed") return <p className="event-message">{payload.status === "waiting_user" ? `执行完成，${payload.changedFiles?.length || 0} 个文件等待审核。` : "本次运行已完成。"}</p>;
  if (event.type === "run_cancelled") return <p className="event-message">本次运行已取消。</p>;
  if (event.type === "files_applied") return <p className="event-message">文件变更已应用到 Workspace。</p>;
  if (event.type === "files_discarded") return <p className="event-message">文件变更已丢弃，Workspace 保持不变。</p>;
  if (event.type === "error") return <p className="event-message error-message">{payload.message || payload.error || "执行过程中发生错误。"}</p>;
  return <p className="event-message">{content || "事件已记录。"}</p>;
}

function EventCard({ event, traceMode = false }) {
  const payload = event.payload || {};
  const [label, icon, tone] = EVENT_META[event.type] || [event.type, "layers", "slate"];
  return <article className={`event-row tone-${tone}`}>
    <div className="event-rail"><span><Icon name={icon} size={16} /></span></div>
    <div className="event-card">
      <header><strong>{label}</strong><time>{formatClock(event.createdAt)}</time></header>
      {traceMode ? <pre className="trace-json">{JSON.stringify(payload, null, 2)}</pre> : <SmartEventBody event={event} payload={payload} />}
      {event.step != null && <small className="step-label">STEP {event.step}</small>}
    </div>
  </article>;
}

function Composer({ task, setTask, onSubmit, disabled, waiting }) {
  return <form className="task-composer" onSubmit={onSubmit}>
    <textarea value={task} onChange={e => setTask(e.target.value)} placeholder={waiting ? "请先在右侧审核本次文件变更…" : "描述你希望 Agent 在当前 Workspace 中完成的任务…"} disabled={disabled || waiting} rows={3} />
    <div className="composer-toolbar"><span><Icon name="layers" size={16} /> Agent 仅能访问当前 Workspace</span><button className="primary send-button" disabled={!task.trim() || disabled || waiting}><Icon name="send" size={17} />发送</button></div>
  </form>;
}

function RunInspector({ run, diff, selectedFile, fileContent, onCloseFile, onApply, onDiscard, onSaveFeedback }) {
  const [rating, setRating] = useState(run?.feedback?.rating || "up");
  const [comment, setComment] = useState(run?.feedback?.comment || "");
  useEffect(() => { setRating(run?.feedback?.rating || "up"); setComment(run?.feedback?.comment || ""); }, [run?.id, run?.feedback?.updatedAt]);
  const status = STATUS_META[run?.status || "idle"];
  const duration = run?.durationMs != null ? formatDuration(run.durationMs) : "—";
  return <aside className="run-inspector">
    <div className="inspector-heading"><div><span>本次运行</span>{run && <small>ID: {shortId(run.id)}</small>}</div>{run && <button className="icon-button" title="复制 Run ID" onClick={() => navigator.clipboard?.writeText(run.id)}><Icon name="copy" size={16} /></button>}</div>
    {!run ? <div className="inspector-empty"><span><Icon name="history" size={24} /></span><strong>还没有运行记录</strong><p>发送任务后，这里会显示状态、文件变更和最终结果。</p></div> : <>
      <section className="inspector-card overview-card">
        <div className="card-title"><strong>运行概览</strong><span className={`status-pill ${status[1]}`}><i />{status[0]}</span></div>
        <dl>
          <div><dt>开始时间</dt><dd>{formatDateTime(run.startedAt || run.createdAt)}</dd></div>
          <div><dt>运行耗时</dt><dd>{duration}</dd></div>
          <div><dt>模型调用</dt><dd>{run.modelCalls ?? 0}</dd></div>
          <div><dt>工具调用</dt><dd>{run.toolCalls ?? 0}</dd></div>
        </dl>
      </section>
      {!!diff.length && <section className="inspector-card changes-card">
        <div className="card-title"><strong>修改文件 <span>({diff.length})</span></strong></div>
        <div className="change-list">{diff.map(item => {
          const counts = countDiff(item.diff);
          return <details key={item.path}><summary><Icon name="file" size={15} /><span>{item.path}</span><b>+{counts.added}</b><em>-{counts.removed}</em><Icon name="chevron" size={14} /></summary><pre>{item.diff || "无文本 Diff"}</pre></details>;
        })}</div>
        {run.status === "waiting_user" && <div className="review-actions"><button className="secondary danger" onClick={onDiscard}><Icon name="trash" size={15} />丢弃</button><button className="primary" onClick={onApply}><Icon name="check" size={15} />应用变更</button></div>}
      </section>}
      {(selectedFile || run.finalResult) && <section className="inspector-card output-card">
        <div className="card-title"><strong>{selectedFile ? "文件预览" : "最终结果"}</strong>{selectedFile && <button className="icon-button" onClick={onCloseFile}><Icon name="close" size={14} /></button>}</div>
        {selectedFile && <code className="file-name">{selectedFile}</code>}
        {selectedFile ? <pre className="file-preview-content">{fileContent}</pre> : <MarkdownView>{run.finalResult}</MarkdownView>}
      </section>}
      {run.error && <section className="inspector-card error-card"><div className="card-title"><strong>错误信息</strong></div><p>{run.error}</p></section>}
      {run.status === "completed" && <section className="inspector-card feedback-card">
        <div className="card-title"><strong>这次 Run 完成任务了吗？</strong></div>
        <div className="rating"><button className={rating === "up" ? "active" : ""} onClick={() => setRating("up")}><Icon name="thumbsUp" size={17} />完成了</button><button className={rating === "down" ? "active down" : ""} onClick={() => setRating("down")}><Icon name="thumbsDown" size={17} />没有完成</button></div>
        <textarea value={comment} onChange={e => setComment(e.target.value)} placeholder="补充反馈说明（可选）" />
        <button className="secondary save-feedback" onClick={() => onSaveFeedback(rating, comment)}>保存反馈</button>
      </section>}
    </>}
  </aside>;
}

function WorkspaceView({ workspace, workspaces, onSelectWorkspace, onCreateWorkspace, onDeleteWorkspace, user, onLogout }) {
  const [files, setFiles] = useState([]);
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState(null);
  const [events, setEvents] = useState([]);
  const [diff, setDiff] = useState([]);
  const [task, setTask] = useState("");
  const [selectedFile, setSelectedFile] = useState("");
  const [fileContent, setFileContent] = useState("");
  const [error, setError] = useState("");
  const [viewMode, setViewMode] = useState("agent");
  const endRef = useRef(null);
  const active = run && ["created", "running"].includes(run.status);
  const waiting = run?.status === "waiting_user";
  const status = STATUS_META[run?.status || "idle"];
  const smartEvents = useMemo(() => dedupeAgentEvents(events.filter(event => [
    "run_started", "agent_message", "tool_call", "tool_output", "tool_result", "file_changed",
    "error", "run_completed", "run_cancelled", "files_applied", "files_discarded",
  ].includes(event.type))), [events]);
  const visibleEvents = viewMode === "trace" ? events : smartEvents;

  async function refreshFiles() { setFiles(await api(`/api/workspaces/${workspace.id}/files`)); }
  async function refreshRun(runId) {
    if (!runId) return;
    const next = await api(`/api/runs/${runId}`); setRun(next);
    if (next.changedFiles?.length) setDiff(await api(`/api/runs/${runId}/diff`));
    return next;
  }
  async function selectRun(runId) {
    try {
      setEvents([]); setDiff([]); setSelectedFile("");
      const next = await api(`/api/runs/${runId}`); setRun(next);
      if (next.changedFiles?.length) setDiff(await api(`/api/runs/${runId}/diff`));
    } catch (err) { setError(err.message); }
  }
  async function load() {
    setError(""); setEvents([]); setDiff([]); setSelectedFile(""); setFileContent("");
    try {
      const [nextFiles, nextRuns] = await Promise.all([api(`/api/workspaces/${workspace.id}/files`), api(`/api/workspaces/${workspace.id}/runs`)]);
      setFiles(nextFiles); setRuns(nextRuns);
      const latest = nextRuns[0] || null; setRun(latest);
      if (latest?.changedFiles?.length) setDiff(await api(`/api/runs/${latest.id}/diff`));
    } catch (err) { setError(err.message); }
  }
  useEffect(() => { load(); }, [workspace.id]);
  useEffect(() => {
    if (!run?.id) return;
    setEvents([]);
    const source = new EventSource(`/api/runs/${run.id}/events`);
    const receive = event => {
      try {
        const parsed = JSON.parse(event.data);
        setEvents(current => current.some(item => item.sequence === parsed.sequence) ? current : [...current, parsed]);
        if (["file_changed", "run_completed", "run_cancelled", "error", "files_applied", "files_discarded"].includes(parsed.type)) {
          refreshRun(run.id).then(next => {
            if (next && ["waiting_user", "completed", "failed", "cancelled"].includes(next.status)) source.close();
          }).catch(() => {});
          refreshFiles().catch(() => {});
        }
        if (["run_completed", "run_cancelled"].includes(parsed.type)) source.close();
      } catch { /* noop */ }
    };
    EVENT_TYPES.forEach(type => source.addEventListener(type, receive));
    source.onerror = () => {
      refreshRun(run.id).then(next => {
        if (next && ["waiting_user", "completed", "failed", "cancelled"].includes(next.status)) source.close();
      }).catch(() => {});
    };
    return () => source.close();
  }, [run?.id]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }); }, [events.length]);

  async function startRun(event) {
    event.preventDefault(); if (!task.trim() || active || waiting) return;
    try {
      setError("");
      const created = await api(`/api/workspaces/${workspace.id}/runs`, { method: "POST", body: JSON.stringify({ workspaceId: workspace.id, task: task.trim() }) });
      setTask(""); setRun(created); setRuns(current => [created, ...current]); setEvents([]); setDiff([]); setSelectedFile("");
    } catch (err) { setError(err.message); }
  }
  async function openFile(path) {
    try {
      const data = await api(`/api/workspaces/${workspace.id}/files/${path.split("/").map(encodeURIComponent).join("/")}`);
      setSelectedFile(path); setFileContent(data.content);
    } catch (err) { setError(err.message); }
  }
  async function review(action) {
    try {
      const next = await api(`/api/runs/${run.id}/${action}`, { method: "POST" }); setRun(next);
      setRuns(current => current.map(item => item.id === next.id ? next : item)); await refreshFiles();
    } catch (err) { setError(err.message); }
  }
  async function saveFeedback(rating, comment) {
    try {
      const saved = await api(`/api/runs/${run.id}/feedback`, { method: "PUT", body: JSON.stringify({ rating, comment }) });
      setRun(current => ({ ...current, feedback: saved }));
    } catch (err) { setError(err.message); }
  }
  async function stopRun() {
    try { await api(`/api/runs/${run.id}/cancel`, { method: "POST" }); await refreshRun(run.id); } catch (err) { setError(err.message); }
  }

  return <div className="dashboard-shell">
    <header className="app-header">
      <Brand compact />
      <div className="workspace-switcher"><span>{workspace.name}</span><small>独立 Workspace</small><Icon name="chevron" size={14} /></div>
      <div className="header-actions">
        <span className={`status-pill header-status ${status[1]}`}><i />{status[0]}{active && <b className="pulse" />}</span>
        {active ? <button className="stop-button" onClick={stopRun}><Icon name="stop" size={15} />停止</button> : <button className="run-button" onClick={() => document.querySelector(".task-composer textarea")?.focus()} disabled={waiting}><Icon name="play" size={15} />运行</button>}
        <span className="header-divider" />
        <div className="user-menu"><span>{(user.username || user.id).slice(0, 1).toUpperCase()}</span><strong>{user.username || user.id}</strong></div>
        <button className="icon-button logout-button" onClick={onLogout} title="退出登录"><Icon name="logout" size={18} /></button>
      </div>
    </header>
    <div className="dashboard-body">
      <ProductNav runCount={runs.length} fileCount={files.filter(file => file.type === "file").length} />
      <WorkspaceSidebar workspaces={workspaces} selectedId={workspace.id} onSelect={onSelectWorkspace} onCreate={onCreateWorkspace} onDelete={onDeleteWorkspace} files={files} selectedFile={selectedFile} onOpenFile={openFile} busy={active} />
      <main className="agent-panel">
        <header className="agent-tabs">
          <div><button className={viewMode === "agent" ? "active" : ""} onClick={() => setViewMode("agent")}>智能代理</button><button className={viewMode === "trace" ? "active" : ""} onClick={() => setViewMode("trace")}>执行 Trace</button></div>
          <div className="run-select-wrap"><label>运行记录</label><select className="run-select" value={run?.id || ""} onChange={event => selectRun(event.target.value)} disabled={!runs.length}><option value="">暂无运行</option>{runs.map(item => <option key={item.id} value={item.id}>{item.task.slice(0, 48)}</option>)}</select><button className="icon-button" onClick={load} title="刷新"><Icon name="refresh" size={16} /></button></div>
        </header>
        {error && <div className="banner-error" role="alert"><Icon name="alert" size={17} /><span>{error}</span><button onClick={() => setError("")}><Icon name="close" size={15} /></button></div>}
        <div className="conversation">
          <Composer task={task} setTask={setTask} onSubmit={startRun} disabled={active} waiting={waiting} />
          <div className="conversation-heading"><div><strong>{viewMode === "trace" ? "完整 Trace" : "执行过程"}</strong>{run && <small>{viewMode === "trace" ? `${events.length} 条事件` : `${visibleEvents.length} 个关键节点`}</small>}</div>{run && <code>{shortId(run.id)}</code>}</div>
          <div className="event-feed">
            {visibleEvents.length ? visibleEvents.map(event => <EventCard key={event.sequence} event={event} traceMode={viewMode === "trace"} />) : <div className="empty-state"><span><Icon name="bot" size={27} /></span><h2>{run ? "正在恢复执行记录" : "把第一项真实任务交给 Agent"}</h2><p>{run ? "历史 Trace 将从数据库回放到这里。" : "Agent 的消息、工具调用、文件变化和最终结果会实时展示。"}</p></div>}
            <div ref={endRef} />
          </div>
        </div>
        <footer className="agent-footer"><Icon name="check" size={14} /> 内容由 AI 生成，文件写回前请仔细审核</footer>
      </main>
      <RunInspector run={run} diff={diff} selectedFile={selectedFile} fileContent={fileContent} onCloseFile={() => setSelectedFile("")} onApply={() => review("apply")} onDiscard={() => review("discard")} onSaveFeedback={saveFeedback} />
    </div>
  </div>;
}

function EmptyWorkspace({ workspaces, onSelect, onCreate, user, onLogout }) {
  const [name, setName] = useState("");
  async function submit(event) { event.preventDefault(); if (name.trim()) await onCreate(name.trim()); }
  return <div className="dashboard-shell empty-dashboard"><header className="app-header"><Brand compact /><div className="header-actions"><div className="user-menu"><span>{(user.username || user.id).slice(0, 1).toUpperCase()}</span><strong>{user.username || user.id}</strong></div><button className="icon-button" onClick={onLogout}><Icon name="logout" /></button></div></header><main className="workspace-onboarding"><span className="onboarding-icon"><Icon name="workspace" size={30} /></span><p className="kicker">YOUR FIRST WORKSPACE</p><h1>创建独立工作区<br />开始第一条真实任务</h1><p>文件、Run、Trace 与 Sandbox 都会绑定在这个 Workspace 中。</p><form onSubmit={submit}><input autoFocus value={name} onChange={e => setName(e.target.value)} placeholder="例如：产品官网" maxLength={120} /><button className="primary" disabled={!name.trim()}><Icon name="plus" size={17} />创建 Workspace</button></form>{!!workspaces.length && <button className="text-button" onClick={() => onSelect(workspaces[0].id)}>返回已有 Workspace</button>}</main></div>;
}

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [workspaces, setWorkspaces] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  useEffect(() => { api("/api/auth/me").then(setUser).catch(() => {}).finally(() => setLoading(false)); }, []);
  useEffect(() => {
    if (user) api("/api/workspaces").then(items => { setWorkspaces(items); setSelectedId(current => current || items[0]?.id || ""); });
  }, [user]);
  const selected = useMemo(() => workspaces.find(item => item.id === selectedId), [workspaces, selectedId]);
  if (loading) return <div className="loading-screen"><Brand /><span>正在恢复 Workspace…</span></div>;
  if (!user) return <Login onLogin={setUser} />;
  async function create(name) {
    const ws = await api("/api/workspaces", { method: "POST", body: JSON.stringify({ name }) });
    setWorkspaces(current => [ws, ...current]); setSelectedId(ws.id);
  }
  async function remove(id) {
    if (!window.confirm("删除 Workspace 及其全部文件和 Run？此操作无法撤销。")) return;
    await api(`/api/workspaces/${id}`, { method: "DELETE" });
    const next = workspaces.filter(item => item.id !== id); setWorkspaces(next); setSelectedId(next[0]?.id || "");
  }
  async function logout() { await api("/api/auth/logout", { method: "POST" }); setUser(null); setWorkspaces([]); setSelectedId(""); }
  if (!selected) return <EmptyWorkspace workspaces={workspaces} onSelect={setSelectedId} onCreate={create} user={user} onLogout={logout} />;
  return <WorkspaceView workspace={selected} workspaces={workspaces} onSelectWorkspace={setSelectedId} onCreateWorkspace={create} onDeleteWorkspace={remove} user={user} onLogout={logout} />;
}

function relativeTime(value) {
  if (!value) return "刚刚";
  const seconds = Math.max(0, (Date.now() - new Date(value).getTime()) / 1000);
  if (seconds < 60) return "刚刚";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
  return `${Math.floor(seconds / 86400)} 天前`;
}

function formatClock(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date(value));
}

function formatDateTime(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date(value));
}

function formatDuration(milliseconds) {
  const total = Math.max(0, Math.round(milliseconds / 1000));
  if (total < 60) return `${total} 秒`;
  return `${Math.floor(total / 60)} 分 ${total % 60} 秒`;
}

function shortId(value) { return value ? `${value.slice(0, 8)}…${value.slice(-4)}` : ""; }

function countDiff(value = "") {
  const lines = value.split("\n");
  return {
    added: lines.filter(line => line.startsWith("+") && !line.startsWith("+++")).length,
    removed: lines.filter(line => line.startsWith("-") && !line.startsWith("---")).length,
  };
}

function stripThinking(value) {
  return value.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
}

function parsePossibleJson(value) {
  if (typeof value !== "string") return value;
  try { return JSON.parse(value); } catch { return value; }
}

function friendlyTool(name) {
  return {
    list_files: "查看 Workspace 文件",
    read_file: "读取文件",
    write_file: "写入文件",
    search_files: "搜索文件内容",
    run_shell: "运行终端命令",
  }[name] || "调用 Workspace 工具";
}

function friendlyField(name) {
  return {
    path: "路径", query: "关键词", glob: "文件范围", command: "命令", content: "写入内容",
    size: "大小", line: "行号", text: "内容", type: "类型", name: "名称",
  }[name] || name.replaceAll("_", " ");
}

function friendlyChange(changeType) {
  return { created: "已创建", modified: "已修改", deleted: "已删除" }[changeType] || "已变更";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function dedupeAgentEvents(items) {
  const seenAnswers = new Set();
  return items.filter(event => {
    if (event.type !== "agent_message") return true;
    const payload = event.payload || {};
    const answer = stripThinking(String(payload.content || payload.answer || payload.message || ""));
    if (!answer) return true;
    if (seenAnswers.has(answer)) return false;
    seenAnswers.add(answer);
    return true;
  });
}

createRoot(document.getElementById("root")).render(<React.StrictMode><App /></React.StrictMode>);
