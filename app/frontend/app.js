"use strict";
const $ = id => document.getElementById(id);
let info, config, live = {}, frameId = -1, revision = -1, saving = 0, saveQueue = Promise.resolve();
let pollTimer, toastTimer, sliderTimer, recording = null, dirtySlider = false;
const modeDescriptions = {
  mouse: "移动光标，捏合完成鼠标操作。", ppt: "翻页、开始放映与退出，无需拿起翻页笔。",
  media: "播放、切歌与音量，手势掌控你的节奏。", custom: "为自己的工作流程，组合喜欢的手势和快捷键。"
};
const modeBadges = {mouse: "MOUSE", ppt: "PRESENT", media: "MEDIA", custom: "CUSTOM"};
const actionChoices = [
  ["none", "不绑定"], ["click:left", "鼠标左键"], ["click:right", "鼠标右键"],
  ["doubleclick", "鼠标双击"], ["scroll:3", "向上滚动"], ["scroll:-3", "向下滚动"],
  ["key:right", "下一页 →"], ["key:left", "上一页 ←"], ["key:f5", "开始放映 F5"],
  ["key:esc", "退出 Esc"], ["key:b", "黑屏 B"], ["key:playpause", "播放 / 暂停"],
  ["key:nexttrack", "下一首"], ["key:prevtrack", "上一首"], ["key:volumeup", "音量 ＋"],
  ["key:volumedown", "音量 −"], ["key:volumemute", "静音"], ["keyboard", "录制快捷键…"]
];

function toast(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $("toast").hidden = true, 4000);
}
async function call(method, ...args) {
  try {
    const result = await window.pywebview.api[method](...args);
    if (result?.ok === false) throw new Error(result.error);
    return result;
  } catch (error) { toast(error.message || String(error)); throw error; }
}
function posture() { return config.postures[config.posture]; }
function renderSliders() {
  for (const field of ["gain_x", "gain_y", "smoothing"]) {
    $(field).value = posture()[field];
    $(field + "_value").textContent = field === "smoothing" ? Math.round(posture()[field] * 100) + "%" : posture()[field].toFixed(2) + "×";
  }
}
function renderConfig() {
  document.querySelectorAll("[data-mode]").forEach(b => b.classList.toggle("active", b.dataset.mode === config.mode));
  document.querySelectorAll("[data-posture]").forEach(b => b.classList.toggle("active", b.dataset.posture === config.posture));
  $("modeDescription").textContent = modeDescriptions[config.mode];
  $("profileBadge").textContent = modeBadges[config.mode];
  $("cameraId").value = config.camera;
  $("cudaToggle").checked = config.cuda;
  $("cudaReason").textContent = info.cuda.reason;
  $("pinchThreshold").value = config.pinch_threshold;
  $("holdMs").value = config.hold_ms;
  $("modelBanner").hidden = info.model_ready;
  renderSliders(); renderBindings(); drawOverlay();
}
function renderBindings() {
  const host = $("bindings"); host.replaceChildren();
  const gestures = [...Object.entries(info.gestures).map(([id, name]) => ({id, name})), ...config.custom_gestures];
  gestures.forEach((gesture, index) => {
    const action = config.bindings[config.mode][gesture.id] || "none";
    const row = document.createElement("div"); row.className = "binding-row";
    const title = document.createElement("div"); title.className = "gesture-title";
    const number = document.createElement("span"); number.className = "gesture-index"; number.textContent = String(index + 1).padStart(2, "0");
    const name = document.createElement("span"); name.className = "name"; name.textContent = gesture.name; name.title = gesture.name;
    title.append(number, name);
    if (gesture.id.startsWith("pose_")) {
      const del = document.createElement("button"); del.className = "delete-gesture"; del.textContent = "×";
      del.title = "删除此手势及所有模式中的绑定"; del.setAttribute("aria-label", "删除 " + gesture.name);
      del.onclick = async () => { if (confirm("删除「" + gesture.name + "」及其绑定？")) { await call("delete_gesture", gesture.id); await refreshInfo(); } };
      title.append(del);
    }
    const group = document.createElement("div"); group.className = "binding-action";
    const select = document.createElement("select"); select.setAttribute("aria-label", gesture.name + "触发操作");
    actionChoices.forEach(([value, label]) => { const option = document.createElement("option"); option.value = value; option.textContent = label; select.append(option); });
    const customKey = action.startsWith("hotkey:") || (action.startsWith("key:") && !actionChoices.some(([key]) => key === action));
    select.value = customKey ? "keyboard" : action;
    const record = document.createElement("button"); record.className = "key-recorder";
    record.textContent = customKey ? action.split(":")[1].toUpperCase() : "按键…";
    record.title = "点击后按下目标快捷键；Esc 取消"; record.hidden = !customKey;
    record.onclick = () => {
      recording?.cancel();
      record.classList.add("recording"); record.textContent = "请按键…";
      recording = {id: gesture.id, mode: config.mode, button: record, old: action,
                   cancel: () => { recording = null; renderBindings(); }};
    };
    select.onchange = async () => {
      await call("pause_output");
      if (select.value === "keyboard") { record.hidden = false; record.click(); }
      else { config.bindings[config.mode][gesture.id] = select.value; await persist(); }
    };
    group.append(select, record); row.append(title, group); host.append(row);
  });
}

function persist() {
  dirtySlider = false;
  const snapshot = JSON.parse(JSON.stringify(config));
  saving++; $("savedStatus").textContent = "保存中…";
  const operation = saveQueue.then(async () => {
    try { await call("save_settings", snapshot); $("savedStatus").textContent = "已保存"; }
    catch (error) { $("savedStatus").textContent = "保存失败"; throw error; }
    finally { saving--; }
  });
  saveQueue = operation.catch(() => {});
  return operation;
}
async function refreshInfo() {
  info = await call("get_info"); config = info.config; renderConfig();
}

function drawOverlay() {
  const canvas = $("overlay"), box = $("preview").getBoundingClientRect(), ratio = devicePixelRatio || 1;
  canvas.width = Math.round(box.width * ratio); canvas.height = Math.round(box.height * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  if (!config || !live.running) return;
  const {gain_x: gx, gain_y: gy, center} = posture();
  const image = $("cameraFrame"), aspect = image.naturalWidth ? image.naturalWidth / image.naturalHeight : 4 / 3;
  const w = Math.min(box.width, box.height * aspect), h = w / aspect;
  ctx.translate((box.width - w) / 2, (box.height - h) / 2);
  ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip();
  const x = (center[0] - .5 / gx) * w, y = (center[1] - .5 / gy) * h;
  ctx.fillStyle = "#49d9b907"; ctx.strokeStyle = "#52d3b7b0"; ctx.lineWidth = 1;
  ctx.setLineDash([6, 5]); ctx.fillRect(x, y, w / gx, h / gy); ctx.strokeRect(x, y, w / gx, h / gy); ctx.setLineDash([]);
  ctx.strokeStyle = "#a2ead3b0"; ctx.beginPath();
  ctx.moveTo(center[0] * w - 8, center[1] * h); ctx.lineTo(center[0] * w + 8, center[1] * h);
  ctx.moveTo(center[0] * w, center[1] * h - 8); ctx.lineTo(center[0] * w, center[1] * h + 8); ctx.stroke();
  const finger = live.landmarks?.[8];
  if (live.tuning && finger) { ctx.fillStyle = "#ffcc78"; ctx.beginPath(); ctx.arc(finger.x * w, finger.y * h, 7, 0, Math.PI * 2); ctx.fill(); }
}
function renderState(state) {
  live = state;
  const enrolling = !!state.capture?.startsWith("保持") || !!state.capture?.startsWith("准备");
  if (state.frame) { $("cameraFrame").src = state.frame; frameId = state.frame_id; }
  $("cameraFrame").hidden = !state.running || !$("cameraFrame").getAttribute("src");
  $("previewPlaceholder").hidden = !$("cameraFrame").hidden;
  $("previewButton").textContent = state.running ? "停止预览" : "启动预览";
  $("armButton").disabled = !state.running || !!state.tuning || enrolling;
  $("armButton").textContent = state.arming ? "切换目标窗口 · " + state.arming.toFixed(1) + "s" : state.armed ? "暂停控制" : "启用控制 ↗";
  $("outputStatus").classList.toggle("armed", state.armed);
  $("outputStatus").replaceChildren(Object.assign(document.createElement("i"), {}), document.createTextNode(state.arming ? "准备启用" : state.armed ? "控制已启用" : "控制已暂停"));
  $("cameraStatus").textContent = state.running ? "实时画面" : "摄像头未启动";
  $("cameraStatus").classList.toggle("on", state.running);
  $("fps").textContent = state.running ? state.fps + " FPS" : "— FPS";
  $("gestureLabel").textContent = state.gesture;
  $("lastAction").textContent = state.last_action;
  $("preprocessing").textContent = state.preprocessing;
  $("cameraId").disabled = state.running;
  $("cudaToggle").disabled = state.running || !info.cuda.available;
  $("centerButton").disabled = !state.running;
  $("captureButton").disabled = !state.running || !!state.tuning || enrolling;
  document.querySelectorAll("[data-mode], [data-posture], .binding-action select, .key-recorder, .delete-gesture, input[type=range], #resetPosture, #centerButton, #pinchThreshold, #holdMs").forEach(element => element.disabled = enrolling);
  $("centerButton").disabled = !state.running || enrolling;
  $("errorBanner").hidden = !state.error; $("errorBanner").textContent = state.error;
  $("captureStatus").hidden = !state.capture; $("captureStatus").textContent = state.capture;
  $("tuningHint").hidden = !state.tuning;
  $("tuningHint").textContent = state.tuning === "smoothing" ? "拇指与食指捏合，上下拖动调整响应速度。松开后可重新定位。" : "拇指与食指捏合，左右拖动调整灵敏度。松开后可重新定位。";
  document.querySelectorAll("[data-tune]").forEach(button => {
    const active = button.dataset.tune === state.tuning;
    button.classList.toggle("active", active); button.textContent = active ? "完成调整" : "手势调节";
    button.disabled = !state.running || enrolling;
  });
  if (state.tuning && !saving && !dirtySlider) {
    config.postures[config.posture] = state.posture_settings; renderSliders();
  }
  drawOverlay();
}
async function poll() {
  try {
    const state = await window.pywebview.api.get_state(frameId, document.hasFocus());
    if (state.revision !== revision && !saving && !dirtySlider && !recording) {
      await refreshInfo(); revision = state.revision;
    }
    renderState(state);
    $("connectionStatus").textContent = "桌面后端已连接";
  } catch (error) { $("connectionStatus").textContent = "连接中断，请重新启动 GUI"; }
  pollTimer = setTimeout(poll, 120);
}

function connectEvents() {
  $("cameraFrame").onload = drawOverlay;
  $("previewButton").onclick = async () => { await call(live.running ? "stop_preview" : "start_preview"); };
  $("armButton").onclick = async () => {
    await call(live.armed || live.arming ? "pause_output" : "arm_output");
    if (!live.armed && !live.arming) toast("3 秒后启用，请切换至目标窗口。F8 随时暂停。");
  };
  $("centerButton").onclick = async () => { await call("center_on_hand"); await refreshInfo(); toast("已将当前舒适手位设为光标中心"); };
  document.querySelectorAll("[data-mode]").forEach(button => button.onclick = async () => {
    recording?.cancel(); config.mode = button.dataset.mode; await persist(); renderConfig();
  });
  document.querySelectorAll("[data-posture]").forEach(button => button.onclick = async () => {
    await call("set_tuning", ""); config.posture = button.dataset.posture; await persist(); renderConfig();
  });
  ["gain_x", "gain_y", "smoothing"].forEach(field => {
    $(field).oninput = () => {
      config.postures[config.posture][field] = Number($(field).value); dirtySlider = true;
      renderSliders(); drawOverlay(); clearTimeout(sliderTimer); sliderTimer = setTimeout(() => persist().catch(() => {}), 250);
    };
    $(field).onpointerdown = () => { call("pause_output"); if (live.tuning) call("set_tuning", ""); };
  });
  document.querySelectorAll("[data-tune]").forEach(button => button.onclick = async () => {
    clearTimeout(sliderTimer); if (dirtySlider) await persist();
    await call("set_tuning", live.tuning === button.dataset.tune ? "" : button.dataset.tune);
  });
  $("resetPosture").onclick = async () => {
    await call("set_tuning", ""); config.postures[config.posture] = {gain_x: 2.2, gain_y: 2.8, smoothing: .4, center: [.5, .55]};
    await persist(); renderConfig();
  };
  $("cameraId").onchange = async () => { config.camera = Number($("cameraId").value); await persist(); };
  $("cudaToggle").onchange = async () => { config.cuda = $("cudaToggle").checked; await persist(); };
  $("pinchThreshold").onchange = async () => { config.pinch_threshold = Number($("pinchThreshold").value); await persist(); };
  $("holdMs").onchange = async () => { config.hold_ms = Number($("holdMs").value); await persist(); };
  $("modelButton").onclick = async () => {
    $("modelButton").disabled = true; $("modelButton").textContent = "下载并校验中…";
    try { await call("prepare_model"); await refreshInfo(); toast("模型已准备好，可以启动预览"); }
    finally { $("modelButton").disabled = false; $("modelButton").textContent = "下载手部模型"; }
  };
  $("captureButton").onclick = async () => { await call("pause_output"); $("captureDialog").returnValue = ""; $("captureDialog").showModal(); $("gestureName").focus(); };
  $("captureDialog").addEventListener("close", async () => {
    if ($("captureDialog").returnValue === "capture") await call("capture_gesture", $("gestureName").value);
  });
  window.addEventListener("focus", () => { if (info) call("pause_output"); });
  window.addEventListener("keydown", async event => {
    if (event.key === "F8") { event.preventDefault(); await call("pause_output"); recording?.cancel(); return; }
    if (!recording) return;
    event.preventDefault(); event.stopPropagation();
    if (event.key === "Escape") { recording.cancel(); return; }
    if (["Control", "Alt", "Shift", "Meta"].includes(event.key)) return;
    const aliases = {ArrowRight: "right", ArrowLeft: "left", ArrowUp: "up", ArrowDown: "down", " ": "space", PageUp: "pageup", PageDown: "pagedown", Enter: "enter", Delete: "delete", Backspace: "backspace", Escape: "esc"};
    const keys = []; if (event.ctrlKey) keys.push("ctrl"); if (event.altKey) keys.push("alt"); if (event.shiftKey) keys.push("shift"); if (event.metaKey) keys.push("win");
    keys.push(aliases[event.key] || event.key.toLowerCase());
    const target = recording; recording = null;
    config.bindings[target.mode][target.id] = (keys.length > 1 ? "hotkey:" : "key:") + keys.join("+");
    await persist(); renderBindings();
  }, true);
  new ResizeObserver(drawOverlay).observe($("preview"));
}

let readyResolve;
window.desktopReady = new Promise(resolve => readyResolve = resolve);
window.addEventListener("pywebviewready", async () => {
  try {
    await refreshInfo(); connectEvents(); await poll();
    window.desktopSmokeResult = {ok: true, title: document.title, bindings: $("bindings").children.length,
                                 mode: config.mode, cudaAvailable: info.cuda.available};
    readyResolve(window.desktopSmokeResult);
  } catch (error) {
    window.desktopSmokeResult = {ok: false, error: error.message}; readyResolve(window.desktopSmokeResult);
  }
});
window.addEventListener("unhandledrejection", event => { event.preventDefault(); });
