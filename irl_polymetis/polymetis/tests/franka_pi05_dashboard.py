#!/usr/bin/env python3
"""Local browser dashboard for the Franka + Pi05 reproduction stack.

Run this file on the robot computer, then open http://127.0.0.1:8765.
The dashboard intentionally binds to localhost because its buttons control
robot hardware and launch processes on this machine.
"""

import html
import json
import os
import re
import signal
import shutil
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pyrealsense2 as rs

from fr3_camera_config import CAMERA_SERIALS_BY_ROLE, DEFAULT_CAMERA_ROLES, policy_camera_mapping


HOST = "127.0.0.1"
PORT = 8765
ROOT = "/home/lsk/irl_polymetis"
GELLO_ROOT = "/home/lsk/gello_software"
POLY_ENV = "/home/lsk/anaconda3/envs/polymetis_env"
DEPLOY_SCRIPT = f"{ROOT}/polymetis/tests/deploy_fr3_chunk-final.py"
POLICY_DIR = "/home/lsk/work/openpi_fr3_2object/checkpoints/pi05_fr3_gello_lerobot_2object_pick_place/fr3_left_wrist_10hz_4gpu_20260929_120118/29999"
POLICY_CONFIG = "pi05_fr3_gello_lerobot_2object_pick_place"
DEFAULT_PROMPT = "pick up the object and place it into the box"
DEFAULT_COLLECTION_TASK = "pick up the object and place it into the box"
DEFAULT_SAVE_DIR = "/home/lsk/dataset/fr3_gello_lerobot"
COLLECTION_PREVIEW_DIR = Path("/tmp/franka_collection_preview")
GELLO_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTC5M02N-if00-port0"
GELLO_AGENT_FILE = f"{GELLO_ROOT}/gello/agents/gello_agent.py"
OFFSET_START_JOINTS = (0, 0, 0, -1.5708, 0, 1.5708, 0.588727)
OFFSET_JOINT_SIGNS = (1, 1, 1, 1, 1, -1, 1)


COMMANDS = {
    "robot": (
        "Robot server",
        "source /home/lsk/anaconda3/etc/profile.d/conda.sh && "
        "conda activate polymetis_env && "
        "export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\" && "
        "exec taskset -c 6,7 launch_robot.py robot_client=franka_hardware "
        "use_real_time=false",
    ),
    "gripper": (
        "Franka gripper",
        "source /home/lsk/anaconda3/etc/profile.d/conda.sh && "
        "conda activate polymetis_env && "
        "export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\" && "
        "exec taskset -c 6,7 launch_gripper.py gripper=franka_hand",
    ),
    "policy": (
        "Pi05 policy server",
        "source /home/lsk/anaconda3/etc/profile.d/conda.sh && "
        "conda activate pi0_env && cd /home/lsk/openpi && "
        "exec uv run scripts/serve_policy.py policy:checkpoint "
        '--policy.config="$POLICY_CONFIG" '
        '--policy.dir="$POLICY_DIR"',
    ),
    "deploy": (
        "Deployment",
        f"source /home/lsk/anaconda3/etc/profile.d/conda.sh && "
        f"conda activate polymetis_env && exec "
        f"/home/lsk/anaconda3/envs/polymetis_env/bin/python {DEPLOY_SCRIPT}",
    ),
    "gello_move": (
        "Move FR3 to GELLO start",
        f"source /home/lsk/anaconda3/etc/profile.d/conda.sh && "
        f"conda activate polymetis_env && "
        f"export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\" && "
        f"cd {GELLO_ROOT} && {POLY_ENV}/bin/python scripts/move_franka_to_gello_start.py "
        "--ip-address localhost --port 50051 "
        "--target-joints 0 0 0 -1.5708 0 1.5708 0.588727 --time-to-go 5.0 --yes",
    ),
    "gello_node": (
        "GELLO FR3 robot node",
        f"source /home/lsk/anaconda3/etc/profile.d/conda.sh && "
        f"conda activate polymetis_env && "
        f"export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\" && "
        f"cd {GELLO_ROOT} && {POLY_ENV}/bin/python experiments/launch_nodes.py "
        "--robot fr3_polymetis --robot-ip localhost --polymetis-port 50051 "
        "--robot-port 6001 --control-gripper --gripper-ip localhost "
        "--gripper-port 50052 --max-gripper-width 0.08 --gripper-speed 0.1 "
        "--gripper-force 20.0",
    ),
    "collector": (
        "LeRobot data collector",
        f"source /home/lsk/anaconda3/etc/profile.d/conda.sh && "
        f"conda activate polymetis_env && "
        f"export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\" && "
        f"cd {GELLO_ROOT} && "
        f"if [ \"$COLLECTION_AUTOSTART\" = \"1\" ]; then AUTO_START=--auto-start; else AUTO_START=; fi; "
        f"exec {POLY_ENV}/bin/python gello/agents/collect_fr3_gello_lerobot.py "
        f"--gello-port \"$GELLO_PORT\" --robot-port 6001 "
        f"--polymetis-ip localhost --polymetis-port 50051 "
        f"--camera-serials $COLLECTION_CAMERA_SERIALS "
        f"--task \"$COLLECTION_TASK\" --save-dir \"$COLLECTION_SAVE_DIR\" "
        f"--preview-dir \"$COLLECTION_PREVIEW_DIR\" $AUTO_START",
    ),
}


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>南京大学AIR实验室franka启动台</title>
<style>
:root{--bg:#0b1018;--panel:#121a25;--panel2:#182333;--line:#2b3a4d;--text:#e9f0f7;--muted:#8da0b5;--cyan:#4de1c1;--amber:#ffbd66;--red:#ff6d7a;--blue:#83a9ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 75% 0,#193247 0,#0b1018 42%);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}main{max-width:1450px;margin:auto;padding:28px 32px 44px}.top{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:24px}.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.16em;text-transform:uppercase;font-weight:700}.title{font-size:30px;letter-spacing:-.04em;margin:5px 0 4px}.subtitle{color:var(--muted);margin:0}.pill{border:1px solid var(--line);border-radius:999px;padding:7px 12px;color:var(--muted);background:#101824}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--amber);margin-right:7px}.dot.ok{background:var(--cyan);box-shadow:0 0 12px var(--cyan)}.layout{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(360px,.8fr);gap:18px}.panel{background:linear-gradient(145deg,rgba(24,35,51,.96),rgba(15,22,33,.98));border:1px solid var(--line);border-radius:16px;box-shadow:0 18px 55px #0003}.panel-head{padding:18px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}.panel-title{font-weight:700}.section{padding:18px 20px}.cams{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.cam{min-height:190px;background:#080d14;border:1px solid var(--line);border-radius:11px;overflow:hidden;position:relative}.cam img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover;background:#080d14}.cam .caption{padding:8px 10px;color:var(--muted);font-size:12px}.cam .serial{color:var(--text);font-family:ui-monospace,monospace}.empty{padding:35px 20px;color:var(--muted);text-align:center;border:1px dashed var(--line);border-radius:11px}.services{display:grid;gap:10px}.service{display:grid;grid-template-columns:12px 1fr auto;gap:12px;align-items:center;padding:13px 14px;background:#0e1621;border:1px solid var(--line);border-radius:11px}.service .dot{margin:0}.service-name{font-weight:650}.service-detail{font-size:12px;color:var(--muted);margin-top:2px}.status{font-size:12px;color:var(--muted)}button{border:0;border-radius:9px;padding:9px 13px;color:#081117;background:var(--cyan);font-weight:750;cursor:pointer}button:hover{filter:brightness(1.08)}button:disabled{opacity:.4;cursor:not-allowed}button.secondary{background:#253449;color:var(--text);border:1px solid #3b506a}button.danger{background:var(--red);color:#21080d}.controls{display:flex;gap:9px;flex-wrap:wrap}.prompt{width:100%;background:#0a111b;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:12px;font:14px/1.4 inherit;resize:vertical;min-height:76px}.label{font-size:12px;color:var(--muted);margin-bottom:7px;display:block}.warning{display:flex;gap:10px;align-items:flex-start;color:#ffdba7;background:#312819;border:1px solid #73572d;border-radius:10px;padding:11px 12px;font-size:12px;margin-top:12px}.warning input{accent-color:var(--amber);margin-top:3px}.log{height:300px;overflow:auto;background:#080d14;color:#b6c8d9;border-radius:10px;border:1px solid var(--line);padding:12px;font:12px/1.55 ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap}.footer{color:var(--muted);font-size:12px;margin-top:18px}.full{grid-column:1/-1}@media(max-width:900px){main{padding:20px 14px}.layout{grid-template-columns:1fr}.cams{grid-template-columns:1fr}.top{flex-direction:column}.full{grid-column:auto}}
/* Desktop layout: cameras occupy the left; launch and logs stack on the right. */
#cameraPanel{grid-column:1;grid-row:1 / span 2}
#launchPanel{grid-column:2;grid-row:1}
#logPanel{grid-column:2;grid-row:2}
#promptPanel{grid-column:1 / -1;grid-row:3}
@media(max-width:900px){#cameraPanel,#launchPanel,#logPanel,#promptPanel{grid-column:auto;grid-row:auto}}
.tabs{display:flex;gap:8px;margin:0 0 18px}.tab-button{background:#162436;color:var(--muted);border:1px solid var(--line)}.tab-button.active{background:var(--cyan);color:#081117;border-color:var(--cyan)}.tab-page{display:none}.tab-page.active{display:block}.collection-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(340px,.8fr);gap:18px}.field{margin-bottom:14px}.field input{width:100%;background:#0a111b;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:11px;font:14px inherit}.path-row{display:flex;gap:8px;align-items:center}.path-row input{flex:1;min-width:0}.folder-button{width:44px;height:42px;padding:0;font-size:20px;line-height:1;background:#253449;color:var(--text);border:1px solid #3b506a}.hint{color:var(--muted);font-size:12px}.step-list{display:grid;gap:10px}.step{display:flex;gap:12px;align-items:flex-start;padding:12px 14px;background:#0e1621;border:1px solid var(--line);border-radius:11px}.step-num{color:var(--cyan);font-weight:800;font-family:ui-monospace,monospace}.collection-actions{display:flex;gap:9px;flex-wrap:wrap}.recording{color:var(--red);font-weight:800}
#collectionCameraPanel{grid-column:1;grid-row:1 / span 2}#collectionControlPanel{grid-column:2;grid-row:1}#collectionLogPanel{grid-column:2;grid-row:2}#collectionStepsPanel{grid-column:1 / -1;grid-row:3}@media(max-width:900px){#collectionCameraPanel,#collectionControlPanel,#collectionLogPanel,#collectionStepsPanel{grid-column:auto;grid-row:auto}}
</style>
<style>.episode-panel{margin-top:18px;border-top:1px solid var(--line);padding-top:15px}.episode-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:9px}.episode-list{display:flex;flex-direction:column;gap:7px;max-height:240px;overflow-y:auto;padding-right:4px}.episode-row{display:flex;align-items:center;gap:9px;width:100%;min-height:38px;padding:8px 10px;background:#0e1621;border:1px solid var(--line);border-radius:8px;font-size:12px}.episode-row input{flex:0 0 auto;accent-color:var(--cyan)}.episode-meta{flex:1;min-width:0;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.episode-state{color:var(--cyan);font-weight:700;flex:0 0 auto}</style>
<style>.camera-config{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 0 14px}.camera-config select{background:#0a111b;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px}.camera-options{display:flex;gap:7px;flex-wrap:wrap}.camera-option{display:flex;gap:5px;align-items:center;color:var(--muted);font-size:12px;background:#0e1621;border:1px solid var(--line);border-radius:7px;padding:6px 8px}.camera-option input{accent-color:var(--cyan)}</style></head>
<body><main>
<header class="top"><div><div class="eyebrow">NANJING UNIVERSITY AIR LAB / FRANKA</div><h1 class="title">南京大学AIR实验室franka启动台</h1><p class="subtitle">机器人、夹爪、策略服务与多路视觉的单页启动台</p></div><div class="pill"><span id="systemDot" class="dot"></span><span id="systemText">正在检查系统</span></div></header>
<nav class="tabs"><button id="controlTabButton" class="tab-button active" onclick="showTab('controlTab','controlTabButton')">推理部署</button><button id="collectionTabButton" class="tab-button" onclick="showTab('collectionTab','collectionTabButton')">数据采集</button></nav>
<div id="controlTab" class="tab-page active"><div class="layout">
<section id="cameraPanel" class="panel"><div class="panel-head"><span class="panel-title">Camera monitor</span><span id="camCount" class="status">检测中…</span></div><div class="section"><div id="cameraConfig" class="camera-config"></div><p id="deploymentCameraHint" class="hint"></p><div id="cams" class="cams"><div class="empty">正在枚举 RealSense 相机…</div></div></div></section>
<section id="launchPanel" class="panel"><div class="panel-head"><span class="panel-title">Launch sequence</span><span class="status">建议按 1 → 2 → 3 → 4</span></div><div class="section">
<div class="field"><label class="label" for="policyDir">Policy checkpoint 目录</label><input id="policyDir" type="text" spellcheck="false" autocomplete="off" placeholder="/home/lsk/.../checkpoints/.../29999" disabled></div>
<div class="field"><label class="label" for="policyConfig">Policy 配置名（与训练时一致）</label><input id="policyConfig" type="text" spellcheck="false" autocomplete="off" placeholder="pi05_fr3_gello_lerobot_2object_pick_place" disabled></div>
<p id="policyHint" class="hint">正在读取策略设置…</p><p id="policyFeedback" class="hint" role="status" style="overflow-wrap:anywhere"></p>
<div id="services" class="services"></div></div></section>
<section id="logPanel" class="panel"><div class="panel-head"><span class="panel-title">Live event log</span><button class="secondary" onclick="clearLog()">清空</button></div><div class="section"><div id="log" class="log"></div></div></section>
<section id="promptPanel" class="panel"><div class="panel-head"><span class="panel-title">Deployment prompt</span><span class="status">仅在启动复现前生效</span></div><div class="section"><label class="label" for="prompt">TASK_PROMPT</label><textarea id="prompt" class="prompt"></textarea><label class="warning"><input id="confirm" type="checkbox">我确认机械臂已处于安全状态、工作空间无人，并允许启动实际控制。</label><div class="controls" style="margin-top:13px"><button id="deploy" onclick="startDeploy()" disabled>启动</button><button class="danger" onclick="stopAll()">停止全部服务</button></div></div></section>
<div class="footer full">只监听 127.0.0.1:8765。网页关闭不会自动停止硬件进程，请使用“停止全部服务”。</div>
</div></div>
<div id="collectionTab" class="tab-page"><div class="collection-layout">
<section id="collectionCameraPanel" class="panel"><div class="panel-head"><span class="panel-title">Camera monitor</span><span id="dataCamCount" class="status">检测中…</span></div><div class="section"><div id="dataCameraConfig" class="camera-config"></div><div id="dataCams" class="cams"><div class="empty">正在枚举 RealSense 相机…</div></div><p id="dataCamHint" class="hint" style="margin-bottom:0">采集器启动后会暂停网页预览，由采集器独占相机。</p></div></section>
<section id="collectionControlPanel" class="panel"><div class="panel-head"><span class="panel-title">LeRobot 数据采集</span><span id="collectorStatus" class="status">准备中</span></div><div class="section">
<div id="collectionServices" class="services" style="margin-bottom:14px"></div>
<div class="field"><label class="label" for="collectionTask">任务指令</label><input id="collectionTask" value="pick up the object and place it into the box"></div>
<div class="field"><label class="label" for="saveDir">保存目录</label><div class="path-row"><input id="saveDir" value="/home/lsk/dataset/fr3_gello_lerobot"><button class="folder-button" type="button" title="选择保存目录" aria-label="选择保存目录" onclick="chooseDirectory()">📁</button></div></div>
<div class="field"><label class="label" for="gelloPort">GELLO 串口</label><input id="gelloPort" value="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTC5M02N-if00-port0"></div>
<label class="warning"><input id="autoStart" type="checkbox">采集器启动后立即开始录制（等价于 `--auto-start`）</label>
<div class="collection-actions" style="margin-top:14px"><button class="secondary" onclick="start('gello_move')">移动到 GELLO 起始位</button><button class="secondary" onclick="recalculateOffsets()">重新计算并写回 offsets</button><button class="secondary" onclick="start('gello_node')">启动 GELLO node</button><button class="danger" onclick="stop('gello_node')">停止 GELLO node</button><button id="collectorStart" onclick="startCollector()">启动采集器</button></div>
<div class="collection-actions" style="margin-top:10px"><button onclick="collectorCommand('r')">开始录制</button><button class="secondary" onclick="collectorCommand('s')">停止并保存</button><button class="secondary" onclick="collectorCommand('d')">丢弃 episode</button><button class="danger" onclick="collectorCommand('q')">退出采集器</button></div>
<p class="hint" style="margin-bottom:0">网页快捷键：r 开始，s 停止并保存，d 丢弃，q 退出。输入框获得焦点时不会触发快捷键。</p>
</div></section>
<section id="collectionLogPanel" class="panel"><div class="panel-head"><span class="panel-title">Live event log</span><button class="secondary" onclick="clearLog()">清空</button></div><div class="section"><div id="collectionLog" class="log"></div></div></section>
<section id="collectionStepsPanel" class="panel"><div class="panel-head"><span class="panel-title">采集启动顺序</span><span class="status">按步骤执行</span></div><div class="section"><div class="step-list"><div class="step"><span class="step-num">01</span><div>启动 Robot server 和 Franka gripper</div></div><div class="step"><span class="step-num">02</span><div>点击“移动到 GELLO 起始位”</div></div><div class="step"><span class="step-num">03</span><div>手动将 GELLO leader 摆到同一物理姿态，点击“重新计算并写回 offsets”</div></div><div class="step"><span class="step-num">04</span><div>点击“启动 GELLO node”，保持节点运行</div></div><div class="step"><span class="step-num">05</span><div>填写任务和保存目录，点击“启动采集器”</div></div><div class="step"><span class="step-num">06</span><div>通过按钮开始录制，停止后保存 MP4 与 parquet</div></div></div></div></section>
</div></div>
</main>
<script>
const names={robot:'Robot server',gripper:'Franka gripper',policy:'Pi05 policy server',deploy:'Deployment',gello_move:'Move FR3 to GELLO start',gello_node:'GELLO FR3 robot node',collector:'LeRobot data collector'};
let state={};
let pendingCameraSelection=null;
let policySettingsInitialized=false;
let policyStartPending=false;
function savedPolicyValue(key,fallback){try{return localStorage.getItem(key)??fallback}catch(e){return fallback}}
function savePolicySettings(){try{localStorage.setItem('frankaPolicyDir',document.getElementById('policyDir').value);localStorage.setItem('frankaPolicyConfig',document.getElementById('policyConfig').value)}catch(e){}}
function renderPolicySettings(s){
    const settings=s.policy_settings;
    if(!settings)return;
    const running=Boolean(s.services.policy?.running);
    const dir=document.getElementById('policyDir');
    const config=document.getElementById('policyConfig');
    if(!policySettingsInitialized||running){
        dir.value=running?settings.checkpoint:savedPolicyValue('frankaPolicyDir',settings.checkpoint);
        config.value=running?settings.config:savedPolicyValue('frankaPolicyConfig',settings.config);
        policySettingsInitialized=true;
    }
    dir.disabled=config.disabled=running||policyStartPending;
    document.getElementById('policyHint').textContent=running?'以上为当前策略使用的设置。更换 checkpoint：停止 Pi05 policy server → 修改 → 再启动。':'填写本机 checkpoint 目录（如 29999）及训练配置名，再点击 Pi05 policy server 的“启动”。输入会保存在当前浏览器中。';
}
async function startPolicy(){
    if(policyStartPending)return;
    const checkpoint=document.getElementById('policyDir').value.trim();
    const config=document.getElementById('policyConfig').value.trim();
    const feedback=document.getElementById('policyFeedback');
    if(!checkpoint||!config){feedback.textContent='请填写 checkpoint 目录和训练配置名';return}
    savePolicySettings();
    policyStartPending=true;
    render(state);
    feedback.textContent='正在启动 Pi05 policy server…';
    try{
        const r=await api('/api/start/policy',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({checkpoint,config})});
        feedback.textContent=r.message||'';
        if(r.message)log(r.message);
    }finally{policyStartPending=false;await refresh()}
}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function log(s){const el=document.getElementById('log');el.textContent+=s+'\n';el.scrollTop=el.scrollHeight}
function clearLog(){document.getElementById('log').textContent='';document.getElementById('collectionLog').textContent=''}
function ensureCollectionFeedback(){let el=document.getElementById('recordingFeedback');if(!el){const groups=document.querySelectorAll('#collectionControlPanel .collection-actions');const host=groups[groups.length-1];el=document.createElement('p');el.id='recordingFeedback';el.className='hint';el.style.marginBottom='0';el.textContent='录制状态：未开始';host.parentElement.appendChild(el)}return el}
function ensureEpisodePanel(){let panel=document.getElementById('episodePanel');if(panel)return panel;const host=document.getElementById('dataCams').parentElement;panel=document.createElement('div');panel.id='episodePanel';panel.className='episode-panel';panel.innerHTML='<div class="episode-head"><span class="panel-title">本次采集 episodes</span><span id="episodeCount" class="status">0 条</span></div><div id="episodeList" class="episode-list"><div class="empty">暂无已保存 episode</div></div><div class="collection-actions" style="margin-top:9px"><button class="danger" onclick="deleteSelectedEpisodes()">删除选中</button><button class="secondary" onclick="refreshEpisodes()">刷新列表</button></div>';host.appendChild(panel);return panel}
async function refreshEpisodes(){if(!document.getElementById('episodeList'))return;const checked=new Set([...document.querySelectorAll('#episodeList input:checked')].map(x=>x.value));const saveDir=document.getElementById('saveDir')?.value.trim()||'';const r=await api('/api/episodes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({save_dir:saveDir})});if(!r.ok)return;const episodes=r.episodes||[];document.getElementById('episodeCount').textContent=`${episodes.length} 条`;document.getElementById('episodeList').innerHTML=episodes.length?episodes.map(e=>`<label class="episode-row"><input type="checkbox" value="${esc(e.id)}" ${checked.has(e.id)?'checked':''}><span class="episode-meta">${esc(e.id)} · ${e.videos} 路视频 · ${(e.size/1024).toFixed(1)} KB</span><span class="episode-state">已保存</span></label>`).join(''):'<div class="empty">暂无已保存 episode</div>'}
async function deleteSelectedEpisodes(){const ids=[...document.querySelectorAll('#episodeList input:checked')].map(x=>x.value);if(!ids.length){log('请先选择要删除的 episode');return}if(!confirm(`确定删除 ${ids.length} 条 episode 及对应视频吗？`))return;const saveDir=document.getElementById('saveDir').value.trim();const r=await api('/api/delete-episodes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({save_dir:saveDir,episode_ids:ids})});if(r.message)log(r.message);if(r.ok)await refreshEpisodes()}
function showTab(tab,button){document.querySelectorAll('.tab-page').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab-button').forEach(x=>x.classList.remove('active'));document.getElementById(tab).classList.add('active');document.getElementById(button).classList.add('active')}
function cameraConfig(id,cameras,selected){const serials=(pendingCameraSelection||selected||[]).filter(s=>cameras.some(c=>c.serial===s));const count=Math.max(0,serials.length);return `<span class="status">选择数量</span><select id="${id}Count" onchange="limitCameraSelection('${id}')">${Array.from({length:cameras.length+1},(_,i)=>`<option value="${i}" ${i===count?'selected':''}>${i}</option>`).join('')}</select><div id="${id}Options" class="camera-options">${cameras.length?cameras.map(c=>`<label class="camera-option"><input type="checkbox" value="${esc(c.serial)}" ${serials.includes(c.serial)?'checked':''} onchange="rememberCameraSelection('${id}')"><span>${esc(c.name)}<br>${esc(c.serial)}</span></label>`).join(''):'<span class="hint">未检测到相机</span>'}</div><button class="secondary" onclick="applyCameraSelection('${id}')">应用选择</button>`}
function rememberCameraSelection(id){pendingCameraSelection=[...document.querySelectorAll('#'+id+'Options input:checked')].map(x=>x.value)}
function limitCameraSelection(id){const count=Number(document.getElementById(id+'Count').value);const boxes=[...document.querySelectorAll('#'+id+'Options input')];boxes.forEach((box,index)=>{box.checked=index<count});rememberCameraSelection(id)}
async function applyCameraSelection(id){rememberCameraSelection(id);if(!pendingCameraSelection.length){log('至少选择一台相机');return}const r=await api('/api/camera-selection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({serials:pendingCameraSelection})});if(r.message)log(r.message);if(!r.ok)return;if(r.ok)pendingCameraSelection=null;await refresh()}
async function api(path,opts={}){const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),45000);try{const r=await fetch(path,{...opts,signal:controller.signal});const data=await r.json();if(!r.ok)throw new Error(data.message||(`HTTP ${r.status}`));return data}catch(e){const message=e.name==='AbortError'?`${path} 请求超时，请查看后台日志`:(e.message||String(e));log(`错误: ${message}`);return {ok:false,message}}finally{clearTimeout(timer)}}
function render(s){state=s;renderPolicySettings(s);const services=document.getElementById('services');services.innerHTML=['robot','gripper','policy'].map(k=>{const x=s.services[k]||{};const ok=x.running;return `<div class="service"><span class="dot ${ok?'ok':''}"></span><div><div class="service-name">${names[k]}</div><div class="service-detail">${ok?'运行中':'未运行'}${x.pid?' · PID '+x.pid:''}</div></div><button class="${ok?'danger':'secondary'}" ${k==='policy'&&policyStartPending?'disabled':''} onclick="${ok?`stop('${k}')`:`start('${k}')`}">${ok?'停止':'启动'}</button></div>`}).join('');
document.getElementById('collectionServices').innerHTML=['robot','gripper'].map(k=>{const x=s.services[k]||{};const ok=x.running;return `<div class="service"><span class="dot ${ok?'ok':''}"></span><div><div class="service-name">${names[k]}</div><div class="service-detail">${ok?'运行中':'未运行'}${x.pid?' · PID '+x.pid:''}</div></div><button class="${ok?'danger':'secondary'}" onclick="${ok?`stop('${k}')`:`start('${k}')`}">${ok?'停止':'启动'}</button></div>`}).join('');
const collector=s.services.collector||{};const node=s.services.gello_node||{};document.getElementById('collectorStatus').textContent=collector.running?'采集中进程运行中':(node.running?'GELLO node 已运行':'准备中');document.getElementById('collectorStart').textContent=collector.running?'采集器运行中':'启动采集器';document.getElementById('collectorStart').disabled=collector.running;
const available=s.cameras.available||[];const selected=s.cameras.selected||[];const displaySelected=(pendingCameraSelection||selected).filter(serial=>available.some(c=>c.serial===serial));const streaming=new Set((s.cameras.active||[]).map(c=>c.serial));const cams=available.filter(c=>displaySelected.includes(c.serial));
document.getElementById('cameraConfig').innerHTML=cameraConfig('mainCamera',available,displaySelected);document.getElementById('dataCameraConfig').innerHTML=cameraConfig('dataCamera',available,displaySelected);
const requiredCameras=Object.entries(s.policy_cameras||{});
const missingCameras=requiredCameras.filter(([role,serial])=>!selected.includes(serial)||!streaming.has(serial));
const camerasReady=requiredCameras.length>0&&missingCameras.length===0;
const deploying=Boolean(s.services.deploy?.running);
const servicesReady=['robot','gripper','policy'].every(k=>s.services[k]?.running);
const deploy=document.getElementById('deploy');deploy.disabled=deploying||collector.running||!(servicesReady&&camerasReady&&document.getElementById('confirm').checked);deploy.textContent=deploying?'运行中':'启动';
const cameraSummary=`当前策略需要 ${requiredCameras.length} 路相机：${requiredCameras.map(([role,serial])=>`${role} = ${serial}`).join('；')}`;
const cameraMessage=collector.running?'。请先退出数据采集器。':(missingCameras.length?`。尚未就绪：${missingCameras.map(([role])=>role).join('、')}，请选中并应用，等待画面连接。`:'。相机已就绪。');
document.getElementById('deploymentCameraHint').textContent=cameraSummary+cameraMessage;
document.getElementById('systemDot').className='dot '+(deploying?'ok':'');document.getElementById('systemText').textContent=deploying?'复现执行中':(servicesReady&&camerasReady&&!collector.running?'系统就绪':'等待硬件就绪');
const cameraRole=serial=>requiredCameras.find(([,value])=>value===serial)?.[0];
const cameraCard=c=>`<div class="cam">${streaming.has(c.serial)?`<img src="/camera/${encodeURIComponent(c.serial)}" alt="${esc(c.name)} ${esc(c.serial)}">`:`<div class="empty" style="min-height:240px">正在连接相机…</div>`}<div class="caption">${cameraRole(c.serial)?esc(cameraRole(c.serial))+' · ':''}${esc(c.name)}<br><span class="serial">${esc(c.serial)}</span></div></div>`;
document.getElementById('camCount').textContent=`${cams.length} / ${available.length} 台已选择`;
document.getElementById('cams').innerHTML=cams.length?cams.map(cameraCard).join(''):'<div class="empty">未选择相机或相机尚未上线</div>';
document.getElementById('dataCamCount').textContent=`${cams.length} / ${available.length} 台已选择`;
document.getElementById('dataCams').innerHTML=cams.length?cams.map(cameraCard).join(''):'<div class="empty">未选择相机或相机尚未上线</div>';
document.getElementById('dataCamHint').textContent=collector.running?'采集器正在独占相机，网页预览已暂停。':`当前选择 ${displaySelected.length} 台相机；点击“应用选择”后用于采集。`;
if(s.logs){const logs=s.logs.join('\n');document.getElementById('log').textContent=logs;document.getElementById('collectionLog').textContent=logs}
}
async function refresh(){try{const s=await api('/api/status');render(s);ensureEpisodePanel();const feedback=ensureCollectionFeedback();const cs=s.collector_state||{};feedback.textContent='录制状态：'+(cs.message||'未开始');feedback.className=cs.state==='recording'?'recording':'hint';document.getElementById('dataCamHint').textContent=s.services.collector?.running?'正在显示采集器实际读取的画面。':'采集器启动后将在这里显示正式采集画面。';refreshEpisodes()}catch(e){log('状态连接失败: '+e)}}
async function start(k){if(k==='policy')return startPolicy();log('正在启动 '+names[k]+'…');const r=await api('/api/start/'+k,{method:'POST'});if(r.message)log(r.message);await refresh()}
async function stop(k){const r=await api('/api/stop/'+k,{method:'POST'});if(r.message)log(r.message);await refresh()}
async function startCollector(){const task=document.getElementById('collectionTask').value.trim();const saveDir=document.getElementById('saveDir').value.trim();const gelloPort=document.getElementById('gelloPort').value.trim();const autoStart=document.getElementById('autoStart').checked;log('正在启动 LeRobot 采集器…');const r=await api('/api/start-collector',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task,save_dir:saveDir,gello_port:gelloPort,auto_start:autoStart})});if(r.message)log(r.message);if(!r.ok)return;await refresh()}
async function recalculateOffsets(){const port=document.getElementById('gelloPort').value.trim();log('正在读取 GELLO 当前姿态并重新计算 offsets，最多等待 30 秒，请保持 leader 不动…');const r=await api('/api/recalculate-offsets',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gello_port:port})});if(r.message)log(r.message);if(!r.ok)return;await refresh()}
async function chooseDirectory(){const r=await api('/api/choose-directory',{method:'POST'});if(r.path){document.getElementById('saveDir').value=r.path;log('已选择保存目录: '+r.path)}else if(r.message)log(r.message)}
async function collectorCommand(command){const r=await api('/api/collector-command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command})});if(r.message)log(r.message);if(!r.ok)return;await refresh()}
document.addEventListener('keydown',e=>{if(!document.getElementById('collectionTab').classList.contains('active'))return;if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))return;const command={r:'r',s:'s',d:'d',q:'q'}[e.key.toLowerCase()];if(!command||e.repeat)return;e.preventDefault();collectorCommand(command)});
async function startDeploy(){if(!document.getElementById('confirm').checked)return;const prompt=document.getElementById('prompt').value.trim();const r=await api('/api/start-deploy',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt})});log(r.message);await refresh()}
async function stopAll(){const r=await api('/api/stop-all',{method:'POST'});log(r.message);await refresh()}
document.getElementById('policyDir').addEventListener('input',savePolicySettings);document.getElementById('policyConfig').addEventListener('input',savePolicySettings);
document.getElementById('prompt').value=localStorage.getItem('frankaPrompt')||'pick up the object and place it into the box';document.getElementById('prompt').addEventListener('input',e=>{localStorage.setItem('frankaPrompt',e.target.value);refresh()});document.getElementById('confirm').addEventListener('change',refresh);refresh();setInterval(refresh,2000);
</script></body></html>"""


def recalculate_and_write_offsets(port):
    """Run the documented offset calibration and update the matching port block."""
    if not Path(port).exists():
        raise FileNotFoundError(f"GELLO 串口不存在: {port}")

    command = [
        f"{POLY_ENV}/bin/python",
        f"{GELLO_ROOT}/scripts/gello_get_offset.py",
        "--start-joints", *[str(x) for x in OFFSET_START_JOINTS],
        "--joint-signs", *[str(x) for x in OFFSET_JOINT_SIGNS],
        "--port", port,
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{POLY_ENV}/lib:" + env.get("LD_LIBRARY_PATH", "")
    result = subprocess.run(
        command, cwd=GELLO_ROOT, env=env, capture_output=True, text=True, timeout=30
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    with PROCESSES.lock:
        PROCESSES.logs.extend(f"[gello_offset] {line}" for line in output.splitlines())
        PROCESSES.logs = PROCESSES.logs[-200:]
    if result.returncode != 0:
        raise RuntimeError("offset 计算失败:\n" + output[-3000:])
    if "comm failed" in output.lower() or "-3001" in output:
        raise RuntimeError(
            "GELLO 舵机总线无响应（comm failed / -3001），未写回 offsets。"
            "请确认只有一个程序占用串口、GELLO 已上电，并检查串口权限和波特率。"
        )

    match = re.search(r"best offsets function of pi:\s*\[([^\]]+)\]", output)
    if not match:
        raise RuntimeError("未在 gello_get_offset.py 输出中找到 best offsets:\n" + output[-2000:])
    multipliers = []
    for token in match.group(1).split(","):
        number = re.search(r"[-+]?\d+", token)
        if not number:
            raise RuntimeError(f"无法解析 offset: {token.strip()}")
        multipliers.append(int(number.group(0)))
    if len(multipliers) != 7:
        raise RuntimeError(f"解析到 {len(multipliers)} 个 offsets，预期 7 个")

    source_path = Path(GELLO_AGENT_FILE)
    source = source_path.read_text()
    port_block = re.compile(
        r'("' + re.escape(port) + r'"\s*:\s*DynamixelRobotConfig\(.*?'
        r'joint_offsets\s*=\s*)\((.*?)\)(\s*,\s*joint_signs\s*=)',
        re.DOTALL,
    )
    def replace_offset_multipliers(match):
        prefix, body, suffix = match.groups()
        # Only replace the leading multiplier. Preserve `/ 2` and any tail,
        # especially the final `- np.pi / 4` calibration convention.
        pattern = re.compile(r"(?m)^(\s*)([-+]?\d+)\s*(\*\s*np\.pi\s*/\s*2)")
        index = 0

        def replace_one(item):
            nonlocal index
            if index >= len(multipliers):
                return item.group(0)
            value = multipliers[index]
            index += 1
            return f"{item.group(1)}{value} {item.group(3)}"

        new_body = pattern.sub(replace_one, body)
        if index != len(multipliers):
            raise RuntimeError(f"joint_offsets 中只找到 {index} 个可替换倍数，预期 7 个")
        return prefix + "(" + new_body + ")" + suffix

    updated, count = port_block.subn(replace_offset_multipliers, source, count=1)
    if count != 1:
        raise RuntimeError(f"gello_agent.py 中没有找到串口配置: {port}")

    backup = source_path.with_suffix(source_path.suffix + ".bak")
    shutil.copy2(source_path, backup)
    source_path.write_text(updated)
    return multipliers, output


class CameraManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.devices = {}
        self.latest = {}
        self.selected = {CAMERA_SERIALS_BY_ROLE[role] for role in DEFAULT_CAMERA_ROLES}
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _enumerate(self):
        context = rs.context()
        found = []
        devices = context.query_devices()
        for i in range(devices.size()):
            d = devices[i]
            try:
                serial = d.get_info(rs.camera_info.serial_number)
                name = d.get_info(rs.camera_info.name)
                found.append((serial, name))
            except RuntimeError:
                continue
        return found

    def _loop(self):
        pipelines = {}
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                for p, _ in pipelines.values():
                    try: p.stop()
                    except Exception: pass
                pipelines.clear()
                with self.lock: self.latest.clear()
                time.sleep(.1)
                continue
            try:
                found = self._enumerate()
                found_map = dict(found)
                with self.lock:
                    if self.selected is None:
                        self.selected = set(found_map)
                    wanted = set(self.selected) & set(found_map)
                for serial, name in found:
                    if serial in wanted and serial not in pipelines:
                        try:
                            p = rs.pipeline()
                            c = rs.config()
                            c.enable_device(serial)
                            c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
                            p.start(c)
                            pipelines[serial] = (p, name)
                        except Exception:
                            pass
                for serial in list(pipelines):
                    if serial not in wanted:
                        try: pipelines.pop(serial)[0].stop()
                        except Exception: pass
                for serial, (p, _) in list(pipelines.items()):
                    try:
                        frame = p.wait_for_frames(100).get_color_frame()
                        if frame:
                            image = np.asanyarray(frame.get_data()).copy()
                            cv2.putText(image, serial, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 2, cv2.LINE_AA)
                            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 82])
                            if ok:
                                with self.lock: self.latest[serial] = encoded.tobytes()
                    except Exception:
                        pass
                with self.lock:
                    # Only report cameras with a successfully started stream
                    # as active; selected-but-busy cameras remain visible in
                    # the UI as a connection placeholder.
                    self.devices = {
                        s: found_map[s] for s in pipelines if s in found_map
                    }
            except Exception:
                with self.lock: self.devices = {}
            time.sleep(.03)
        for p, _ in pipelines.values():
            try: p.stop()
            except Exception: pass

    def pause(self):
        """Release camera pipelines so the deployment process can own them."""
        self.pause_event.set()
        # Give the capture thread time to stop every active RealSense pipeline.
        time.sleep(0.25)

    def resume(self):
        """Resume the dashboard preview after deployment stops."""
        self.pause_event.clear()

    def set_selection(self, serials):
        available = {serial for serial, _ in self._enumerate()}
        selected = {str(serial) for serial in serials} & available
        if not selected:
            return False, "至少选择一台当前可用的相机"
        with self.lock:
            self.selected = selected
        return True, f"已选择 {len(selected)} 台相机"

    def status(self):
        available = self._enumerate()
        with self.lock:
            selected = sorted(self.selected or {serial for serial, _ in available})
            active = [{"serial": s, "name": n} for s, n in self.devices.items()]
        with PROCESSES.lock:
            collector_running = 'collector' in PROCESSES.procs and PROCESSES.procs['collector'].poll() is None
        if collector_running:
            names = dict(available)
            active = [{"serial": s, "name": names.get(s, s)} for s in selected if s in names]
        return {"available": [{"serial": s, "name": n} for s, n in available], "selected": selected, "active": active}

    def jpeg(self, serial):
        with self.lock: return self.latest.get(serial)


def validate_policy_settings(checkpoint, config):
    """Validate local checkpoint arguments without importing or loading the model."""
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError('请填写 checkpoint 目录')
    if not isinstance(config, str) or not config.strip():
        raise ValueError('请填写与训练时一致的 policy 配置名')
    if '\x00' in checkpoint or '\x00' in config:
        raise ValueError('checkpoint 目录和配置名不能包含空字符')
    path = Path(checkpoint.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError('checkpoint 请使用本机绝对路径，例如 /home/lsk/.../29999')
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f'checkpoint 目录不存在: {path}')
    if not (path / 'params').is_dir() and not (path / 'model.safetensors').is_file():
        raise ValueError('请选择包含 params/ 或 model.safetensors 的 checkpoint 目录（如 29999）')
    return {'checkpoint': str(path), 'config': config.strip()}


class ProcessManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.procs = {}
        self.logs = []
        self.collector_state = {'state': 'idle', 'message': '准备中'}
        self.policy_settings = {'checkpoint': POLICY_DIR, 'config': POLICY_CONFIG}
        self.policy_cameras = {role: CAMERA_SERIALS_BY_ROLE[role] for role in DEFAULT_CAMERA_ROLES}

    def _reader(self, key, proc):
        for line in iter(proc.stdout.readline, ''):
            message = f"[{key}] {line.rstrip()}"
            with self.lock:
                self.logs.append(message)
                if key == 'collector':
                    if 'Recording episode' in line:
                        self.collector_state = {'state': 'recording', 'message': '正在录制当前 episode'}
                    elif 'Stopping episode and writing' in line:
                        self.collector_state = {'state': 'saving', 'message': '正在停止并保存 MP4 与 parquet'}
                    elif 'Parquet saved:' in line:
                        self.collector_state = {'state': 'saved', 'message': '当前 episode 已保存完成'}
                    elif 'Current in-memory episode discarded' in line:
                        self.collector_state = {'state': 'idle', 'message': '当前 episode 已丢弃'}
            print(message, flush=True)
        proc.stdout.close()

    def _watch(self, key, proc):
        code = proc.wait()
        with self.lock:
            if self.procs.get(key) is proc: self.procs.pop(key, None)
            self.logs.append(f"[{key}] process exited with code {code}")
            if key == 'collector' and self.collector_state['state'] not in ('saved', 'idle'):
                self.collector_state = {'state': 'idle', 'message': '采集器已退出'}
        if key in ('deploy', 'collector'):
            CAMERAS.resume()

    def send(self, key, command):
        with self.lock:
            proc = self.procs.get(key)
        if not proc or proc.poll() is not None or proc.stdin is None:
            return False, f"{names_for(key)} 未运行"
        try:
            proc.stdin.write(command + "\n")
            proc.stdin.flush()
            return True, f"已发送采集命令: {command}"
        except (BrokenPipeError, OSError) as exc:
            return False, f"发送采集命令失败: {exc}"

    def start(self, key, env_extra=None, retry=True):
        with self.lock:
            if key in self.procs and self.procs[key].poll() is None:
                return False, f"{names_for(key)} 已在运行，请先停止后再修改设置"
            label, command = COMMANDS[key]
            env = os.environ.copy()
            if key == 'policy':
                env.update(POLICY_DIR=self.policy_settings['checkpoint'], POLICY_CONFIG=self.policy_settings['config'])
            env.update(env_extra or {})
            if key == 'policy':
                try:
                    settings = validate_policy_settings(env['POLICY_DIR'], env['POLICY_CONFIG'])
                    policy_cameras = policy_camera_mapping(settings['config'])
                except (ValueError, OSError, RuntimeError, SyntaxError) as exc:
                    self.logs.append(f"[policy] 启动前检查失败: {exc}")
                    return False, str(exc)
                # Quoted shell variables pass user input as arguments, never as shell code.
                env.update(POLICY_DIR=settings['checkpoint'], POLICY_CONFIG=settings['config'])
            proc = subprocess.Popen(['/bin/bash','-lc',command], cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            self.procs[key] = proc
            if key == 'policy':
                self.policy_settings = settings
                self.policy_cameras = policy_cameras
                self.logs.append(f"[policy] config: {settings['config']} | checkpoint: {settings['checkpoint']}")
        threading.Thread(target=self._reader,args=(key,proc),daemon=True).start()
        threading.Thread(target=self._watch,args=(key,proc),daemon=True).start()
        time.sleep(1.5)
        if proc.poll() is not None and key in ('robot','gripper'):
            with self.lock: recent='\n'.join(self.logs[-20:])
            if 'port unavailable' in recent.lower() and retry:
                subprocess.run(['pkill','-9','run_server'], check=False)
                time.sleep(.5)
                return self.start(key, env_extra, retry=False)
        if key == 'policy':
            if proc.poll() is not None:
                return False, 'Pi05 policy server 启动失败，请查看日志中的配置或模型加载错误'
            return True, f"{label} 启动命令已提交 | config: {settings['config']} | checkpoint: {settings['checkpoint']}"
        return True, f"{label} 启动命令已提交"

    def stop(self, key):
        with self.lock: proc=self.procs.get(key)
        if not proc or proc.poll() is not None: return False, f"{names_for(key)} 未运行"
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=3)
        except Exception:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        return True, f"{names_for(key)} 已停止"

    def stop_all(self):
        """Stop every process group launched by this dashboard instance."""
        # Stop dependents first so GELLO/collector do not keep using the robot.
        messages = []
        for key in ('collector', 'gello_node', 'deploy', 'policy', 'gripper', 'robot'):
            ok, message = self.stop(key)
            if ok:
                messages.append(message)
        return messages

    def status(self):
        with self.lock:
            services={k:{'running':p.poll() is None,'pid':p.pid} for k,p in self.procs.items()}
            logs=self.logs[-80:]
            collector_state = dict(self.collector_state)
            policy_settings = dict(self.policy_settings)
            policy_cameras = dict(self.policy_cameras)
        return {'services':services,'logs':logs,'collector_state':collector_state,'policy_settings':policy_settings,'policy_cameras':policy_cameras}


def names_for(key):
    return {'robot':'Robot server','gripper':'Franka gripper','policy':'Pi05 policy server','deploy':'Deployment','gello_move':'Move FR3 to GELLO start','gello_node':'GELLO FR3 robot node','collector':'LeRobot data collector'}.get(key,key)


CAMERAS = CameraManager()
PROCESSES = ProcessManager()


def _episode_root(save_dir):
    root = Path(save_dir).expanduser().resolve()
    if root == Path('/') or root == Path.home():
        raise ValueError('不允许使用根目录或 home 目录作为数据目录')
    return root


def list_episodes(save_dir):
    root = _episode_root(save_dir)
    data_dir = root / 'data'
    episodes = []
    for parquet in sorted(data_dir.glob('episode_*.parquet')):
        stat = parquet.stat()
        episodes.append({
            'id': parquet.stem,
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'videos': len(list((root / 'videos').glob(f'*_{parquet.stem}.mp4'))),
        })
    return episodes


def delete_episodes(save_dir, episode_ids):
    root = _episode_root(save_dir)
    allowed = {p['id'] for p in list_episodes(root)}
    deleted = []
    for episode_id in {str(value) for value in episode_ids} & allowed:
        targets = [root / 'data' / f'{episode_id}.parquet']
        targets.extend((root / 'videos').glob(f'*_{episode_id}.mp4'))
        for target in targets:
            resolved = target.resolve()
            if root not in resolved.parents:
                continue
            if resolved.is_file():
                resolved.unlink()
        deleted.append(episode_id)
    return deleted


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def json(self, data, status=HTTPStatus.OK):
        payload=json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/':
            data=PAGE.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data); return
        if path=='/api/status':
            data=PROCESSES.status(); data['cameras']=CAMERAS.status(); self.json(data); return
        if path.startswith('/camera/'):
            serial=path.split('/',2)[2]; self.send_response(200); self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame'); self.end_headers()
            try:
                while True:
                    frame=CAMERAS.jpeg(serial)
                    if not frame:
                        preview_path = COLLECTION_PREVIEW_DIR / f'{serial}.jpg'
                        try:
                            if preview_path.is_file():
                                frame = preview_path.read_bytes()
                        except OSError:
                            frame = None
                    if frame: self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(frame)).encode()+b'\r\n\r\n'+frame+b'\r\n'); self.wfile.flush()
                    time.sleep(.06)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        self.send_error(404)
    def do_POST(self):
        path=urlparse(self.path).path
        if path.startswith('/api/start/'):
            key=path.rsplit('/',1)[1]
            if key not in COMMANDS: return self.json({'message':'未知服务'},400)
            if key == 'policy':
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    body = json.loads(self.rfile.read(length) or b'{}')
                    if not isinstance(body, dict):
                        raise ValueError('策略设置必须是 JSON 对象')
                    env = {}
                    if 'checkpoint' in body:
                        env['POLICY_DIR'] = body['checkpoint']
                    if 'config' in body:
                        env['POLICY_CONFIG'] = body['config']
                    ok, msg = PROCESSES.start(key, env)
                except (ValueError, OSError) as exc:
                    return self.json({'ok': False, 'message': str(exc)}, HTTPStatus.BAD_REQUEST)
                return self.json({'ok': ok, 'message': msg}, HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST)
            ok,msg=PROCESSES.start(key); return self.json({'ok':ok,'message':msg})
        if path.startswith('/api/stop/'):
            key=path.rsplit('/',1)[1]; ok,msg=PROCESSES.stop(key); return self.json({'ok':ok,'message':msg})
        if path=='/api/start-deploy':
            length=int(self.headers.get('Content-Length','0')); body=json.loads(self.rfile.read(length) or b'{}'); prompt=body.get('prompt','').strip() or DEFAULT_PROMPT
            cameras=CAMERAS.status(); process_state=PROCESSES.status(); running=process_state['services']
            if running.get('deploy',{}).get('running'): return self.json({'ok':False,'message':'推理客户端已在运行'},409)
            if running.get('collector',{}).get('running'): return self.json({'ok':False,'message':'请先退出数据采集器，释放相机后再启动推理'},409)
            if not all(running.get(k,{}).get('running') for k in ('robot','gripper','policy')): return self.json({'ok':False,'message':'请先启动 Robot、Gripper 和 Pi05 policy'},409)
            required = process_state['policy_cameras']
            active = {c['serial'] for c in cameras['active']} & set(cameras['selected'])
            missing = [f'{role} ({serial})' for role, serial in required.items() if serial not in active]
            if missing: return self.json({'ok':False,'message':'训练所需相机尚未就绪: '+', '.join(missing)},409)
            CAMERAS.pause()
            serials=','.join(required.values())
            ok,msg=PROCESSES.start('deploy', {'TASK_PROMPT':prompt, 'SHOW_CAMERA_PREVIEW':'0', 'CAMERA_SERIALS':serials})
            if not ok: CAMERAS.resume()
            return self.json({'ok':ok,'message':msg+f' | prompt: {prompt} | cameras: {required}'})
        if path=='/api/camera-selection':
            length=int(self.headers.get('Content-Length','0')); body=json.loads(self.rfile.read(length) or b'{}')
            ok, message=CAMERAS.set_selection(body.get('serials', []))
            if ok:
                serials = ', '.join(sorted(str(serial) for serial in body.get('serials', [])))
                with PROCESSES.lock:
                    PROCESSES.logs.append(f"[camera] {message}: {serials}")
                    PROCESSES.logs = PROCESSES.logs[-200:]
            return self.json({'ok':ok,'message':message}, HTTPStatus.OK if ok else HTTPStatus.CONFLICT)
        if path=='/api/recalculate-offsets':
            length=int(self.headers.get('Content-Length','0')); body=json.loads(self.rfile.read(length) or b'{}')
            port=body.get('gello_port','').strip() or GELLO_PORT
            if PROCESSES.status()['services'].get('collector',{}).get('running'):
                return self.json({'ok':False,'message':'请先退出当前数据采集器，再重新计算 offsets'},409)
            try:
                multipliers, _ = recalculate_and_write_offsets(port)
                return self.json({'ok':True,'message':f'offsets 已写回 {GELLO_AGENT_FILE}（已备份为 .bak）: {multipliers}'})
            except Exception as exc:
                return self.json({'ok':False,'message':str(exc)},500)
        if path=='/api/choose-directory':
            if not shutil.which('zenity'):
                return self.json({'ok':False,'message':'系统未安装 zenity，请手动输入保存目录'},500)
            try:
                result=subprocess.run(
                    ['zenity','--file-selection','--directory','--title=选择数据保存目录'],
                    capture_output=True, text=True, timeout=300,
                )
                selected=result.stdout.strip()
                if result.returncode == 0 and selected:
                    return self.json({'ok':True,'path':selected,'message':'目录选择成功'})
                return self.json({'ok':False,'message':'已取消目录选择'})
            except subprocess.TimeoutExpired:
                return self.json({'ok':False,'message':'目录选择超时'},504)
            except Exception as exc:
                return self.json({'ok':False,'message':f'无法打开目录选择器: {exc}'},500)
        if path=='/api/episodes':
            length=int(self.headers.get('Content-Length','0')); body=json.loads(self.rfile.read(length) or b'{}')
            try:
                return self.json({'ok':True,'episodes':list_episodes(body.get('save_dir','') or DEFAULT_SAVE_DIR)})
            except Exception as exc:
                return self.json({'ok':False,'message':f'读取 episodes 失败: {exc}'},400)
        if path=='/api/delete-episodes':
            length=int(self.headers.get('Content-Length','0')); body=json.loads(self.rfile.read(length) or b'{}')
            try:
                deleted=delete_episodes(body.get('save_dir','') or DEFAULT_SAVE_DIR, body.get('episode_ids', []))
                message=f'已删除 {len(deleted)} 条 episode' if deleted else '没有删除任何 episode'
                with PROCESSES.lock:
                    PROCESSES.logs.append(f'[episodes] {message}')
                    PROCESSES.logs = PROCESSES.logs[-200:]
                return self.json({'ok':True,'message':message,'deleted':deleted})
            except Exception as exc:
                return self.json({'ok':False,'message':f'删除 episodes 失败: {exc}'},400)
        if path=='/api/start-collector':
            length=int(self.headers.get('Content-Length','0')); body=json.loads(self.rfile.read(length) or b'{}')
            services=PROCESSES.status()['services']; cameras=CAMERAS.status()
            if len(cameras['active'])!=2: return self.json({'ok':False,'message':f'数据采集需要选择 2 台相机，当前选择 {len(cameras["active"])} 台'},409)
            if not services.get('gello_node',{}).get('running'): return self.json({'ok':False,'message':'请先启动 GELLO FR3 robot node'},409)
            task=body.get('task','').strip() or DEFAULT_COLLECTION_TASK
            save_dir=body.get('save_dir','').strip() or DEFAULT_SAVE_DIR
            gello_port=body.get('gello_port','').strip() or GELLO_PORT
            serials=' '.join(c['serial'] for c in cameras['active'])
            env={'COLLECTION_TASK':task,'COLLECTION_SAVE_DIR':save_dir,'GELLO_PORT':gello_port,'COLLECTION_CAMERA_SERIALS':serials,'COLLECTION_PREVIEW_DIR':str(COLLECTION_PREVIEW_DIR),'COLLECTION_AUTOSTART':'1' if body.get('auto_start') else '0'}
            CAMERAS.pause(); ok,msg=PROCESSES.start('collector',env)
            if not ok: CAMERAS.resume()
            return self.json({'ok':ok,'message':msg+f' | task: {task}'})
        if path=='/api/collector-command':
            length=int(self.headers.get('Content-Length','0')); body=json.loads(self.rfile.read(length) or b'{}'); command=body.get('command')
            if command not in ('r','s','d','q','h'): return self.json({'ok':False,'message':'无效采集命令'},400)
            ok,msg=PROCESSES.send('collector',command)
            if ok:
                messages = {
                    'r': ('recording', '已开始采集：正在录制当前 episode'),
                    's': ('saving', '已停止录制：正在保存 MP4 与 parquet，请稍候'),
                    'd': ('idle', '已丢弃当前 episode'),
                    'q': ('idle', '正在退出采集器'),
                    'h': ('idle', '已发送帮助命令'),
                }
                state, message = messages[command]
                with PROCESSES.lock:
                    PROCESSES.collector_state = {'state': state, 'message': message}
                    PROCESSES.logs.append(f'[collector-ui] {message}')
                    PROCESSES.logs = PROCESSES.logs[-200:]
                msg = message
            return self.json({'ok':ok,'message':msg})
        if path=='/api/stop-all':
            for key in ('collector','gello_node','deploy','policy','gripper','robot'): PROCESSES.stop(key)
            return self.json({'ok':True,'message':'已发送停止信号给全部服务'})
        self.send_error(404)


if __name__ == '__main__':
    print(f'Franka Pi05 dashboard: http://{HOST}:{PORT}', flush=True)
    try: ThreadingHTTPServer((HOST,PORT),Handler).serve_forever()
    except KeyboardInterrupt:
        print('正在关闭 dashboard 启动的全部 Franka 相关进程...', flush=True)
        for message in PROCESSES.stop_all():
            print(message, flush=True)
    finally:
        CAMERAS.stop_event.set()
