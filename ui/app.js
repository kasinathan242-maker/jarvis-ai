/**
 * JARVIS Lite — HUD Command Center Engine.
 * Manages WebSocket streaming, telemetry gauges, interactive security approvals,
 * and emergency halt triggers.
 */

(function () {
  const API_BASE = window.location.origin;
  const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/v1/ws/chat`;

  // State
  let ws = null;
  let authToken = localStorage.getItem('jarvis_token') || 'local_development_token';
  let activeApprovalId = null;
  let countdownTimer = null;
  let isWakeWordActive = false;

  // DOM Elements
  const chatFeed = document.getElementById('chat-feed');
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const btnEmergency = document.getElementById('btn-emergency-halt');
  const btnTtsStop = document.getElementById('btn-tts-stop');
  const btnWakeToggle = document.getElementById('btn-wakeword-toggle');
  const btnMicCore = document.getElementById('btn-mic-core');
  const micVisualizer = document.getElementById('mic-visualizer');
  const wakeStatusPill = document.getElementById('wake-status-pill');
  const auditList = document.getElementById('audit-list');
  const auditCount = document.getElementById('audit-count');
  const alertsFeed = document.getElementById('alerts-feed');

  // Modal Elements
  const modal = document.getElementById('approval-modal');
  const modalToolName = document.getElementById('modal-tool-name');
  const modalRiskBadge = document.getElementById('modal-risk-badge');
  const modalParamsJson = document.getElementById('modal-params-json');
  const modalCountdown = document.getElementById('modal-countdown');
  const btnApprove = document.getElementById('btn-approve-action');
  const btnReject = document.getElementById('btn-reject-action');

  // Gauge Elements
  const cpuBar = document.querySelector('.cpu-bar');
  const valCpu = document.getElementById('val-cpu');
  const subCpu = document.getElementById('sub-cpu');

  const ramBar = document.querySelector('.ram-bar');
  const valRam = document.getElementById('val-ram');
  const subRam = document.getElementById('sub-ram');

  const vramBar = document.querySelector('.vram-bar');
  const valVram = document.getElementById('val-vram');
  const subVram = document.getElementById('sub-vram');

  const diskBar = document.querySelector('.disk-bar');
  const valDisk = document.getElementById('val-disk');
  const subDisk = document.getElementById('sub-disk');

  const valLatency = document.getElementById('val-latency');
  const valNetworkIo = document.getElementById('val-network-io');
  const valBattery = document.getElementById('val-battery');

  // Helper: Set SVG Circular Gauge percentage (0 - 100)
  function setGauge(barElement, valElement, percent, colorOverride = null) {
    if (!barElement || !valElement) return;
    const clamped = Math.max(0, Math.min(100, percent));
    const circumference = 264; // 2 * PI * 42
    const offset = circumference - (clamped / 100) * circumference;
    barElement.style.strokeDashoffset = offset;
    valElement.textContent = `${Math.round(clamped)}%`;
    if (colorOverride) {
      barElement.style.stroke = colorOverride;
    }
  }

  // Fetch Telemetry Snapshot
  async function updateTelemetry() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/telemetry/health`, {
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      if (!res.ok) return;
      const data = await res.json();
      const r = data.resources;
      const n = data.network;

      // CPU
      setGauge(cpuBar, valCpu, r.cpu_percent);
      subCpu.textContent = `${r.cpu_count_logical} Cores`;

      // RAM
      setGauge(ramBar, valRam, r.ram_percent);
      subRam.textContent = `${(r.ram_used_mb / 1024).toFixed(1)} / ${(r.ram_total_mb / 1024).toFixed(1)} GB`;

      // GPU VRAM (RTX 3050 Laptop GPU)
      if (r.gpu) {
        setGauge(vramBar, valVram, r.gpu.vram_percent);
        subVram.textContent = `${(r.gpu.used_vram_mb / 1024).toFixed(2)} / ${(r.gpu.total_vram_mb / 1024).toFixed(2)} GB`;
        // High alert color if VRAM > 5.5 GB (90%+)
        if (r.gpu.used_vram_mb >= 5500) {
          vramBar.style.stroke = '#FF3366';
        }
      }

      // Disk
      setGauge(diskBar, valDisk, r.disk_percent);
      subDisk.textContent = `${r.disk_free_gb} GB Free`;

      // Network & Battery
      valLatency.textContent = n.latency_ms !== null ? `${n.latency_ms} ms` : 'Offline';
      valNetworkIo.innerHTML = `&uarr; ${(n.bytes_sent_per_sec / 1024).toFixed(1)} KB/s &bull; &darr; ${(n.bytes_recv_per_sec / 1024).toFixed(1)} KB/s`;

      if (r.battery) {
        valBattery.textContent = `${r.battery.percent}% (${r.battery.is_plugged ? 'Plugged' : 'Battery'})`;
      }

      // Alerts
      if (data.alerts && data.alerts.length > 0) {
        alertsFeed.innerHTML = data.alerts.map(a => `
          <div class="alert-item-critical">
            <strong>${a.severity.toUpperCase()}:</strong> ${a.message}
          </div>
        `).join('');
      } else {
        alertsFeed.innerHTML = '<div class="no-alerts">All safety thresholds normal. VRAM &lt; 5.5 GB.</div>';
      }
    } catch (e) {
      // Server starting up or network blip
    }
  }

  // Fetch Audit Logs
  async function updateAuditLogs() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/audit?limit=15`, {
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      if (!res.ok) return;
      const logs = await res.json();
      auditCount.textContent = `${logs.length} CALLS`;

      if (logs.length > 0) {
        auditList.innerHTML = logs.map(item => {
          const isPassed = item.execution_status === 'SUCCESS' && item.verification_passed;
          const isBlocked = item.execution_status === 'BLOCKED';
          const isRejected = item.execution_status === 'REJECTED';
          const statusClass = isBlocked ? 'blocked' : (isRejected ? 'rejected' : 'passed');
          const statusText = isBlocked ? 'BLOCKED' : (isRejected ? 'REJECTED' : 'VERIFIED');

          return `
            <div class="audit-item ${statusClass}">
              <div class="audit-item-header">
                <span class="audit-tool-name">${item.tool_name}</span>
                <span class="audit-status ${statusClass}">${statusText}</span>
              </div>
              <div class="audit-params">${JSON.stringify(item.parameters)}</div>
            </div>
          `;
        }).join('');
      }
    } catch (e) {
      // Silent error
    }
  }

  // Poll for Active Approvals
  async function checkPendingApprovals() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/security/approvals`, {
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      if (!res.ok) return;
      const approvals = await res.json();

      if (approvals.length > 0 && !activeApprovalId) {
        const ticket = approvals[0];
        showApprovalModal(ticket);
      } else if (approvals.length === 0 && activeApprovalId) {
        hideApprovalModal();
      }
    } catch (e) {
      // Ignore
    }
  }

  function showApprovalModal(ticket) {
    activeApprovalId = ticket.request_id;
    modalToolName.textContent = `Authorize Tool: ${ticket.tool_name}`;
    modalRiskBadge.textContent = `${ticket.risk_level} LEVEL ACTION`;
    modalRiskBadge.style.color = ticket.risk_level === 'DANGEROUS' ? '#FF3366' : '#FFB703';
    modalParamsJson.textContent = JSON.stringify(ticket.parameters, null, 2);

    let secondsLeft = Math.max(1, Math.round(ticket.timeout_seconds - (Date.now() / 1000 - ticket.created_at)));
    modalCountdown.textContent = `${secondsLeft}s`;

    if (countdownTimer) clearInterval(countdownTimer);
    countdownTimer = setInterval(() => {
      secondsLeft -= 1;
      modalCountdown.textContent = `${secondsLeft}s`;
      if (secondsLeft <= 0) {
        clearInterval(countdownTimer);
        hideApprovalModal();
      }
    }, 1000);

    modal.classList.add('show');
  }

  function hideApprovalModal() {
    activeApprovalId = null;
    if (countdownTimer) clearInterval(countdownTimer);
    modal.classList.remove('show');
  }

  // Resolve Ticket
  async function resolveTicket(approved) {
    if (!activeApprovalId) return;
    try {
      await fetch(`${API_BASE}/api/v1/security/approvals/${activeApprovalId}/decision`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({ approved: approved, source: 'ui' })
      });
      appendMessage('system', approved ? 'Action authorized by user.' : 'Action rejected by user. Nothing executed.');
      hideApprovalModal();
      updateAuditLogs();
    } catch (e) {
      console.error('Failed to resolve approval:', e);
    }
  }

  btnApprove.addEventListener('click', () => resolveTicket(true));
  btnReject.addEventListener('click', () => resolveTicket(false));

  // Append Message to Chat Feed
  function appendMessage(sender, text, toolCall = null) {
    const msgEl = document.createElement('div');
    msgEl.className = `message ${sender}-msg`;
    const avatar = sender === 'user' ? 'YOU' : (sender === 'ai' ? 'AI' : 'SYS');
    const headerTitle = sender === 'user' ? 'COMMAND' : (sender === 'ai' ? 'JARVIS ASSISTANT' : 'SYSTEM STATUS');
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    let toolMarkup = '';
    if (toolCall) {
      toolMarkup = `<div class="tool-chip verified">&#x2714; Tool verified: ${toolCall}</div>`;
    }

    msgEl.innerHTML = `
      <div class="msg-avatar">${avatar}</div>
      <div class="msg-content">
        <div class="msg-header">${headerTitle} <span class="msg-time">${timeStr}</span></div>
        <div class="msg-body">${escapeHtml(text)}</div>
        ${toolMarkup}
      </div>
    `;

    chatFeed.appendChild(msgEl);
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Submit Command
  async function handleCommandSubmit(e) {
    if (e) e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    chatInput.value = '';
    appendMessage('user', query);

    try {
      const res = await fetch(`${API_BASE}/api/v1/command`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({ text: query, play_speech: false })
      });
      const data = await res.json();
      appendMessage('ai', data.response);
      updateAuditLogs();
    } catch (err) {
      appendMessage('system', `Error dispatching command: ${err.message}`);
    }
  }

  chatForm.addEventListener('submit', handleCommandSubmit);

  // Quick Chips
  document.querySelectorAll('.chip-prompt').forEach(chip => {
    chip.addEventListener('click', () => {
      chatInput.value = chip.getAttribute('data-prompt');
      handleCommandSubmit();
    });
  });

  // Emergency Halt Button
  btnEmergency.addEventListener('click', async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/emergency_stop`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      const data = await res.json();
      appendMessage('system', `EMERGENCY STOP HALT EXECUTED: Cancelled ${data.tasks_cancelled} task(s), ${data.approvals_cancelled} approval(s) in ${data.duration_ms} ms.`);
      hideApprovalModal();
      updateAuditLogs();
    } catch (err) {
      appendMessage('system', `Emergency halt error: ${err.message}`);
    }
  });

  // Stop TTS button
  btnTtsStop.addEventListener('click', async () => {
    try {
      await fetch(`${API_BASE}/api/v1/emergency_stop`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      appendMessage('system', 'Speech audio playback stopped.');
    } catch (e) {
      // Ignore
    }
  });

  // Toggle Wake Word
  btnWakeToggle.addEventListener('click', async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/voice/wake_word/toggle`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      const data = await res.json();
      isWakeWordActive = data.is_listening;
      updateWakeUi();
    } catch (e) {
      console.error(e);
    }
  });

  function updateWakeUi() {
    if (isWakeWordActive) {
      micVisualizer.classList.add('active');
      wakeStatusPill.textContent = 'LISTENING';
      wakeStatusPill.style.color = '#00F2FE';
    } else {
      micVisualizer.classList.remove('active');
      wakeStatusPill.textContent = 'STANDBY';
      wakeStatusPill.style.color = '#8E9AA8';
    }
  }

  // Clear Chat
  document.getElementById('btn-clear-chat').addEventListener('click', () => {
    chatFeed.innerHTML = '';
    appendMessage('system', 'Console cleared.');
  });

  // Polling Loops (every 2s for metrics, every 1s for approvals)
  setInterval(updateTelemetry, 2000);
  setInterval(updateAuditLogs, 3000);
  setInterval(checkPendingApprovals, 1000);

  // Initial loads
  updateTelemetry();
  updateAuditLogs();
  checkPendingApprovals();
})();
