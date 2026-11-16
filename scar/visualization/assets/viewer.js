/* Offline, read-only Cytoscape view. All graph records remain in the part cache. */
(() => {
  'use strict';
  const manifest = window.SCAR_GRAPH_MANIFEST || {};
  const parts = new Map(), pending = new Map(), views = new Map();
  const $ = id => document.getElementById(id);
  const relationNames = {data:'数据', control:'控制', state:'状态', call:'调用', structure:'结构', resource:'资源', other:'其他'};
  const relationColors = {data:'#79adc8', control:'#caab76', state:'#b497d0', call:'#69bca8', structure:'#63788f', resource:'#86b99a', other:'#8c97a7'};
  const sourceKey = JSON.stringify([manifest.source || null, manifest.root, manifest.resource_root || null]);
  let keyHash = 2166136261;
  for (const char of sourceKey) keyHash = Math.imul(keyHash ^ char.charCodeAt(0), 16777619);
  const storageKey = 'scar.graph.layout.v1.' + (keyHash >>> 0).toString(16);
  let currentView = 'program', state, cy, selected = null, toastTimer, busy = 0, currentLayout;
  const filters = new Set(Object.keys(relationNames));
  const api = window.SCAR_VIEW = {register, expand, collapse, openView, snapshot: summary};

  function toast(message, error = false) {
    clearTimeout(toastTimer); $('toast').textContent = message;
    $('toast').classList.toggle('error', error); $('toast').hidden = false;
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4200);
  }
  function loading(delta, message) {
    busy = Math.max(0, busy + delta); $('loading').hidden = busy === 0;
    if (message) $('loading-label').textContent = message;
  }
  function register(part) {
    if (!part || typeof part.id !== 'string' || !Array.isArray(part.nodes) || !Array.isArray(part.edges)) {
      toast('图分片格式错误：缺少 id / nodes / edges', true); return;
    }
    parts.set(part.id, part);
    const request = pending.get(part.id);
    if (request) { pending.delete(part.id); request.resolve(part); }
  }
  function loadPart(id) {
    if (parts.has(id)) return Promise.resolve(parts.get(id));
    if (pending.has(id)) return pending.get(id).promise;
    const spec = manifest.parts && manifest.parts[id];
    if (!spec || typeof spec.file !== 'string') return Promise.reject(new Error('未声明的子图分片：' + id));
    if (/^(?:[a-z]+:|\/|\\)/i.test(spec.file) || spec.file.split(/[\\/]/).includes('..'))
      return Promise.reject(new Error('分片必须使用导出目录内的相对路径'));
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    pending.set(id, {promise, resolve, reject});
    const script = document.createElement('script'); script.src = spec.file; script.async = true;
    script.onerror = () => {
      pending.delete(id); reject(new Error('无法加载分片 ' + spec.file + '；请保留完整导出目录。'));
    };
    script.onload = () => {
      if (!parts.has(id)) { pending.delete(id); reject(new Error('分片没有注册预期 ID：' + id)); }
    };
    document.head.appendChild(script);
    return promise;
  }
  function children(record) {
    return [...new Set([...(Array.isArray(record.children_parts) ? record.children_parts : []),
      ...(record.children_part ? [record.children_part] : [])])].filter(id => typeof id === 'string');
  }
  function conditionCount(record) {
    const values = [record.conditions, record.details && record.details.conditions];
    const entries = values.flatMap(value => Array.isArray(value) ? value :
      value && (typeof value !== 'object' || Object.keys(value).length) ? [value] : []);
    return new Set(entries.map(value => JSON.stringify(value))).size;
  }
  function makeState(root) {
    return {root, nodes:new Map(), edges:new Map(), parents:new Map(), instances:new Map(), positions:new Map(), pan:null, zoom:null, selected:null, notices:new Set()};
  }
  function ingest(target, part, owner) {
    const fallback = part.root && (target.nodes.has(part.root) || part.nodes.some(node => node.id === part.root)) ? part.root : owner;
    for (const node of part.nodes) {
      if (typeof node.id !== 'string' || !node.id) continue;
      if (!target.nodes.has(node.id)) {
        target.nodes.set(node.id, node);
        const declared = Object.prototype.hasOwnProperty.call(node, 'parent');
        const parent = declared ? node.parent : (node.id === fallback ? null : fallback);
        target.parents.set(node.id, typeof parent === 'string' && parent !== node.id ? parent : null);
      }
    }
    for (const edge of part.edges) {
      if (typeof edge.id === 'string' && typeof edge.source === 'string' && typeof edge.target === 'string') {
        if (target.nodes.has(edge.id)) { target.notices.add('节点与关系 ID 冲突：' + edge.id); continue; }
        if (!target.edges.has(edge.id)) target.edges.set(edge.id, edge);
      }
    }
  }
  function activeRecords(target, prune = true) {
    let nodes, edges;
    for (;;) {
      nodes = new Set(); edges = new Set();
      for (const item of target.instances.values()) {
        for (const node of item.part.nodes) nodes.add(node.id);
        for (const edge of item.part.edges) edges.add(edge.id);
      }
      let changed = false;
      if (prune) for (const [key, item] of target.instances) {
        if (item.owner && !nodes.has(item.owner)) { target.instances.delete(key); changed = true; }
      }
      if (!changed) break;
    }
    // A shared child may still need a container from a collapsed sibling.
    // Retain that visual ancestry rather than deleting somebody else's child.
    for (const id of [...nodes]) {
      const seen = new Set([id]); let parent = target.parents.get(id);
      while (parent && target.nodes.has(parent) && !seen.has(parent)) {
        seen.add(parent); nodes.add(parent); parent = target.parents.get(parent);
      }
    }
    return {nodes, edges};
  }
  function isExpanded(id) {
    return [...state.instances.values()].some(item => item.owner === id || (item.owner && item.part.root === id));
  }
  function visualParent(id, visible) {
    const parent = state.parents.get(id);
    if (!parent || !visible.has(parent)) return null;
    const seen = new Set([id]); let cursor = parent;
    while (cursor && visible.has(cursor)) {
      if (seen.has(cursor)) { state.notices.add('循环视觉父级：' + id); return null; }
      seen.add(cursor); cursor = state.parents.get(cursor);
    }
    return parent;
  }
  function nodeStyle(record) {
    const kind = String(record.kind || '').toUpperCase();
    let type = 'operation-node';
    if (record.layer === 'resource' || /RESOURCE|HARDWARE|MEMORY_DOMAIN|CPU_CORE|HARDWARE_THREAD/.test(kind)) type = 'resource-node';
    else if (/STATE|WORLD|VERSION/.test(kind)) type = 'state-node';
    else if (/VALUE|TENSOR|MATERIAL|PARAMETER|INPUT|OUTPUT|PORT/.test(kind)) type = 'value-node';
    return type + (children(record).length ? ' expandable-node' : '');
  }
  function capturePositions() {
    if (!cy || !state) return;
    cy.nodes().forEach(node => state.positions.set(node.id(), {...node.position()}));
    state.pan = {...cy.pan()}; state.zoom = cy.zoom();
  }
  function render(capture = true) {
    if (capture) capturePositions();
    const active = activeRecords(state), ids = active.nodes, parentMap = new Map();
    for (const id of ids) if (state.nodes.has(id)) parentMap.set(id, visualParent(id, ids));
    function depth(id) { let n = 0, p = parentMap.get(id); const seen = new Set([id]);
      while (p && !seen.has(p)) { seen.add(p); n++; p = parentMap.get(p); } return n; }
    const ordered = [...parentMap.keys()].sort((a, b) => depth(a) - depth(b));
    cy.batch(() => {
      cy.edges().remove();
      cy.nodes().filter(node => !ids.has(node.id())).remove();
      for (const [offset, id] of ordered.entries()) {
        const raw = state.nodes.get(id), hasChildren = children(raw).length, parent = parentMap.get(id);
        const data = {id, label:String(raw.label || id), kind:String(raw.kind || 'UNKNOWN'),
          displayLabel:String(raw.label || id) + (hasChildren ? (isExpanded(id) ? '  −' : '  ＋') : ''),
          parent:parent || undefined};
        let node = cy.getElementById(id);
        if (!node.length) {
          const origin = parent && state.positions.get(parent) || {x:0, y:0};
          node = cy.add({group:'nodes', data, classes:nodeStyle(raw),
            position:state.positions.get(id) || {x:origin.x + (offset % 5) * 180, y:origin.y + Math.floor(offset / 5) * 90}});
        } else {
          if ((node.parent().id() || null) !== parent) node.move({parent});
          node.data('displayLabel', data.displayLabel);
        }
      }
      for (const id of active.edges) {
        const raw = state.edges.get(id);
        if (!raw || !cy.getElementById(raw.source).length || !cy.getElementById(raw.target).length) continue;
        const layer = Object.prototype.hasOwnProperty.call(relationNames, raw.layer) ? raw.layer : 'other';
        const strength = String(raw.strength || 'UNKNOWN');
        const conditions = conditionCount(raw);
        cy.add({group:'edges', data:{id, source:raw.source, target:raw.target, layer, strength,
          conditionCount:conditions,
          proofStyle:conditions ? 'conditional' : strength === 'EXACT' ? 'exact' : strength === 'OBSERVED' ? 'observed' : 'uncertain',
          displayLabel:$('edge-labels').checked ? String(raw.label || raw.kind || '') : '',
          lineColor:conditions ? '#d2a368' : strength === 'EXACT' ? relationColors[layer] : strength === 'OBSERVED' ? '#75b9ed' : '#d2a368'}});
      }
    });
    applyFilters();
    if (selected && !cy.getElementById(selected.id).length) inspect(null);
    else if (selected) cy.getElementById(selected.id).addClass('focused');
  }
  function applyFilters() {
    cy.edges().forEach(edge => edge.style('display', filters.has(edge.data('layer')) ? 'element' : 'none'));
    updateStatus();
  }
  function summary() {
    if (!state || !cy) return {ready:false};
    const active = activeRecords(state), shownEdges = cy.edges().filter(edge => filters.has(edge.data('layer'))).length;
    const ids = new Set([...state.instances.values()].map(item => item.part.id));
    let frontiers = 0, boundaries = 0;
    for (const id of ids) {
      const part = parts.get(id);
      frontiers += Array.isArray(part.frontiers) ? part.frontiers.length : 0;
      boundaries += Array.isArray(part.boundaries) ? part.boundaries.length : 0;
    }
    return {ready:true, view:currentView, root:state.root, loaded_part_ids:[...ids],
      loaded_parts:parts.size, declared_parts:Object.keys(manifest.parts || {}).length,
      nodes:cy.nodes().length, rendered_edges:cy.edges().length, shown_edges:shownEdges,
      pending_endpoint_edges:active.edges.size - cy.edges().length, frontiers, boundaries,
      expanded_owners:[...new Set([...state.instances.values()].filter(item => item.owner).map(item => item.owner))],
      notices:[...state.notices]};
  }
  function updateStatus() {
    const info = summary(); if (!info.ready) return;
    $('graph-status').textContent = info.nodes + ' 节点 · ' + info.shown_edges + '/' + info.rendered_edges + ' 关系' +
      (info.pending_endpoint_edges ? ' · ' + info.pending_endpoint_edges + ' 条关系端点尚未展开' : '');
    $('part-status').textContent = '当前 ' + info.loaded_part_ids.length + ' 个分片 · 缓存 ' + info.loaded_parts + '/' + info.declared_parts;
    $('coverage-label').textContent = '仅当前展开范围；未展开 / 未执行不等于不存在。当前分片 frontier ' + info.frontiers + ' · boundary ' + info.boundaries;
    $('zoom-label').textContent = Math.round(cy.zoom() * 100) + '%';
    const counts = {};
    cy.edges().forEach(edge => { const key = edge.data('layer'); counts[key] = (counts[key] || 0) + 1; });
    for (const key of Object.keys(relationNames)) $('count-' + key).textContent = counts[key] || 0;
    $('empty-state').hidden = cy.nodes().length !== 0;
    if (!cy.nodes().length) $('empty-state').textContent = '此分片没有节点；请查看“范围与来源”中的边界和覆盖说明。';
  }
  function inspect(selection) {
    selected = selection; cy.elements().removeClass('focused');
    const raw = selection && (selection.type === 'node' ? state.nodes : state.edges).get(selection.id);
    if (!raw) {
      $('inspector-title').textContent = '节点与证据'; $('selection-badge').textContent = '尚未选择';
      $('selection-description').textContent = '点击节点或关系，查看原始 ID、类型、关系强度和来源记录。';
      $('node-actions').hidden = true;
      $('raw-details').textContent = JSON.stringify({提示:'此浏览器仅展示已有图；不会运行、优化或改写目标程序。'}, null, 2); return;
    }
    const element = cy.getElementById(selection.id); element.addClass('focused');
    $('inspector-title').textContent = selection.type === 'node' ? '节点' : '关系';
    $('selection-badge').textContent = String(raw.kind || 'UNKNOWN');
    $('selection-description').textContent = selection.type === 'node' ? String(raw.label || raw.id) :
      '强度：' + String(raw.strength || 'UNKNOWN') + ' · 图层：' + String(raw.layer || 'other') +
      (conditionCount(raw) ? ' · 带条件（' + conditionCount(raw) + ' 项）' : '');
    $('raw-details').textContent = JSON.stringify(raw, null, 2);
    $('node-actions').hidden = selection.type !== 'node';
    $('expand-node').disabled = !children(raw).length;
    $('expand-node').textContent = '展开 ' + children(raw).length + ' 个子分片';
    $('collapse-node').disabled = !isExpanded(raw.id);
  }
  async function expand(id, notify = true) {
    const target = state, record = target && target.nodes.get(id);
    if (!record) return false;
    const names = children(record);
    if (!names.length) { if (notify) toast('该节点没有可展开分片；详情中保留其边界信息。'); return false; }
    loading(1, '加载 ' + names.length + ' 个子分片…');
    try {
      const loaded = await Promise.all(names.map(loadPart));
      for (const part of loaded) {
        ingest(target, part, id); target.instances.set(id + '\u001f' + part.id, {owner:id, part});
      }
      if (state === target) { render(); inspect({type:'node', id}); }
      return true;
    } catch (error) { toast(error.message, true); return false; }
    finally { loading(-1); }
  }
  function collapse(id) {
    if (!state) return;
    for (const [key, item] of state.instances) {
      if (item.owner === id || (item.owner && item.part.root === id)) state.instances.delete(key);
    }
    render(); inspect(cy.getElementById(id).length ? {type:'node', id} : null);
  }
  function layout(name, fit = true) {
    if (currentLayout) currentLayout.stop();
    const options = {name, animate:false, fit, padding:45, avoidOverlap:true, nodeDimensionsIncludeLabels:true};
    if (name === 'breadthfirst') Object.assign(options, {directed:true, spacingFactor:1.35});
    if (name === 'cose') Object.assign(options, {randomize:false, nodeRepulsion:12000, idealEdgeLength:110, numIter:450});
    currentLayout = cy.elements().layout(options); currentLayout.run();
  }
  async function openView(view) {
    if (!cy) return;
    const root = view === 'resource' ? manifest.resource_root : manifest.root;
    if (!root) { toast('本次导出没有单独资源分片。请查看范围说明。'); return; }
    capturePositions(); currentView = view;
    state = views.get(view) || makeState(root); views.set(view, state);
    selected = null; cy.elements().remove();
    $('program-tab').classList.toggle('active', view === 'program');
    $('resource-tab').classList.toggle('active', view === 'resource');
    $('scope-label').textContent = (manifest.parts && manifest.parts[root] && manifest.parts[root].label || root) + ' · 离线按需展开';
    loading(1, '加载根分片…');
    const target = state;
    try {
      const part = await loadPart(root);
      ingest(target, part, null); target.instances.set('$root', {owner:null, part});
      if (state !== target) return;
      render(false); inspect(null);
      if (target.zoom !== null && target.pan) { cy.zoom(target.zoom); cy.pan(target.pan); }
      else layout('breadthfirst');
    } catch (error) { toast(error.message, true); $('graph-status').textContent = error.message; }
    finally { loading(-1); }
  }
  function snapshot() {
    capturePositions();
    return {schema:'scar.graph-view-layout', schema_version:1, source_key:sourceKey,
      view:currentView, expanded_owners:summary().expanded_owners,
      positions:Object.fromEntries(state.positions), zoom:cy.zoom(), pan:cy.pan(),
      relations:[...filters], edge_labels:$('edge-labels').checked};
  }
  async function restore(data) {
    if (!data || data.schema !== 'scar.graph-view-layout' || data.schema_version !== 1 || data.source_key !== sourceKey)
      throw new Error('布局来源与当前图不一致，已拒绝恢复。');
    await openView(data.view === 'resource' ? 'resource' : 'program');
    for (const [key, item] of state.instances) if (item.owner) state.instances.delete(key);
    render();
    let owners = new Set(Array.isArray(data.expanded_owners) ? data.expanded_owners : []);
    while (owners.size) {
      let progress = false;
      for (const id of [...owners]) if (cy.getElementById(id).length) {
        owners.delete(id); await expand(id, false); progress = true;
      }
      if (!progress) break;
    }
    if (Array.isArray(data.relations)) {
      filters.clear(); for (const name of data.relations) if (name in relationNames) filters.add(name);
      for (const name of Object.keys(relationNames)) $('filter-' + name).checked = filters.has(name);
    }
    $('edge-labels').checked = !!data.edge_labels;
    render();
    const positions = data.positions || {};
    cy.nodes().forEach(node => {
      const position = positions[node.id()];
      if (position && Number.isFinite(position.x) && Number.isFinite(position.y)) node.position(position);
    });
    if (Number.isFinite(data.zoom) && data.zoom > 0) cy.zoom(data.zoom);
    if (data.pan && Number.isFinite(data.pan.x) && Number.isFinite(data.pan.y)) cy.pan(data.pan);
    updateStatus(); toast(owners.size ? '已恢复可定位的布局；部分节点不在当前分片范围。' : '已恢复本地布局。');
  }
  function manifestDetails() {
    selected = null; cy.elements().removeClass('focused');
    $('inspector-title').textContent = '范围与来源'; $('selection-badge').textContent = String(manifest.schema || 'UNKNOWN');
    $('selection-description').textContent = '清单与当前已加载分片的原始 frontiers / boundaries；未展开部分尚未读取。';
    $('node-actions').hidden = true;
    $('raw-details').textContent = JSON.stringify({manifest, current_view:summary(), current_parts:[...new Set(
      [...state.instances.values()].map(item => item.part.id))].map(id => {
        const part = parts.get(id); return {id, label:part.label, frontiers:part.frontiers, boundaries:part.boundaries};
      })}, null, 2);
  }
  function setup() {
    if (!window.cytoscape) { $('graph-title').textContent = '图引擎未加载'; $('graph-status').textContent = '缺少 vendor/cytoscape.min.js；请保留完整导出目录。'; return; }
    if (!manifest.root || !manifest.parts) { $('graph-title').textContent = '缺少图清单'; $('graph-status').textContent = '需要同目录 manifest.js。'; return; }
    $('graph-title').textContent = manifest.title || 'SCAR 程序图'; document.title = (manifest.title || '程序图') + ' · SCAR';
    $('resource-tab').disabled = !manifest.resource_root;
    for (const [key, label] of Object.entries(relationNames)) {
      const row = document.createElement('label'); row.className = 'filter-row';
      const input = document.createElement('input'); input.type = 'checkbox'; input.checked = true; input.id = 'filter-' + key;
      input.addEventListener('change', () => { input.checked ? filters.add(key) : filters.delete(key); applyFilters(); });
      const dot = document.createElement('span'); dot.className = 'filter-dot'; dot.style.background = relationColors[key];
      const text = document.createElement('span'); text.textContent = label;
      const count = document.createElement('span'); count.className = 'count'; count.id = 'count-' + key; count.textContent = '0';
      row.append(input, dot, text, count); $('relation-filters').appendChild(row);
    }
    cy = window.cytoscape({container:$('cy'), elements:[], minZoom:.04, maxZoom:4, wheelSensitivity:.22,
      boxSelectionEnabled:true, selectionType:'single', style:[
        {selector:'node', style:{'shape':'roundrectangle','background-color':'#263b50','border-width':1,'border-color':'#6184a2',
          'color':'#e2ecf5','label':'data(displayLabel)','font-size':11,'font-family':'sans-serif','text-wrap':'wrap','text-max-width':142,
          'text-valign':'center','text-halign':'center','width':160,'height':48,'overlay-opacity':0}},
        {selector:'.expandable-node',style:{'background-color':'#18433f','border-color':'#5cbfae'}},
        {selector:'.state-node',style:{'background-color':'#392b4c','border-color':'#a08abe'}},
        {selector:'.value-node',style:{'background-color':'#213d53','border-color':'#6aa4cc','shape':'ellipse','width':134,'height':43}},
        {selector:'.resource-node',style:{'background-color':'#263e33','border-color':'#77ac8d'}},
        {selector:'node:parent',style:{'background-opacity':.12,'border-color':'#456579','border-width':1,'padding':24,
          'shape':'roundrectangle','text-valign':'top','text-margin-y':-8,'font-size':12}},
        {selector:'edge',style:{'curve-style':'bezier','width':1.3,'line-color':'data(lineColor)','target-arrow-color':'data(lineColor)',
          'target-arrow-shape':'triangle','arrow-scale':.7,'opacity':.75,'label':'data(displayLabel)','font-size':9,
          'color':'#aabbd0','text-background-color':'#111924','text-background-opacity':.9,'text-background-padding':2}},
        {selector:'edge[proofStyle != "exact"]',style:{'line-style':'dashed','line-dash-pattern':[5,4]}},
        {selector:'.focused',style:{'border-width':3,'border-color':'#f2d390','overlay-color':'#ead092','overlay-opacity':.12}},
        {selector:'edge.focused',style:{'width':3,'opacity':1,'line-color':'#f2d390','target-arrow-color':'#f2d390'}}
      ]});
    api.cy = cy;
    cy.on('tap', 'node', event => inspect({type:'node', id:event.target.id()}));
    cy.on('tap', 'edge', event => inspect({type:'edge', id:event.target.id()}));
    cy.on('tap', event => { if (event.target === cy) inspect(null); });
    cy.on('dbltap', 'node', event => { const id = event.target.id(); isExpanded(id) ? collapse(id) : expand(id); });
    cy.on('zoom', () => { $('zoom-label').textContent = Math.round(cy.zoom() * 100) + '%'; });
    cy.on('dragfree', 'node', event => state.positions.set(event.target.id(), {...event.target.position()}));
    $('program-tab').onclick = () => openView('program'); $('resource-tab').onclick = () => openView('resource');
    $('manifest-button').onclick = manifestDetails; $('clear-selection').onclick = () => inspect(null);
    $('expand-node').onclick = () => selected && expand(selected.id); $('collapse-node').onclick = () => selected && collapse(selected.id);
    $('fit-button').onclick = () => cy.fit(undefined, 45);
    $('zoom-in').onclick = () => cy.zoom({level:Math.min(cy.maxZoom(), cy.zoom() * 1.3), renderedPosition:{x:cy.width()/2,y:cy.height()/2}});
    $('zoom-out').onclick = () => cy.zoom({level:Math.max(cy.minZoom(), cy.zoom() / 1.3), renderedPosition:{x:cy.width()/2,y:cy.height()/2}});
    document.querySelectorAll('[data-layout]').forEach(button => { button.onclick = () => layout(button.dataset.layout); });
    $('edge-labels').onchange = () => render();
    $('search-form').onsubmit = event => {
      event.preventDefault(); const query = $('search-input').value.trim().toLocaleLowerCase();
      if (!query) return;
      const matches = cy.nodes().filter(node => {
        const raw = state.nodes.get(node.id()); return String(raw.id + ' ' + (raw.label || '')).toLocaleLowerCase().includes(query);
      });
      $('search-result').textContent = matches.length ? '当前图匹配 ' + matches.length + ' 个节点' : '当前展开范围没有匹配节点';
      if (matches.length) { inspect({type:'node', id:matches[0].id()}); cy.fit(matches, 100); }
    };
    $('save-layout').onclick = () => { try { localStorage.setItem(storageKey, JSON.stringify(snapshot())); toast('布局已保存到本浏览器。'); }
      catch (_) { toast('浏览器不允许本地存储；请使用“导出布局”。', true); } };
    $('restore-layout').onclick = async () => { try {
      const raw = localStorage.getItem(storageKey); if (!raw) return toast('尚未保存本地图布局。'); await restore(JSON.parse(raw));
    } catch (error) { toast(error.message, true); } };
    $('export-layout').onclick = () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot(), null, 2)], {type:'application/json'}));
      const link = document.createElement('a'); link.href = url; link.download = 'scar-layout.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    $('import-layout').onclick = () => $('layout-file').click();
    $('layout-file').onchange = async event => {
      const file = event.target.files[0]; if (!file) return;
      try { await restore(JSON.parse(await file.text())); } catch (error) { toast(error.message, true); } event.target.value = '';
    };
    $('copy-details').onclick = async () => { try { await navigator.clipboard.writeText($('raw-details').textContent); toast('已复制原始记录。'); }
      catch (_) { const selection = window.getSelection(); const range = document.createRange(); range.selectNodeContents($('raw-details')); selection.removeAllRanges(); selection.addRange(range); toast('已选中记录，可按 Ctrl/Cmd+C 复制。'); } };
    document.addEventListener('keydown', event => {
      if (/INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key === '/') { event.preventDefault(); $('search-input').focus(); }
      else if (event.key.toLowerCase() === 'f') cy.fit(undefined, 45);
      else if (event.key === 'Escape') inspect(null);
      else if (selected && selected.type === 'node' && event.key.toLowerCase() === 'e') expand(selected.id);
      else if (selected && selected.type === 'node' && event.key.toLowerCase() === 'c') collapse(selected.id);
    });
    openView('program');
  }
  setup();
})();
