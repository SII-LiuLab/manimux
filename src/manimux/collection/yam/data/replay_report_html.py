"""Self-contained HTML for recorded achieved reconstruction (not physical tracking)."""

# ruff: noqa: E501
PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Achieved · 逐关节重建对比</title>
<style>body{font:15px system-ui;margin:0;background:#f4f6fa;color:#21344c}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:30px}h1 span{color:#e76b00}p{line-height:1.7}section,article{background:white;padding:20px;border-radius:12px;margin:16px 0}video{width:460px;max-width:100%;border-radius:8px}button,input,select{font:inherit;padding:7px;margin:4px;border:1px solid #cbd4df;border-radius:5px}button{cursor:pointer}#charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}article{margin:0;min-width:0}canvas{width:100%;height:240px;cursor:crosshair}small{color:#63758b}.controls{position:sticky;top:0;background:#fffef9;z-index:2;padding:12px;box-shadow:0 2px 6px #dce2ea}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:right;border-bottom:1px solid #edf0f5}#table{overflow:auto}#legend{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:800px){#charts{grid-template-columns:1fr}main{padding:12px}}</style>
<main><h1>原始 achieved <span>× 两种低频重建</span></h1>
<p>此页是已录数据的离线重建；真机 controller gap 需由三次 replay 新记录计算。</p>
<section><details><summary>数据与采样方法</summary><p>@@NOTE@@</p><small id="source"></small></details>
<p id="legend"></p><video id="video" controls preload="metadata" src="data:video/mp4;base64,@@VIDEO@@"></video>
<p id="video-status"></p><p>小空心点：真正保存的 achieved 采样点（横坐标可能落在两格之间）。蓝线：每个比较帧取此前最近值。橙／绿线：线性重建。点击任意曲线定位原视频；帧号与 Viser 相同。</p></section>
<div class="controls"><button id="prev">上一帧</button><button id="next">下一帧</button>
比较帧 <input id="frame" type="number" min="0" value="0" style="width:80px">
窗口 <select id="span"><option value="30">30 帧</option><option value="60" selected>60 帧</option><option value="120">120 帧</option><option value="all">全部</option></select>
<input id="seek" type="range" min="0" value="0" style="width:35%"><b id="status"></b></div>
<div id="charts"></div><section><h2>全段逐关节误差（度）</h2><p>两种方案使用相同有效比较帧；这是重建误差，不是 MIT 控制精度。</p><div id="table"></div></section></main>
<script id="data" type="application/json">@@DATA@@</script><script>
'use strict';const D=JSON.parse(document.getElementById('data').textContent),$=s=>document.getElementById(s),colors=['#2069ac','#ea6a00','#15966e','#984cc6'];
let frame=0;const N=D.time_s.length,panels=[];$('frame').max=$('seek').max=N-1;
$('source').textContent=D.episode+' · '+Object.entries(D.source_rates_hz).map(([a,h])=>a+' '+h.toFixed(2)+' Hz').join(' / ');
D.methods.forEach((m,i)=>{const t=document.createElement('b');t.style.color=colors[i%colors.length];t.textContent=m.label;$('legend').append(t)});
for(const [arm,A] of Object.entries(D.arms))for(let j=0;j<6;j++){
 const a=document.createElement('article');a.innerHTML='<b>'+ (arm==='left'?'左':'右')+'从臂 · J'+(j+1)+'</b><canvas></canvas><small></small>';$('charts').append(a);
 const p={a,arm,A,j,c:a.querySelector('canvas'),label:a.querySelector('small')};panels.push(p);
 p.c.onclick=e=>{const r=p.c.getBoundingClientRect();setFrame(Math.round(p.lo+(e.clientX-r.left-56)/(r.width-72)*(p.hi-p.lo)),true)};
}
const fmt=v=>v===null?'缺失':v.toFixed(4);function bounds(){const span=$('span').value==='all'?N:+$('span').value;const lo=Math.max(0,Math.min(N-span,frame-Math.floor(span/2)));return[lo,Math.min(N-1,lo+span-1)]}
function paint(p){const [lo,hi]=bounds();p.lo=lo;p.hi=hi;const c=p.c,w=c.clientWidth,h=240,dpr=window.devicePixelRatio||1;c.width=w*dpr;c.height=h*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);
 let ys=[];for(const m of D.methods)for(let i=lo;i<=hi;i++){const y=p.A.curves[m.key][i][p.j];if(y!==null)ys.push(y)}
 const raw=p.A.raw_time_s.map((t,i)=>({x:(t-D.time_s[0])*D.target_hz,y:p.A.raw_deg[i][p.j]})).filter(v=>v.x>=lo&&v.x<=hi);ys.push(...raw.map(v=>v.y));
 if(!ys.length){ctx.fillText('缺失数据',70,80);return}let min=Math.min(...ys),max=Math.max(...ys),pad=Math.max(.005,(max-min)*.12);min-=pad;max+=pad;
 const x=f=>56+(f-lo)/(hi-lo||1)*(w-72),y=q=>18+(max-q)/(max-min)*(h-46);ctx.font='11px system-ui';ctx.lineWidth=1;
 for(let i=0;i<=4;i++){const q=min+(max-min)*i/4;ctx.strokeStyle='#e5eaf1';ctx.beginPath();ctx.moveTo(56,y(q));ctx.lineTo(w-16,y(q));ctx.stroke();ctx.fillStyle='#607086';ctx.fillText(q.toFixed(2),3,y(q)+3)}
 const stride=hi-lo<=120?1:Math.ceil((hi-lo)/20);for(let i=lo;i<=hi;i+=stride){ctx.strokeStyle='#edf0f5';ctx.beginPath();ctx.moveTo(x(i),18);ctx.lineTo(x(i),h-28);ctx.stroke();if(i%Math.max(stride,Math.ceil((hi-lo)/10))===0){ctx.fillStyle='#607086';ctx.fillText(i,x(i)-5,h-9)}}
 for(let k=0;k<D.methods.length;k++){const m=D.methods[k],v=p.A.curves[m.key];ctx.strokeStyle=colors[k%colors.length];ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=1.8;ctx.setLineDash(k?[5,4]:[]);ctx.beginPath();let last=null;
 for(let i=lo;i<=hi;i++){const q=v[i][p.j];if(q===null){last=null;continue}if(last===null)ctx.moveTo(x(i),y(q));else{if(k===0)ctx.lineTo(x(i),y(last));ctx.lineTo(x(i),y(q))}last=q}ctx.stroke();ctx.setLineDash([]);
 if(hi-lo<=120)for(let i=lo;i<=hi;i++){const q=v[i][p.j];if(q===null)continue;ctx.beginPath();ctx.arc(x(i),y(q),2.1,0,Math.PI*2);ctx.fill()}}
 ctx.strokeStyle=colors[0];ctx.lineWidth=1;for(const v of raw){ctx.beginPath();ctx.arc(x(v.x),y(v.y),3.5,0,Math.PI*2);ctx.stroke()}
 ctx.strokeStyle='#9b728b';ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(x(frame),18);ctx.lineTo(x(frame),h-28);ctx.stroke();ctx.setLineDash([]);
 p.label.textContent='帧 '+frame+' · '+D.methods.map(m=>m.label+': '+fmt(p.A.curves[m.key][frame][p.j])+'°').join(' / ')+' · 原始行 '+p.A.source_index[frame]+'，帧龄 '+p.A.source_age_ms[frame].toFixed(2)+' ms';
}
function draw(){for(const p of panels)paint(p);$('frame').value=$('seek').value=frame;$('status').textContent=frame+' / '+(N-1)+' · '+D.time_s[frame].toFixed(4)+' s';$('video-status').textContent='对应原视频帧 '+D.video_index[frame]+'，按保存的主机时间戳匹配。'}
function setFrame(f,video){frame=Math.max(0,Math.min(N-1,Number.isFinite(f)?f:0));if(video){$('video').pause();const i=D.video_index[frame];if(i>=0)$('video').currentTime=D.video_pts_s[i]+.00001}draw()}
$('prev').onclick=()=>setFrame(frame-1,true);$('next').onclick=()=>setFrame(frame+1,true);$('frame').onchange=()=>setFrame(Math.round(+$('frame').value),true);$('seek').oninput=()=>setFrame(+$('seek').value,true);$('span').onchange=draw;
function before(a,t){let lo=0,hi=a.length;while(lo<hi){const m=(lo+hi)>>1;if(a[m]<=t)lo=m+1;else hi=m}return lo-1}
$('video').ontimeupdate=()=>{if($('video').paused)return;const i=before(D.video_pts_s,$('video').currentTime+.00001);if(i>=0)setFrame(Math.max(0,before(D.time_s,D.camera_time_s[i])),false)};
const table=document.createElement('table');table.innerHTML='<thead><tr><th>臂 / 关节</th><th>方案</th><th>有效帧</th><th>RMSE</th><th>MAE</th><th>P95</th><th>最大</th></tr></thead>';const tbody=document.createElement('tbody');
for(const r of D.metrics){const row=document.createElement('tr');for(const v of [r.arm+' J'+r.joint,r.low_hz+'→'+D.target_hz,r.valid_samples,...['rmse_deg','mae_deg','p95_abs_deg','max_abs_deg'].map(k=>fmt(r[k]))]){const cell=document.createElement('td');cell.textContent=v;row.append(cell)}tbody.append(row)}table.append(tbody);$('table').append(table);window.addEventListener('resize',draw);draw();
</script></html>'''
