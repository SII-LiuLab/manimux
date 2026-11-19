"""Render saved offline results with matplotlib; does not read robot recordings.

Use a Python environment containing numpy/matplotlib. The analysis environment
only needs ManiMux's existing numpy/scipy/zarr dependencies.
"""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(output):
    payload=json.loads((output/'view-data.json').read_text())
    s=payload['summary']; extra=payload['extra']; controls=s['controllers'];names=[v['name'] for v in controls]
    short=['Recorded command','Velocity FF','Velocity feedback','FF + velocity feedback']
    blue='#246b93';gray='#6b7280';orange='#b56b22';colors=[gray,blue,orange,'#253749']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                        'svg.fonttype':'none','figure.dpi':140})
    figures=[]
    def save(fig,name,caption):
        fig.text(.01,.005,'Source: ManiMux throw-ball recordings, 2026-11-13 UTC; offline analysis 2026-11-17. '+caption,
                 fontsize=7,color=gray)
        fig.tight_layout(rect=(0,.04,1,1))
        fig.savefig(output/(name+'.svg'));fig.savefig(output/(name+'.png'));plt.close(fig)
        figures.append((name,caption))
    joint=[v.replace('left_arm/','L ').replace('right_arm/','R ') for v in s['joints']]
    x=np.arange(14)
    fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True)
    ax[0].plot(x,s['measured_test_joint_rmse_deg'],color=blue,label='Measured command-to-state RMS')
    ax[0].plot(x,s['model_test_joint_rmse_deg'],color=gray,label='Model-to-state residual RMS')
    ax[0].set(ylabel='Joint position RMS error (deg)',title='Measured following error and held-out model error (4 complete episodes)');ax[0].legend()
    ax[1].bar(x,extra['median_lag_ms'],color=blue);ax[1].set(xticks=x,xticklabels=joint,ylabel='Apparent lag (ms)',xlabel='Arm / joint',title='Best time-shift alignment: diagnostic, not a causal controller')
    save(fig,'01-tracking-and-lag','Lag is the median over test segments; RMS is episode balanced. Delay includes unknown feedback age.')
    fig,ax=plt.subplots(1,2,figsize=(12,4.3))
    errors=[s['heldout_rmse_deg'][n] for n in names]
    ax[0].bar(np.arange(4),errors,color=colors)
    for i,e in enumerate(errors):ax[0].text(i,e+.025,f'{e:.3f}',ha='center')
    ax[0].set(xticks=np.arange(4),xticklabels=['Baseline','FF','VFB','FF+VFB'],ylabel='Simulated tracking RMS (deg)',xlabel='Offline controller',title='Held-out frozen-reference counterfactual')
    for i,n in enumerate(names[1:]):
        low,high=s['robustness_gain_range'][n];nom=s['heldout_gain_pct'][n]
        ax[1].plot([low,high],[i,i],color=colors[i+1],lw=4);ax[1].plot(nom,i,'o',color=colors[i+1])
    ax[1].set(yticks=range(3),yticklabels=short[1:],xlabel='RMS reduction versus same-scenario baseline (%)',ylabel='Controller',title='13 model / delay / noise scenarios (range, not CI)')
    save(fig,'02-counterfactual','Fit: 20 episodes; tune: 5; test: 4. No real robot intervention or task-success estimate.')
    example=next(c for c in payload['cases'] if c['id']==str(s['example_segment']))
    t=np.array(example['time']);j=s['example_joint'];u=np.array(example['command'])[:,j];q=np.array(example['state'])[:,j];r=np.array(example['reference'])[:,j]
    mid=t[np.argmax(np.abs(np.diff(u,prepend=u[0])))];mask=(t>mid-3)&(t<mid+3)
    fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True)
    ax[0].plot(t[mask],r[mask],color=gray,ls=':',label='Scheduled reference')
    ax[0].plot(t[mask],u[mask],color=blue,label='Recorded command');ax[0].plot(t[mask],q[mask],color=orange,label='Measured state')
    ax[0].set(title=f"Observed dynamic interval: {joint[j]} / {example['label']}",ylabel='Joint angle (deg)');ax[0].legend()
    for i in [0,1,2]:
        y=np.array(example['simulated'][i])[:,j]
        ax[1].plot(t[mask],(y-u)[mask],color=colors[i],label=short[i]+' (simulated)')
    ax[1].set(ylabel='Simulated state minus command (deg)',xlabel='Time since first recorded tick (s)',title='Same fixed recorded reference, simulated feedback in every counterfactual');ax[1].legend()
    save(fig,'03-dynamic-trace','Trace decimated to 20 Hz for display only; simulation runs at 100 Hz. Window selected by reference speed.')
    n=payload['noise'];fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True)
    ax[0].plot(n['time'],n['naive'],color=orange,label='Raw delta position / tick dt')
    ax[0].plot(n['time'],n['filtered'],color=blue,label='Causal 8 Hz low-pass estimate')
    ax[0].set(ylabel='Estimated joint velocity (rad/s)',title='Timestamp jitter can turn position differences into velocity spikes: '+n['joint']);ax[0].legend()
    ax[1].plot(n['time'],n['dt_ms'],color=gray);ax[1].set(xlabel='Time since first recorded tick (s)',ylabel='Control tick interval (ms)',title='Original irregular timestamps, not uniform resampling')
    save(fig,'04-velocity-estimation','Neither curve is a hardware velocity measurement; raw feedback acquisition timestamps are unavailable.')
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for i,n in enumerate(names):
        acc=extra['controls']['accel_p99'][n];gain=s['heldout_gain_pct'][n]
        ax[0].scatter(acc,gain,color=colors[i],s=40);ax[0].annotate(['Baseline','FF','VFB','FF+VFB'][i],(acc,gain),xytext=(4,5),textcoords='offset points')
    ax[0].set(xlabel='Command acceleration p99 (rad/s²)',ylabel='Simulated RMS reduction (%)',title='Tracking benefit requires more aggressive position commands')
    residual=np.asarray(s['model_test_joint_rmse_deg']);ax[1].bar([0,1],[float(np.sqrt(np.mean(residual**2))),s['heldout_rmse_deg'][names[1]]-s['heldout_rmse_deg'][names[3]]],color=[gray,blue])
    ax[1].set(xticks=[0,1],xticklabels=['Plant model residual','Extra FF+VFB gain'],ylabel='Joint error scale (deg)',xlabel='Evidence scale (different aggregations)',title='Small incremental feedback gain is below model uncertainty')
    save(fig,'05-benefit-cost','Acceleration uses differences of command velocity at 100 Hz. Additional real safety/effort limits are unmodelled.')
    table=''.join(f'<tr><td>{html.escape(short[i])}</td><td>{errors[i]:.3f}</td><td>{s["heldout_gain_pct"][n]:.1f}%</td><td>{extra["controls"]["accel_p99"][n]:.1f}</td></tr>' for i,n in enumerate(names))
    figure_blocks=[]
    for name,caption in figures:
        svg=(output/(name+'.svg')).read_text()
        figure_blocks.append('<figure>'+svg[svg.index('<svg'):]+
                             '<figcaption>'+html.escape(caption)+'</figcaption></figure>')
    figure_html=''.join(figure_blocks)
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Tianji tossing：速度反馈离线评估</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 24px;color:#20242b;background:#fff}}h1{{font-size:26px}}h2{{font-size:21px;margin-top:36px}}small,figcaption{{color:#626b75}}svg{{width:100%;height:auto}}figure{{margin:28px 0}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:10px}}code{{background:#f2f4f6;padding:2px 5px}}aside{{padding:16px;background:#f2f4f6}}a{{color:#246b93}}</style>
<h1>Tianji tossing：先补偿跟踪滞后，再评估速度反馈</h1>
<p>当前证据支持继续研究<strong>位置接口上的速度前馈／相位提前</strong>；不支持仅凭差分位置就认定新增速度反馈环能明显改善高动态。这里是离线筛选，没有修改控制器、连接机器人或运行真机。</p>
<aside>数据截至 2026-11-13，分析日期 2026-11-17。65 个已保存 attempt 中，49 个非空：30 个正常结束、19 个异常结束；正常结束不等于投掷成功。最新样本没有 sigmoid，均为当时记录的 Smooth 8 Hz / 100 Hz 控制配置。故障记录保留在独立 stress 集。</aside>
<h2>数据真正说明了什么</h2>
<p>最新 4 条完整留出 rollout：实测 command→state RMS（episode 等权）约 <strong>{extra['observed']['command_rmse_deg']:.2f}°</strong>，scheduled→command 整形 RMS 约 <strong>{extra['observed']['shaping_rmse_deg']:.2f}°</strong>。这两项 RMS 不能直接相加。关节最佳时间对齐滞后主要为 <strong>70–90 ms</strong>，但包括未知的反馈采样年龄和传输延迟。</p>
<p>最新完整 rollout 的原始位置差分出现约 17.8 rad/s 的估计尖峰。缺少逐 tick 反馈采集时间、硬件速度／力矩、有效手臂总线发送样本，因此这些尖峰不能当成真实速度。当前代码只证明外部发送关节位置；不能据此判断固件内部有没有速度环。</p>
<h2>固定参考的因果反事实</h2>
<p>用 20 条较早完整 rollout 拟合有延迟的一阶／二阶位置模型，5 条中间 rollout 选择模型和增益，最新 4 条完整 rollout 保留测试。另有 18 条异常结束轨迹可作 stress 回放。按 session 分组；不把相邻 tick 随机拆成训练和测试。每段至少 3 秒，前 1 秒不计分，超过 50 ms 的空档切段。异常轨迹最后 0.5 秒不用于 surrogate stress，原始诊断保留全段。</p>
<p><code>u = r + Tff × v_ref + Tv × (v_ref − v_hat)</code>；r 是记录的 executor command，v_ref 和 v_hat 均使用因果差分加 8 Hz 低通。模拟反馈来自新的模拟状态，不使用冻结的实测状态冒充闭环。夹爪轨迹不改。每个模型均为稳定的 unit-DC-gain SISO surrogate，未建模双臂耦合、接触、负载或固件内部限制。</p>
<table><thead><tr><th>方案</th><th>模拟 RMS / °</th><th>模拟下降</th><th>指令加速度 p99 / rad·s⁻²</th></tr></thead><tbody>{table}</tbody></table>
<p>验证集选择的离线候选：FF=80 ms、VFB=20 ms、FF+VFB=80+10 ms。<strong>不是可直接部署的增益。</strong>反馈在 FF 之上的增量收益只有约 {s['heldout_gain_pct'][names[3]]-s['heldout_gain_pct'][names[1]]:.2f} 个百分点，明显小于模型残差。13 个模型／延迟／噪声情景包含把 10–30 ms 表观延迟归给传感器的不可辨识性检查；这些区间是人为情景范围，不是收益保证。</p>
<p>位置修正先软限幅 3°，再经过关节位置、速度和加速度限制。由于 slew 限制，最终指令相对 r 的偏移可能超过 3°；原始表中保留 clipping、rate-limit 占比和最大偏移。这意味着当前方案仍需约束设计，不能作为已验证的安全控制器。更大的指令加速度不等于更大的真实机器人带宽。</p>
{figure_html}
<h2>下一步验证应以什么为通过条件</h2>
<ol><li>先记录反馈采集时间／sequence、实际提交位置和返回时刻、原生关节速度（若接口提供）、控制模式与限速状态。当前 host tick 不足以分离执行延迟和反馈延迟。</li>
<li>在相同的受限参考轨迹上比较现有位置命令、有限速度前馈、前馈加小速度反馈；先固定参考，再恢复视觉策略闭环。记录 RMSE、p95、方向反转超调、速度／加速度、限幅比例和故障；不要仅看误差均值。</li>
<li>要证明投掷收益，还需实际 TCP 速度与松爪时刻、球的释放／落点和任务成功率。现有离线 joint replay 不能回答这些物理结果。</li>
<li>先要求模型在独立轨迹上的残差小于拟检验的收益，并在延迟、采样抖动、负载变化和饱和情景下不恶化。目前 VFB 的小幅收益尚不满足这个证据强度。</li></ol>
<h2>复现与证据</h2>
<p><code>.venv/bin/python -m scripts.experiments.analyze_tracking_loop</code><br>
<code>/home/jw/miniforge3/bin/python -m scripts.experiments.render_tracking_audit</code></p>
<p><a href="summary.json">Summary</a> · <a href="cohort.json">逐 rollout 来源、配置身份与输入 hash</a> · <a href="models.json">模型参数</a> · <a href="heldout.json">逐 episode 模拟结果</a> · <a href="robustness.json">敏感性情景</a> · <a href="validation-sweep.json">完整调参搜索</a> · <a href="partial-stress.json">异常结束轨迹 stress</a></p>
<p>控制概念参考：<a href="https://control.ros.org/rolling/doc/ros2_controllers/pid_controller/doc/userdoc.html">ROS 2 PID controller：接口类型、参考导数和前馈</a>；<a href="https://control.ros.org/master/doc/control_toolbox/doc/index.html">control_toolbox：PID 与滤波</a>。这些文档解释方法，不能代替 Tianji 固件资料。</p>
</html>'''
    (output/'report.html').write_text(report)
    print('Rendered',len(figures),'figures and',output/'report.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/analysis/throw-ball-velocity-loop-20261117'))
    args=parser.parse_args();main(args.output)
