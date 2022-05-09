# 导入模块
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# 设置绘图风格
plt.style.use('ggplot')
# 设置中文编码和负号的正常显示

config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman', ],
    "font.size": 11,
    "mathtext.fontset": 'stix',
}
plt.rcParams.update(config)

#plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 读取需要绘图的数据
values = list(range(-3,5))
values = [1.0 * pow(2, i) for i in values]
#values = [100.0 * pow(2, i) for i in values]
#values = list(range(-3,5))
auc_results = [0.04,	0.11,	0.31,	0.43,	0.41,	0.32,	0.13,	0.04]
mae_results = [0.04,	0.06,	0.13,	0.20,	0.14,	0.10,	0.07,	0.03]

xylabel_fontdict = {'size': 11, 'color': 'k', 'family': 'Times New Roman'}
title_fontdict = {'size': 14, 'color': 'k', 'family': 'Times New Roman'}

fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(4.3)
ax2 = ax.twinx()

left, width = 0.25, 0.5
bottom, height = 0.25, 0.5
right = left + width
top = bottom + height
p = patches.Rectangle((left, bottom), width, height, fill=False, transform=ax.transAxes, clip_on=False)
#ax.add_patch(p)
ax.set_axis_on()
ax2.set_axis_on()
ax.grid(False)
ax2.grid(False)
ax.set_facecolor('white')
ax2.set_facecolor('white')
for spine in ['left','right','top','bottom']:
    ax.spines[spine].set_color('k')
    ax2.spines[spine].set_color('k')

#ax.spines['right'].set_visible(True)
#ax.spines['top'].set_visible(True)
#ax.spines['left'].set_visible(True)
#ax.spines['bottom'].set_visible(True)
auc_plot = ax.plot(values, auc_results, color='b', label='AUC Gain', marker='o')
mae_plot = ax2.plot(values, mae_results, color='r', label='MAE Gain', marker='o')

auc_y_shift = [-15, -15, 10, 10, 10, 10, -15, -15]
mae_y_shift = [10, 10, -15, -15, -15, -15, 10, 10]

lns = auc_plot + mae_plot
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

i = 0
for xy in zip(values, auc_results):
    if xy[0] > 1.0:
        x_shift=-5
    else:
        x_shift=-10
    y_shift = auc_y_shift[i]
    i = i + 1
    ax.annotate('%s' % xy[1], xy=xy, xytext=(x_shift, y_shift), textcoords='offset points', color='blue')#arrowprops=dict(facecolor='blue', shrink=0.05))

i = 0
for xy in zip(values, mae_results):
    if xy[0] > 1.0:
        x_shift=-5
    else:
        x_shift=-10

    y_shift = mae_y_shift[i]
    i = i + 1
    ax2.annotate('%s' % xy[1], xy=xy, xytext=(x_shift, y_shift), textcoords='offset points', color='red')#arrowprops=dict(facecolor='red', shrink=0.05))

#ax.set_title('The impact of similarity loss weight', fontdict = title_fontdict, fontweight='bold', bbox=dict(facecolor='w', edgecolor='blue', alpha=0.65 ))
ax.set_ylabel('AUC Gain (%)', fontdict = xylabel_fontdict, fontweight='bold' )
ax2.set_ylabel('MAE Gain (%)', fontdict = xylabel_fontdict, fontweight='bold' )
ax.set_xlabel('Loss Weight Ratio Relative to the CTR Task', fontdict = xylabel_fontdict, fontweight='bold' )

auc_y_ticks = np.arange(0.0, 0.6, 0.1)
ax.set_yticks(auc_y_ticks)

mae_y_ticks = np.arange(0.0, 0.3, 0.05)
ax2.set_yticks(mae_y_ticks)

plt.xscale('log',base=2) 

# 显示图形
plt.savefig('./weight_sweep.png')
plt.show()


