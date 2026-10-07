#%%
import sphere_ref_lib as srl
import numpy as np
import pandas as pd
import ultraplot as uplt

#%%
fp = "./output_focus/"

R2s = [1.001, 1.2, 2.0, 5.0]
ths = np.arange(0, 91, 1)

rfd_power, ald = srl.fd.square_func(R2s, ths)

a2a1 = srl.fd.square_func_rw2015(R2s, ths, ald)

#%%
rfd_power

#%%
a2a1

#%%
uplt.rc.update(fontsize=16)

ymax = 1.0
ymin = 1e-2

plot_colors = ["red", "darkorange", "springgreen", "mediumblue"]
cycle = uplt.Cycle(colors=plot_colors)

fig, ax = uplt.subplots(suptitle="Reflected waves power (from radar equation)", xlabel="s/c angle alpha (degree)", ylabel="Ratio to the incident wave", figsize=(10,7))
ax.format(ylim=(ymin,ymax), xlim=(0,180), ylocator=ymax/10, yscale="log")
ax.plot(ald, rfd_power, cycle=cycle, label=[f"{h}" for h in R2s], ls="--", legend="b", legend_kw={"title": "height"}, linewidth=2)
ax.plot(ald, a2a1, cycle=cycle, label=[f"{h}" for h in R2s], legend="b", legend_kw={"lw": 8, "ncols": 4, "title": "height"}, linewidth=2)

fig.savefig(fp + "Result_compared_rw2015_log.png")
fig.show()

uplt.rc.reset()

#%%
ald