import matplotlib.pyplot as plt
import numpy as np

x = ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '8', '8', '8', '8', '8', '8', '8', '8', '16', '16', '16', '16', '32', '32', '64']
y = [0.60126, 0.84982, 0.85388, 0.84836, 0.7584, 0.563, 0.78868, 0.68606, 0.82702, 0.76676, 0.18076, 0.13502, 0.6634, 0.81138, 0.16018, 0.83458, 0.7596, 0.79558, 0.84236, 0.7981, 0.61662, 0.75688, 0.8377, 0.16094, 0.81096, 0.83222, 0.74092, 0.82002, 0.80992, 0.79434, 0.63588, 0.81326, 0.64776, 0.81238, 0.83998, 0.83672, 0.79866, 0.27512, 0.17942, 0.8213, 0.5438, 0.83014, 0.8382, 0.8337, 0.79242, 0.81138, 0.7932, 0.78662, 0.84522, 0.69966, 0.70468, 0.83208, 0.79078, 0.83962, 0.84638, 0.70668, 0.84024, 0.63732, 0.79726, 0.69594, 0.80534, 0.77514, 0.70058, 0.83276, 0.84982, 0.8526656, 0.7584, 0.78868, 0.82702, 0.18076, 0.81138, 0.83458, 0.79543608, 0.84236, 0.75688, 0.8377, 0.83192236, 0.82002, 0.8052771599999999, 0.81326, 0.81238, 0.8384152, 0.79866, 0.8213, 0.83014, 0.8362604999999999, 0.80980632, 0.7925091000000001, 0.84522, 0.83208, 0.83962, 0.84638, 0.84024, 0.79726, 0.8052796, 0.83276, 0.8521113600000001, 0.78855888, 0.82702, 0.8329792, 0.8422714800000001, 0.8377, 0.83094322, 0.8126433200000001, 0.8376356599999999, 0.8139872800000001, 0.83589504, 0.8089984599999999, 0.84146196, 0.84372332, 0.8401110599999999, 0.8320470799999999, 0.85208034, 0.8311477999999999, 0.8396385599999999, 0.8296125799999999, 0.8376235, 0.83567506, 0.8434387200000001, 0.8380159599999999, 0.8521947, 0.8375525800000001, 0.8368118, 0.84308076, 0.8509703, 0.84176842, 0.8496634600000001]
conf_max = 0.8602235620634561
conf_min = 0.8391033579365442

plt.scatter(x,y,marker="+",c="k",linewidth=1)
ax = plt.gca()
ax.axhline(y=conf_min, linewidth=1, c="k")
ax.axhline(y=conf_max, linewidth=1, c="k")
plt.ylim((0.25,1.0))
plt.xlabel('experiment size (# trials)')
plt.ylabel('accuracy')
plt.title('MNIST rotated REEC')
plt.savefig('rot', dpi=300)