import numpy as np
import matplotlib.pyplot as plt

beta = [300, 400, 500, 600]
# color = ['r', 'g', 'b', 'c', 'm', 'y']
#
# for b, c in zip(beta, color):
#     nrmse = np.load('./result_store/nrmse/nrmse%i.npy'%b)
#     print(b, np.min(nrmse))
#     cnrmse = np.load('./result_store/nrmse/nrmse%ic.npy'%b)
#     print('%ic'%b, np.min(cnrmse))
#     plt.plot(nrmse, linestyle='-', label=b, color=c)
#     plt.plot(cnrmse, linestyle='--', label='%ic'%b, color=c)
# plt.legend()

fig, axes = plt.subplots(2, 4, figsize=(21, 6))
for b, i in zip(beta, range(7)):
    obj = np.load('./result_store/obj/obj%i.npy'%b)
    cobj = np.load('./result_store/obj/obj%ic.npy'%b)
    axes[0, i].imshow(np.angle(obj), cmap='rainbow')
    axes[0, i].set_title(b)
    axes[1, i].imshow(np.angle(cobj), cmap='rainbow')
    axes[1, i].set_title('%ic'%b)
for ax in axes.flat:
    ax.axis('off')
plt.show()
