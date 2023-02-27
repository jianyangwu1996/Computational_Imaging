import numpy as np
import matplotlib.pyplot as plt

beta = [300, 400, 500, 600]
color = ['r', 'g', 'b', 'c', 'm', 'y']

for b, c in zip(beta, color):
    # mpe = np.load('./result_store/mpe/mpe%i.npy'%b)
    # print(b, np.min(mpe))
    cmpe = np.load('./result_store/mpe/mpe%ic.npy'%b)
    print('%ic'%b, np.min(cmpe))
    # plt.plot(mpe, linestyle='-', label=b, color=c)
    plt.plot(cmpe, linestyle='--', label='%ic'%b, color=c)
plt.legend()

# fig, axes = plt.subplots(2, 7, figsize=(21, 6))
# for b, i in zip(beta, range(7)):
#     obj = np.load('./result_store/obj/obj%i.npy'%b)
#     cobj = np.load('./result_store/obj/obj%ic.npy'%b)
#     axes[0, i].imshow(np.angle(obj), cmap='gray')
#     axes[1, i].imshow(np.angle(cobj), cmap='gray')

plt.show()
