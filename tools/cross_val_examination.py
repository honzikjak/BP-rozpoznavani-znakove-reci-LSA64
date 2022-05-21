import pickle
import copy
from matplotlib import pyplot as plt

def get_max_min_mean(file, path, epochs):
    min = pickle.load(open(path + file + str(1) + ".p", "rb"))
    min = min[0:epochs]
    max= copy.copy(min)
    mean = [0] * len(min)
    for i in range(2, 11):
        cross_val = pickle.load(open(path + file + str(i) + ".p", "rb"))
        for j in range(epochs):
            mean[j] = mean[j] + 1/(i-2+1)*(cross_val[j] - mean[j])
            if cross_val[j] < min[j]:
                min[j] = cross_val[j]
            if cross_val[j] > max[j]:
                max[j] = cross_val[j]
    return max, min, mean

file = "sgd_lr_0.001_te"
path = "./analysis/validation_losses/"
epochs = 25
t = list(range(96, epochs+96))

max_val, min_val, mean_val = get_max_min_mean(file, path, epochs)
plt.plot(t, mean_val, 'C1', t, min_val, 'C3--', t, max_val, 'C3--')
plt.legend(["průměr", "meze"])
plt.title("Průběh ztráty")
plt.ylabel("ztráta")
plt.xlabel("epocha")
plt.show()

path = "./analysis/validation_acc/"
max_val, min_val, mean_val = get_max_min_mean(file, path, epochs)
plt.plot(t, mean_val, 'C1', t, min_val, 'C3--', t, max_val, 'C3--')
plt.legend(["průměr", "meze"])
plt.title("Průběh přesnosti")
plt.ylabel("přesnost")
plt.xlabel("epocha")
plt.show()


file = "adam_lr_1e-5_te"
path = "./analysis/validation_losses/"
epochs = 35
t = list(range(96, epochs+96))
max_val, min_val, mean_val = get_max_min_mean(file, path, epochs)
plt.plot(t, mean_val, 'C1', t, min_val, 'C3--', t, max_val, 'C3--')
plt.legend(["průměr", "meze"])
plt.title("Průběh ztráty")
plt.ylabel("ztráta")
plt.xlabel("epocha")
plt.show()

path = "./analysis/validation_acc/"
max_val, min_val, mean_val = get_max_min_mean(file, path, epochs)
plt.plot(t, mean_val, 'C1', t, min_val, 'C3--', t, max_val, 'C3--')
plt.legend(["průměr", "meze"])
plt.title("Průběh přesnosti")
plt.ylabel("přesnost")
plt.xlabel("epocha")
plt.show()