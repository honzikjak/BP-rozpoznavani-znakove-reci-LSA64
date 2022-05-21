import json
from matplotlib import pyplot as plt
import pickle

PATH = "../checkpoints/checkpoint_lsa64/sgd_def.json"
PICKLE_DUMP = False

f = open(PATH, "r")
f.readline()
loss_train = []
acc_train = []
acc_val = []
loss_val = []
t_train = []
t_val = []
t = 0
min_loss = 50
top_5_acc_last = 0
min_1_acc_val = 0
min_5_acc_val = 0
min_1_acc_train = 0
min_5_acc_train = 0
min_i = 0
for x in f:
    x = json.loads(x)
    if t == 0:
        t = x["epoch"]
    t += 20/768
    if x["mode"] == "train":
        loss_train.append(x["loss"])
        acc_train.append(x["top1_acc"])
        top_5_acc_last = x["top5_acc"]
        t_train.append(t)
    if x["mode"] == "val":
        if x["loss"] < min_loss and x["epoch"] <= 130:
            min_loss = x["loss"]
            min_i = x["epoch"]
            min_1_acc_val = x["top1_acc"]
            min_5_acc_val = x["top5_acc"]
            min_1_acc_train = acc_train[-1]
            min_5_acc_train = top_5_acc_last
        print("epoch: " + str(x["epoch"]) + ", loss: " + str(x["loss"]))
        loss_val.append(x["loss"])
        acc_val.append(x["top1_acc"])
        t_val.append(t)
print("\nbest val score:")
print("epoch: " + str(min_i) + ", top_1_acc: " + str(min_1_acc_val) + ", loss: " + str(min_loss))
print("accuracy:")
print("train: " + ", top_1_acc: " + str(min_1_acc_train) + ", top_5_acc: " + str(min_5_acc_train))
print("val: " + ", top_1_acc: " + str(min_1_acc_val) + ", top_5_acc: " + str(min_5_acc_val))
print("for tabular:")
print(str(min_i) + " & " + str(min_1_acc_train) + " & " + str(min_5_acc_train) + " & " + str(min_1_acc_val) + " & " + str(min_5_acc_val))
#plot_name = "adam lr=0.0001, val vid = 2, test person = 2"
plot_name = "Adam"
plt.figure(PATH[32:-5] + " loss")
plt.plot(t_train, loss_train, t_val, loss_val)
plt.legend(["train", "val"])
plt.ylabel("ztráta")
plt.xlabel("epocha")
plt.ylim(-0.2, 4.5)
plt.title("Průběh ztráty")
plt.show()

plt.figure(PATH[32:-5] + " acc")
plt.plot(t_train, acc_train, t_val, acc_val)
plt.legend(["train", "val"])
plt.ylabel("přesnost")
plt.xlabel("epocha")
plt.title("Průběh přesnosti")
plt.show()

if PICKLE_DUMP:
    pickle.dump(loss_val, open("./analysis/validation_losses/" + PATH[32:-9] + ".p", "wb"))
    pickle.dump(acc_val, open("./analysis/validation_acc/" + PATH[32:-9] + ".p", "wb"))