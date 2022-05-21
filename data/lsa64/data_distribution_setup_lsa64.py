import shutil
import os
import random

CHOSEN_TEST_PERSON = 10
CHOSEN_VAL_VID = random.randint(1, 5)

print("vybrané test video: " + str(CHOSEN_TEST_PERSON))
print("vybrané val video: " + str(CHOSEN_VAL_VID))

root_dir = ""
test_dir = 'lsa64_test'
train_dir = 'lsa64_train'
val_dir = 'lsa64_val'

file_names = os.listdir(root_dir+test_dir)
for file_name in file_names:
    shutil.move(os.path.join(root_dir+test_dir, file_name), root_dir+train_dir)
file_names = os.listdir(root_dir+val_dir)
for file_name in file_names:
    shutil.move(os.path.join(root_dir+val_dir, file_name), root_dir+train_dir)

people = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
videos = [1, 2, 3, 4, 5]
train_people = people.copy()
train_videos = videos.copy()
val_people = train_people
val_videos = [train_videos.pop(CHOSEN_VAL_VID - 1)]
test_people = [train_people.pop(CHOSEN_TEST_PERSON - 1)]
test_videos = videos
f_train = open("lsa64_ann_train_list.txt", "w")
f_val = open("lsa64_ann_val_list.txt", "w")
f_test = open("lsa64_ann_test_list.txt", "w")
for i in range(1, 65):
    for j in range(1, 11):
        for k in range(1, 6):
            file_name = str(i).zfill(3) + "_" + str(j).zfill(3) + "_" + str(k).zfill(3) + ".mp4 "
            if j in train_people and k in train_videos:
                f_train.write("lsa64_train/" + file_name + str(i - 1) + "\n")
            if j in val_people and k in val_videos:
                shutil.move(os.path.join(root_dir + train_dir, file_name), val_dir)
                f_val.write("lsa64_val/" + file_name + str(i - 1) + "\n")
            if j in test_people and k in test_videos:
                shutil.move(os.path.join(root_dir + train_dir, file_name), test_dir)
                f_test.write("lsa64_test/" + file_name + str(i - 1) + "\n")
f_train.close()
f_val.close()
f_test.close()