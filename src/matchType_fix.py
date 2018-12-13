file_train_path = "../data/train/train.csv"
file_test_path = "../data/test/test.csv"

file_train_path_new = "../data/train/train_new.csv"
file_test_path_new = "../data/test/test_new.csv"

match_categories = ((",solo,", ",1,"),
                    (",duo,", ",2,"),
                    (",squad,", ",3,"),
                    (",solo-fpp,", ",4,"),
                    (",duo-fpp,", ",5,"),
                    (",squad-fpp,", ",6,"),
                    (",normal-solo,", ",7,"),
                    (",normal-duo,", ",8,"),
                    (",normal-squad,", ",9,"),
                    (",normal-solo-fpp,", ",10,"),
                    (",normal-duo-fpp,", ",11,"),
                    (",normal-squad-fpp,", ",12,"),
                    (",crashfpp,", ",13,"),
                    (",flaretpp,", ",14,"),
                    (",flarefpp,", ",15,"),
                    (",crashtpp,", ",16,"))

print("Indexing training set")
with open(file_train_path) as f_in:
    with open(file_train_path_new, "w") as f_out:
        for line in f_in:
            for pair in match_categories:
                line = line.replace(*pair)
            f_out.write(line)

f_in.close()
f_out.close()
print("Done with training set")

print("Indexing testing set")
with open(file_test_path) as f_in:
    with open(file_test_path_new, "w") as f_out:
        for line in f_in:
            for pair in match_categories:
                line = line.replace(*pair)
            f_out.write(line)

f_in.close()
f_out.close()
print("Done with testing set")
print("You may now run root.py")
