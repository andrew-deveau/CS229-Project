from keras.models import load_model 
import glob
import convnet as c

X_dev, y_dev, cat_y_dev, _ = c.load_data(300, languages = ['english', 'mandarin'], which = 'dev')

accr = []
for f in glob.glob("saved_models/*"):
     cur_model = load_model(f)
     accr.append(f + "  " + str(cur_model.evaluate(X_dev, cat_y_dev, verbose = 0)[1]) + "\n")


with open("accr.txt", "wt") as f:
    f.write("\n".join(accr))

