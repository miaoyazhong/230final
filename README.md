csv_fixer.py: add quote around all columns to avoid the commas in title break <br>
to_folder.py: read cover path csv and copy the cover to the right category folder: viral or no viral<br>
data_split.py: label and split titles to train, dev, and test set, store them in csv for later use<br>
data_load.py: load data and covert the views to label(no_viral<=10,000, viral >= 60,000)<br>
image_model.py: train and test the cover prediction model<br>
text_model.py: train the title prediction model<br>
test.py: test the fusion model in test set<br>
plot.py: plot the accuracy, loss, and f1 score<br>
