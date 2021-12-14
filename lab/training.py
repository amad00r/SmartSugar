import models

ds = models.Dataset()
pred = models.BloodSugarModel()

ds.import_dataset()

""" pred.train_blood_sugar(ds.X_train_blood_sugar, ds.y_train_blood_sugar, ds.X_test_blood_sugar, ds.y_test_blood_sugar)
pred.save_blood_sugar() """


pred.train_tendency(ds.X_train_tendency, ds.y_train_tendency, ds.X_test_tendency, ds.y_test_tendency)
pred.save_tendency()