import tensorflow

new_model = tensorflow.keras.models.load_model('pinCoordinatesConvertion.h5')
print(new_model.predict(272,657))