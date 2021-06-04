from keras.preprocessing.image import ImageDataGenerator #Generates batches of tensor image data with augmentation

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/nonaugmented_dataset/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 classes = ('normal','covid'),
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/nonaugmented_dataset/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            classes = ('normal','covid'),
                                            class_mode = 'categorical')

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 1000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 1000)'''