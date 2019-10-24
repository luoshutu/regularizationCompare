# 使用IMDB数据集比较几种不同正则化方法所得结果

### 评论分类数据集导入
  from keras.datasets import imdb
  import numpy as np

  (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

### 输入数据标准化
  def vectorize_sequences(sequences, dimension = 10000):
      results = np.zeros((len(sequences), dimension))
      for i, sequence in enumerate(sequences):
          results[i, sequence] = 1.
      return results

  x_train = vectorize_sequences(train_data)
  x_test  = vectorize_sequences(test_data)

  y_train = np.asarray(train_labels).astype('float32')
  y_test  = np.asarray(test_labels).astype('float32')

### original模型建立
  from keras import models
  from keras import layers

  original_model = models.Sequential()
  original_model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
  original_model.add(layers.Dense(16, activation = 'relu'))
  original_model.add(layers.Dense(1, activation = 'sigmoid'))

  original_model.compile(optimizer = 'rmsprop',
                         loss = 'binary_crossentropy',
                         metrics = ['acc'])

### 模型训练并在测试集上验证
  original_history = original_model.fit(x_train, y_train,
                                        epochs = 20,
                                        batch_size = 512,
                                        validation_data = (x_test, y_test))

### L1正则化模型建立
  from keras import regularizers

  L1_model = models.Sequential()
  L1_model.add(layers.Dense(16, kernel_regularizer = regularizers.l1(0.001),
                            activation = 'relu', input_shape = (10000,)))
  L1_model.add(layers.Dense(16, kernel_regularizer = regularizers.l1(0.001),
                            activation = 'relu'))
  L1_model.add(layers.Dense(1, activation = 'sigmoid'))

  L1_model.compile(optimizer = 'rmsprop',
                   loss = 'binary_crossentropy',
                   metrics = ['acc'])

### 模型训练并在测试集上验证
  L1_history = L1_model.fit(x_train, y_train,
                            epochs = 20,
                            batch_size = 512,
                            validation_data = (x_test, y_test))

### L2正则化模型建立
  L2_model = models.Sequential()
  L2_model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001),
                            activation = 'relu', input_shape = (10000,)))
  L2_model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001),
                            activation = 'relu'))
  L2_model.add(layers.Dense(1, activation = 'sigmoid'))

  L2_model.compile(optimizer = 'rmsprop',
                   loss = 'binary_crossentropy',
                   metrics = ['acc'])

### 模型训练并在测试集上验证
  L2_history = L2_model.fit(x_train, y_train,
                            epochs = 20,
                            batch_size = 512,
                            validation_data = (x_test, y_test))

### L1&L2正则化模型
  L1_L2_model = models.Sequential()
  L1_L2_model.add(layers.Dense(16,
                         kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 = 0.001),
                         activation = 'relu', input_shape = (10000,)))
  L1_L2_model.add(layers.Dense(16,
                         kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 = 0.001),
                         activation = 'relu'))
  L1_L2_model.add(layers.Dense(1, activation = 'sigmoid'))

  L1_L2_model.compile(optimizer = 'rmsprop',
                      loss = 'binary_crossentropy',
                      metrics = ['acc'])

### 模型训练并在测试集上验证
  L1_L2_history = L1_L2_model.fit(x_train, y_train,
                            epochs = 20,
                            batch_size = 512,
                            validation_data = (x_test, y_test))

### dropout正则化模型
  dropout_model = models.Sequential()
  dropout_model.add(layers.Dense(16,
                                 activation = 'relu',
                                 input_shape = (10000,)))
  dropout_model.add(layers.Dropout(0.5))
  dropout_model.add(layers.Dense(16,activation = 'relu'))
  dropout_model.add(layers.Dropout(0.5))
  dropout_model.add(layers.Dense(1, activation = 'sigmoid'))

  dropout_model.compile(optimizer = 'rmsprop',
                   loss = 'binary_crossentropy',
                   metrics = ['acc'])

### 模型训练并在测试集上验证
  dropout_history = dropout_model.fit(x_train, y_train,
                            epochs = 20,
                            batch_size = 512,
                            validation_data = (x_test, y_test))

### 画图分析
  import matplotlib.pyplot as plt

  epochs = 20
  x_axis = range(1, epochs + 1)

  #验证数据上的损失曲线
  original_loss = original_history.history['val_loss']
  L1_loss = L1_history.history['val_loss']
  L2_loss = L2_history.history['val_loss']
  L1_L2_loss = L1_L2_history.history['val_loss']
  dropout_loss = dropout_history.history['val_loss']

  plt.figure()
  plt.plot(x_axis, original_loss, '+', label = 'Original model')
  plt.plot(x_axis, L1_loss, 'b', label = 'L1 model')
  plt.plot(x_axis, L2_loss, 'r', label = 'L2 model')
  plt.plot(x_axis, L1_L2_loss, 'g', label = 'L1_L2 model')
  plt.plot(x_axis, dropout_loss, 'y', label = 'Dropout model')
  plt.title('Loss Compare')
  plt.xlabel('Epochs')
  plt.ylabel('Validation Loss')
  plt.legend()

  #验证数据上的精度曲线
  original_acc = original_history.history['val_acc']
  L1_acc = L1_history.history['val_acc']
  L2_acc = L2_history.history['val_acc']
  L1_L2_acc = L1_L2_history.history['val_acc']
  dropout_acc = dropout_history.history['val_acc']

  plt.figure()
  plt.plot(x_axis, original_acc, '+', label = 'Original model')
  plt.plot(x_axis, L1_acc, 'b', label = 'L1 model')
  plt.plot(x_axis, L2_acc, 'r', label = 'L2 model')
  plt.plot(x_axis, L1_L2_acc, 'g', label = 'L1_L2 model')
  plt.plot(x_axis, dropout_acc, 'y', label = 'Dropout model')
  plt.title('acc Compare')
  plt.xlabel('Epochs')
  plt.ylabel('Validation acc')
  plt.legend()


  #原始模型的损失以及精度比较
  origin_train_loss = original_history.history['loss']
  origin_trian_acc  = original_history.history['acc']

  plt.figure()
  plt.plot(x_axis, origin_train_loss, 'bo', label = 'Train Loss')
  plt.plot(x_axis, original_loss, 'ro', label = 'Validation Loss')
  plt.title('Original Model Loss Compare')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.figure()
  plt.plot(x_axis, origin_trian_acc, 'bo', label = 'Train Acc')
  plt.plot(x_axis, original_acc, 'ro', label = 'Validation Acc')
  plt.title('Original Model Acc Compare')
  plt.xlabel('Epochs')
  plt.ylabel('Acc')
  plt.legend()

  plt.show()

