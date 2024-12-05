# Model Training Output
So far the best performance I think I have is with `models/img_model-transfer.c.h5` which is a transfer learning models based off
of *MobileNetV2*. 294 images were used during training (plus 10 augmented/generated images) giving a total of 2940 images (80/20 
training vs validation split)

[TRAINING] <_BatchDataset element_spec=(TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 3), dtype=tf.float32, name=None))>
[TRAINING] Building the model (MobileNetV2 base)...
[TRAINING] Compiling the model...
[TRAINING] Starting training...
Epoch 1/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 9s 88ms/step - accuracy: 0.5562 - binary_accuracy: 0.7737 - loss: 0.4882 - val_accuracy: 0.6584 - val_binary_accuracy: 0.9263 - val_loss: 0.2161
Epoch 2/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 8s 86ms/step - accuracy: 0.6434 - binary_accuracy: 0.9146 - loss: 0.2278 - val_accuracy: 0.7002 - val_binary_accuracy: 0.9572 - val_loss: 0.1376
Epoch 3/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 8s 86ms/step - accuracy: 0.6660 - binary_accuracy: 0.9446 - loss: 0.1539 - val_accuracy: 0.6584 - val_binary_accuracy: 0.9706 - val_loss: 0.1030
Epoch 4/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 8s 91ms/step - accuracy: 0.6637 - binary_accuracy: 0.9578 - loss: 0.1182 - val_accuracy: 0.6553 - val_binary_accuracy: 0.9748 - val_loss: 0.0862
Epoch 5/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 8s 94ms/step - accuracy: 0.6791 - binary_accuracy: 0.9699 - loss: 0.0948 - val_accuracy: 0.6955 - val_binary_accuracy: 0.9789 - val_loss: 0.0726
Epoch 6/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 8s 87ms/step - accuracy: 0.6924 - binary_accuracy: 0.9745 - loss: 0.0819 - val_accuracy: 0.6677 - val_binary_accuracy: 0.9825 - val_loss: 0.0625
Epoch 7/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 8s 86ms/step - accuracy: 0.6742 - binary_accuracy: 0.9781 - loss: 0.0717 - val_accuracy: 0.7141 - val_binary_accuracy: 0.9815 - val_loss: 0.0589
Epoch 8/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 8s 86ms/step - accuracy: 0.6850 - binary_accuracy: 0.9854 - loss: 0.0576 - val_accuracy: 0.6971 - val_binary_accuracy: 0.9845 - val_loss: 0.0520
Epoch 9/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 7s 82ms/step - accuracy: 0.6977 - binary_accuracy: 0.9837 - loss: 0.0520 - val_accuracy: 0.6491 - val_binary_accuracy: 0.9861 - val_loss: 0.0479
Epoch 10/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 8s 86ms/step - accuracy: 0.6468 - binary_accuracy: 0.9893 - loss: 0.0454 - val_accuracy: 0.6723 - val_binary_accuracy: 0.9856 - val_loss: 0.0429
[TRAINING] Fine-tine training...
Epoch 1/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 30s 286ms/step - accuracy: 0.7046 - binary_accuracy: 0.8428 - loss: 0.3627 - val_accuracy: 0.6600 - val_binary_accuracy: 0.9763 - val_loss: 0.0582
Epoch 2/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 22s 263ms/step - accuracy: 0.7082 - binary_accuracy: 0.9423 - loss: 0.1553 - val_accuracy: 0.6445 - val_binary_accuracy: 0.9701 - val_loss: 0.0705
Epoch 3/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 23s 270ms/step - accuracy: 0.7006 - binary_accuracy: 0.9673 - loss: 0.1002 - val_accuracy: 0.6321 - val_binary_accuracy: 0.9686 - val_loss: 0.0730
Epoch 4/10
81/81 ━━━━━━━━━━━━━━━━━━━━ 23s 270ms/step - accuracy: 0.7035 - binary_accuracy: 0.9727 - loss: 0.0769 - val_accuracy: 0.6337 - val_binary_accuracy: 0.9717 - val_loss: 0.0675
[TRAINING] Saving the model to models/img_model-transfer.c.h5...