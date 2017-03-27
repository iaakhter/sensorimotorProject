import model_example
import tensorflow as tf
import processImages

#globals
epochs_completed = 0
index_in_epoch = 0

# Load Training Data
imagePath ="trainingData/trainingFeature.txt"
labelPath = "trainingData/trainingLabel.txt"
numOfExamples = 2000

# data = Data()
xTrain = processImages.constructXFromTargetFocusLocations(numOfExamples, imagePath)
yTrain = processImages.convertLabelToArray(numOfExamples, labelPath)

# Start session
sess = tf.InteractiveSession()

# Learning Functions
L2NormConst = 0.001
train_vars = tf.trainable_variables()
loss = tf.reduce_mean(tf.square(tf.subtract(model_example.y_, model_example.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess.run(tf.global_variables_initializer())

# Training loop variables
epochs = 100
batch_size = 50
num_samples = numOfExamples
step_size = int(num_samples / batch_size)

for epoch in range(epochs):
    for i in range(step_size):
    	start = index_in_epoch
    	index_in_epoch += batch_size
    	if index_in_epoch >= numOfExamples:
    		#Finished epoch
    		epochs_completed += 1
    		#Start new epoch
    		start = 0
    		index_in_epoch = batch_size
    	end = index_in_epoch
        batch = xTrain[start:end], yTrain[start:end]

        train_step.run(feed_dict={model_example.x: batch[0], model_example.y_: batch[1], model_example.keep_prob: 0.8})

        if i%10 == 0:
          loss_value = loss.eval(feed_dict={model_example.x:batch[0], model_example.y_: batch[1], model_example.keep_prob: 1.0})
          print("epoch: %d step: %d loss: %g"%(epoch, epoch * batch_size + i, loss_value))

# Save the Model
saver = tf.train.Saver()
saver.save(sess, "model.ckpt")
