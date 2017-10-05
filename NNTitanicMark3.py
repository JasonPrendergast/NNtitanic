#from create_sentiment_featuresets import create_feature_sets_and_labels
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
import numpy as np
# pickle.load(open("sentiment_set.pickle","rb"))
#train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
train_x,train_y,test_x,test_y = pickle.load(open("titanic_set.pickle","rb"))

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


# Nothing changes
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output


def train_neural_network(x):
	prediction = neural_network_model(x)
	cost =tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	#tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		#sess.run(tf.initialize_all_variables())
		sess.run(tf.global_variables_initializer())
	    	try:
                    epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
                    
                    
                    print('STARTING:',epoch)
                except:
                    epoch = 1
                while epoch <= hm_epochs:
                    if epoch != 1:
                        saver.restore(sess,"./model.ckpt")
                    epoch_loss = 1
                    i=0
		    while i < len(train_x):
                        start = i
			end = i+batch_size
			batch_x = np.array(train_x[start:end])
			batch_y = np.array(train_y[start:end])
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
			epoch_loss += c
			i+=batch_size
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		saver.save(sess, "./model.ckpt")
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		with open(tf_log,'a') as f:
                    f.write(str(epoch)+'\n') 
                epoch +=1

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

saver = tf.train.import_meta_graph('./model.ckpt.meta')

def use_neural_network(input_data):
    prediction = neural_network_model(x)
   # with open('lexicon-2500-2638.pickle','rb') as f:
   #     lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        #tf.reset_default_graph()
        sess.run(tf.global_variables_initializer())
        #tf.reset_default_graph()
        saver.restore(sess,"model.ckpt")
        #print('model restored')
       # current_words = word_tokenize(input_data.lower())
       # current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = #np.zeros(len(lexicon))

       # for word in current_words:
       #    if word.lower() in lexicon:
       #         index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
       #         features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        if result[0] == 0:
          #  print('Positive:',input_data)
           return 'Positive:',input_data
        elif result[0] == 1:
            #Print('Neg')
          #  print('Negative:',input_data)
            return 'Negative:',input_data
            
	    
train_neural_network(x)
