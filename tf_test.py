import tensorflow as tf
import pickle
import numpy as np

if __name__ == "__main__":
    npzfile = np.load("data/3_rating_matrix.npz")

    print "load finishing"

    matrix1 = npzfile['arr_0']
    matrix_B = npzfile['arr_1']
    matrix2 = npzfile['arr_2']
    M1, N1 = matrix1.shape
    M2, N2 = matrix2.shape
    alpha=0.5
    beta=0.8
    K = 50
    graph = tf.Graph()
    with graph.as_default():

        U=tf.placeholder(tf.float32,[M1,N1])
        #V=tf.placeholder(tf.float32,[M2,N2])
        B=tf.placeholder(tf.float32,[M1,N1])

        X = tf.Variable(tf.random_uniform([M1,K],minval=0,maxval=1,dtype=tf.float32))
        Y = tf.Variable(tf.random_uniform([N1, K], minval=0, maxval=1, dtype=tf.float32))
        #Z = tf.Variable(tf.random_uniform([N2,K],minval=0,maxval=1,dtype=tf.float32))

        loss1 = 0.5 * tf.reduce_sum((tf.square(tf.multiply(B,tf.subtract(U,tf.matmul(X, tf.transpose(Y)))))))
        #loss2 = loss1+alpha * 0.5 * tf.reduce_sum(tf.square(tf.subtract(V,tf.matmul(X, tf.transpose(Z)))))
        loss = loss1+beta * 0.1 * (tf.reduce_sum(tf.square(X)) + tf.reduce_sum(tf.square(Y)))    #+tf.reduce_sum(tf.square(Z)))

        train_step = tf.train.AdagradOptimizer(2).minimize(loss)

    cost_history=[]
    num_loop=2000
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_loop):
            feed = {U: matrix1,B:matrix_B}
            _, cost = session.run([train_step, loss], feed_dict=feed)
            print("cost: %f" % cost)
            if step%10==0:
                X_array=X.eval(session)
                Y_array=Y.eval(session)
                #Z_array=Z.eval(session)
                np.savez(open("data/3_2_SVD_XYZ_tf.npz", "w"), X=X_array, Y=Y_array)
                print "save ok"
            #cost_history.append(session.run(loss, feed_dict=feed))


















