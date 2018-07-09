
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[4]:


hello=tf.constant("hello tensorflow")


sess=tf.Session()

print(sess.run(hello))


# In[6]:


node1=tf.constant(3.0,tf.float32)
node2=tf.constant(4.0)
node3=tf.add(node1,node2)

sess=tf.Session()
print(sess.run(node3))


# In[12]:


a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder_node=a+b

print(sess.run(adder_node,feed_dict={a:3,b:3}))


# In[18]:


x_train=[1,2,3]
y_train=[1,2,3]

W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=x_train*W+b

cost=tf.reduce_mean(tf.square(hypothesis-y_train))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

sess=tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))

