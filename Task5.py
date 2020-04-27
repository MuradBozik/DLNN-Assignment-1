import numpy as np
import matplotlib.pyplot as plt
lr = 0.8
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))
predicted_output = np.random.uniform(size=(4))
def sigmoid (x):
    return 1/(1 + np.exp(-x))
def dmse(x):
    return x * (1 - x)

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

def puremse(weights):
    global hidden_weights, hidden_bias, output_weights, output_bias
    hidden_layer_activation = np.dot(inputs,hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    error = (predicted_output - expected_output)
    # refer to the normal mse calculate the result
    answer = 0
    for each in error:
        answer += each**2
    return answer/4
#question 2
mses = []
def sigmoidmse(weights):
    global hidden_weights, hidden_bias, output_weights, output_bias,predicted_output,mses
    # add sigmoid to do the gradient
    hidden_layer_activation = np.dot(inputs,hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    #Backpropagation
    error = expected_output - predicted_output
    answer = 0
    for each in error:
        answer += each**2
    mses.append(answer)
    #print("Current MSE:{}".format(answer))

    return error * dmse(predicted_output), hidden_layer_output

# question 3
def grdmse(weights):
    global hidden_weights, hidden_bias, output_weights, output_bias
    #xor_net(0,0,weights)
    # use mse to calculate the mse of the output and the input nerual's output.
    d_predicted_output, hidden_layer_output = sigmoidmse(weights)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * dmse(hidden_layer_output)

    #Updating Weights and Biases with the grdmse
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

    ow = output_weights.reshape(2).tolist()
    ob = output_bias.reshape(1).tolist()
    hw = hidden_weights.reshape(4).tolist()
    hb = hidden_bias.reshape(2).tolist()
    new_weights = [hb[0]] + hw[0:2] + [hb[1]] + hw[2:4] + [ob[0]] + ow[0:2]
    return new_weights

# question 1
def xor_net(x1,x2,weights):
    global hidden_weights, hidden_bias, output_weights, output_bias
    hidden_weights = np.array(weights[1:3] + weights[4:6]).reshape(2,2)
    hidden_bias = np.array([weights[0]] + [weights[3]]).reshape(1,2)
    output_weights = np.array(weights[7:9]).reshape(2,1)
    output_bias = np.array([weights[6]]).reshape(1,1)
    '''
    print("Initial hidden weights: ",end='')
    print(*hidden_weights)
    print("Initial hidden biases: ",end='')
    print(*hidden_bias)
    print("Initial output weights: ",end='')
    print(*output_weights)
    print("Initial output biases: ",end='')
    print(*output_bias)
    '''

weights = np.random.uniform(size=9).tolist()
print(weights)
# finally:
turn = []
for i in range(10000):
    #print(weights)
    weights = grdmse(weights)
    turn.append(i)
print("the final weights is {}".format(weights))
final_answer = []
for predicted in predicted_output:
    if (predicted > 0.5):
        final_answer.append(1)
    else:
        final_answer.append(0)
print("the final answer is {}".format(final_answer))
plt.plot(turn,mses)
plt.xlabel('training steps')
plt.ylabel('current mse')
plt.show()

#ex3
for i in range(3):
    j = 0
    while(True):
        j = j + 1
        weights = np.random.uniform(size=9).tolist()
        weights = grdmse(weights)
        final_answer = []
        for predicted in predicted_output:
            if (predicted > 0.5):
                final_answer.append(1)
            else:
                final_answer.append(0)
        print(final_answer)
        if (final_answer[0] == 0 and final_answer[1] == 1 and final_answer[2] == 1 and final_answer[3] == 0):
            break
    print("turn spend: {}".format(j))
