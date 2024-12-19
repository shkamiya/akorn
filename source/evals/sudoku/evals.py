
import torch

def compute_board_accuracy(pred, Y, is_input):
    #print(pred.shape)
    B = pred.shape[0]
    pred = pred.reshape((B, -1, 9)).argmax(-1)
    Y = Y.argmax(dim=-1).reshape(B, -1)
    mask = 1 - is_input.reshape(B, -1) 

    num_blanks = mask.sum(1)
    num_correct = (mask * (pred == Y)).sum(1) 
    board_correct =  (num_correct == num_blanks).int()  
    return num_blanks, num_correct, board_correct
    
    
