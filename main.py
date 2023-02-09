# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def loss_ex1(data):
    loss = 2 * data ** 2 - 11 * data + 1
    loss_grad = 4 * data - 11
    return [loss, loss_grad]
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    import torch

    # import chap3/three_min_lib.py as three
    #
    # x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    # three.three_min.print_tensor(x)

    #y = x**2 -2x + 3

    x = 10 #torch.tensor(10.0, requires_grad=True)
    l_r = 1
    for i in range(100) :
        [loss, loss_grad] = loss_ex1(x)
        print(x, loss, loss_grad)
        if abs(loss_grad) < 0.01 :
            print("Answer : ",x)
            break
        else :
            if(loss_grad>0):
                x = x - l_r
            else :
                x = x + l_r






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
