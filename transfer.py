import torch
import paddle
def transfer():#transfer linear
    input_fp = "yolox_s.pth"
    output_fp = "yolox_s.pdparams"
    torch_dict = torch.load(input_fp,map_location=torch.device('cpu'))['model']
    paddle_dict ={}
    fc_names = []
    for key in torch_dict:
        weight = torch_dict[key].cpu().numpy()
        flag = [i in key for i in fc_names]
        if any(flag):
            print("weight{}need to be tansferd".format(key))
            weight = weight.transpose()
        paddle_dict[key] = weight
    paddle.save(paddle_dict,output_fp)
def show_keys_torch():
    params = torch.load("yolox_s.pth", map_location=torch.device('cpu'))['model']
    for key, value in params.items():
        print(key,value.shape)
def show_keys_paddle():
    params=paddle.load("yolox_s.pdparams")
    for key, value in params.items():
        print(key,value.shape)
def show_value():
    params1=paddle.load("yolox_s.pdparams")
    print("paddle")
    for key1, value1 in params1.items():
        print(key1,value1)
    print("YOLOX")
    params2 = torch.load("yolox_s.pth", map_location=torch.device('cpu'))['model']
    for key2, value2 in params2.items():
        print(key2,value2)

# def switch_keys():#用于转换排查出不同名字的权重
    # for k, v in params.items():
        # if k.endswith(".running_mean"):
        #     new_params[k.replace(".running_mean","._mean")] = v.detach().numpy()
        # elif k.endswith(".running_var"):
        #     new_params[k.replace(".running_var","._variance")] = v.detach().numpy()
        #     bn_w_name_list.append(k.replace(".running_var", ".weight"))
        # else:
        #     new_params[k] = v.detach().numpy()

 
if __name__ == "__main__":
    # transfer()
    # show_keys_torch()
    show_value()

