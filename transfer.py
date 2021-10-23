# import torch
import paddle
from collections import OrderedDict
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
    # params=paddle.load("yolox_s.pdparams")
    params=paddle.load("yolox_new02.pdparams")
    for key, value in params.items():
        print(key,value.shape)
def show_value_paddle():
    params=paddle.load("yolox_s.pdparams")
    for key, value in params.items():
        print(key,value)
def switch_keys():#用于转换排查出不同名字的权重
    new_params=paddle.load("yolo.pdparams")
    params=paddle.load("yolox_s.pdparams")
    for k, v in params.items():
        if k.endswith(".running_mean"):
            new_params[k.replace(".running_mean","._mean")] = v.detach().numpy()
        elif k.endswith(".running_var"):
            new_params[k.replace(".running_var","._variance")] = v.detach().numpy()
            bn_w_name_list.append(k.replace(".running_var", ".weight"))
        else:
            new_params[k] = v.detach().numpy()
    paddle_model.set_dict(new_params)
#用于排查出不同名字的权重值转换
def switch_values():
    paddle_weight=paddle.load("yoloQAQ.pdparams")
    p_weight = paddle.load("yolox_s.pdparams")
    paddle_keys = []  # 存放paddle模型的权重键值
    p_keys = []  # 存放torch模型的权重键值
    for k in p_weight:  # 遍历torch模型权重键值
        p_keys.append(k)

    for k in paddle_weight:  # 遍历paddle模型权重键值
        paddle_keys.append(k)

    key_pair_length = min(len(p_keys), len(paddle_keys)) # 获取最小对应权重长度

# 将pytorch模型参数赋值给paddle模型
    for i, k in enumerate(paddle_keys):
        if i >= key_pair_length:
            break
        if paddle_weight[k].shape == p_weight[p_keys[i]].detach().numpy().shape: # 权重参数shape比较，只有一一对应才会赋值
            paddle_weight[k] = p_weight[p_keys[i]].detach().numpy()

    for k, p in p_weight.items():
        if k in paddle_weight:
            p_param = p_weight[k].detach().numpy()
            if p_param.shape == paddle_weight[k].shape:
                paddle_weight[k] = p_param
            else:
                print('torch param {} dose not match paddle param {}'.format(k, k))
        elif 'running_mean' in k:
            p_param = p_weight[k].detach().numpy()
            if p_param.shape == paddle_weight[k[:-12]+'_mean'].shape:
                paddle_weight[k[:-12]+'_mean'] = p_param
            else:
                print('torch param {} dose not match paddle param {}'.format(k, k[:-12]+'_mean'))
        elif 'running_var' in k:
            p_param = p_weight[k].detach().numpy()
            if p_param.shape == paddle_weight[k[:-11] + '_variance'].shape:
                paddle_weight[k[:-11] + '_variance'] = p_param
            else:
                print('torch param {} dose not match paddle param {}'.format(k, k[:-11] + '_variance'))
        else:
            print('torch param {} not exist in paddle modle'.format(k))

def show_value():
    # params1=paddle.load("yolox_s_new0.pdparams")
    params1=paddle.load("_coco_eval_log.pdparams")

    print("paddle")
    for key1, value1 in params1.items():
        if 'head' in key1:
            print(key1,value1.shape)
        # if '.reg_convs.5.bn._variance ' in key1:
        #     print(key1,value1)
        # if key1.endswith(".reg_convs.5.bn._variance "):
        #     print(key1,value1)
        # if key1.endswith(".reg_convs.5.bn._variance "):
        #     print(key1,value1)
            

def change_value():
    torch_keys = []  # 存放torch模型的权重键值
    paddle_keys = []  # 存放paddle模型的权重键值
    torch_weight = torch.load(input_fp,map_location=torch.device('cpu'))['model']
    paddle_weight=paddle.load("yolox_s.pdparams")
    for k in torch_weight.items():  # 遍历torch模型权重键值
        torch_keys.append(k)

    for k in paddle_weight.items():  # 遍历paddle模型权重键值
        paddle_keys.append(k)

    key_pair_length = min(len(torch_keys), len(paddle_keys)) # 获取最小对应权重长度

# 将pytorch模型参数赋值给paddle模型
    for i, k in enumerate(paddle_keys):
        if i >= key_pair_length:
            break
        if paddle_weight[k].shape == torch_weight[torch_keys[i]].detach().numpy().shape: # 权重参数shape比较，只有一一对应才会赋值
            paddle_weight[k] = torch_weight[torch_keys[i]].detach().numpy()

# 将paddle模型参数赋值给pytorch模型
    for i, k in enumerate(torch_keys):
        if i >= key_pair_length:
            break
        if torch_weight[k].detach().numpy().shape == paddle_weight[paddle_keys[i]].shape: # 权重参数shape比较，只有一一对应才会赋值
            torch_weight[k] = paddle_weight[paddle_keys[i]]
def show():
    paddle_weight=paddle.load("yoloQAQ.pdparams")
    for k, p in paddle_weight.items():
        if k in paddle_weight:
            param = paddle_weight[k].detach().numpy()
            if param.shape == paddle_weight[k].shape:
                paddle_weight[k] = p_param

    #head.reg_convs.5.bn._variance
if __name__ == "__main__":
    # transfer()
    # switch_values()
    # show_keys_paddle()
    # show_keys_torch()
    show_value()