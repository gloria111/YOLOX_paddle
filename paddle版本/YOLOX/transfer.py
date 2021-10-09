def transfer():
    input_fp = "model.pth"
    output_fp = "model.pdparmas"
    torch_dict = torch.load(input_fp)['model']
    paddle_dict ={}
    fc_names = []
    for key in torch_dict:
        weight = torch_dict[key].cpu().numoy()
        flag = [i in key for i in fc_names]
        if any(flag):
            print("weight{}need to be tansferd".format(key))
            weight = weight.transpose()
        paddle_dict[key] = weight
    paddle.save(paddle_dict,output_fp)