if __name__ == '__main__':
    from model_ASPP import *
    m = create_model(2, True)
    m = model_load_weights("/home/jayda960825/ManTraNet_2020/pretrained_weights/ManTraNet_Ptrain4.h5", m)
    m = m.cuda()
    train = torch.rand(8,3,256,256).cuda()
    m(train)
    m.eval()
    for i in range(4):
        val = torch.rand(16,3,256,256).cuda()
        with torch.no_grad():
            m(val)