from obj_classes import obj_det

ui_inp = "obj_det"
def mapping(user_inp):
    user_inp += "()"
    model = eval(user_inp)
    model.data_processing()
    model.train()
    model.test_images()
    model.test_video()

if __name__ == '__main__':
    mapping(ui_inp)
    