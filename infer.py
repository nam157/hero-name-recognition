import glob
import os

import torch
from PIL import Image

from train import DenseNet121, test_transforms

classes = ['Ahri', 'Akali', 'Alistar', 'Amumu', 'Annie', 'Ashe', 'Blitzcrank', 'Camille', 'Corki', 'Darius', 'Diana',
            'Dr._Mundo', 'Evelynn', 'Ezreal', 'Fiora', 'Fizz', 'Garen', 'Gragas', 'Graves', 'Janna', 'Jarvan_IV', 'Jax', 'Jhin',
              'Jinx', 'KaiSa', 'Katarina', 'Kennen', 'Lee_Sin', 'Leona', 'Lulu', 'Lux', 'Master_Yi', 'Miss_Fortune']

def convert_model(checkpoint):
    print("--Convert model start--")
    model = DenseNet121(classCount=33,isTrained=True)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, './save_model/model_hero_jit.pt')


def test(file_convert_model,folder_img,save_file):
    print("--Load model TorchScript--")
    model = torch.jit.load(file_convert_model)
    names = []
    targets = []
    print("--Inference--")
    with torch.no_grad():
        for path_img in glob.glob(folder_img + "/*.jpg"):
            img = Image.open(path_img).convert('RGB')
            img = test_transforms(img)
            img = img[None,:,:]
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            names.append(os.path.basename(path_img))
            targets.append(classes[predicted])
    print("--save results--")
    with open(save_file,'w') as file:
        for name,label in zip(names,targets):
            file.writelines(str(name) + "\t" + str(label))
            file.writelines("\n")

if __name__ == "__main__":
    convert_model('hero_model.pth')
    test(checkpoint = 'model_hero_jit',folder_img='./test_data/test_images/',save_file='./test.txt')

