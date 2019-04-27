
import torch
import os
import json

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
        return
    def forward(self, input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):
    model_path_prefix = './dtse_model/'
    possible_actions = 4

    lr = 0.00025

    def __init__(self, params):
        super(Model,self).__init__()
        # self.architecture = params["architecture"] if params["architecture"] else None
        self.can_cuda = params["cuda"] if params["cuda"] else False
        
        self.features,self.action = self.build_model()
        self.model = torch.nn.Sequential(self.features,self.action)
        self.global_step = 0
        self.filename = params["filename"] if params["filename"] else self.to_string()
        self.reward_index = params["reward_index"]
        return
    
    def forward(self,x):
        x= x.view(x.shape[0],-1)
        x = self.model(x)
        return x

    def light_actions(self,x):
        return torch.nn.argmax(self.forward(x))
    
    def get_global_step(self):
        return self.global_step

    def build_model(self):
        # Conv needs : features int, kernel_size int, pooling bool, actv_fn str
        # Fc needs: features int
        # self.features = torch.nn.Sequential(
        #     torch.nn.Conv2d(4,16,(3,7),stride=(1,1),padding=(1,0)),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(16,32,(3,7),stride=(1,1),padding=(1,0)),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(32,64,(3,5),stride=(1,1),padding=(1,0)),
        #     torch.nn.ReLU(),
        #     Flatten()
        # )

        # previous_output = self._get_conv_output([1,4,8,20])
        self.features = torch.nn.Sequential(
                        torch.nn.Linear(640,256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256,256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256,128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128,32),
                        torch.nn.ReLU()
        )
        # self.features = torch.nn.Sequential(self.features,self.fc_features)
    
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 4)
        )
        return self.features,self.classifier


    def _get_conv_output(self,shape):
        input_tensor = torch.rand(size=shape)
        output = self.features(input_tensor)
        return output.data.view(1,-1).size(1)

    def to_string(self):
        return str(self)
    
    def save_model(self):
        torch.save(self.model.state_dict(), "{}/{}/model_weights.pt".format(Model.model_path_prefix,self.filename))
        file = open("{}/{}/global_step.txt".format(Model.model_path_prefix,self.filename),"w")
        file.write(str(self.global_step))
        return

def load_model(filename):
    if(os.path.isfile("{}/{}/parameters.json".format(Model.model_path_prefix,filename))):
        model = Model(loadJson("{}/{}".format(Model.model_path_prefix,filename)))
        model.model.load_state_dict(torch.load("{}/{}/model_weights.pt".format(Model.model_path_prefix,model.filename)))
        file = open("{}/{}/global_step.txt".format(Model.model_path_prefix,model.filename),"r")
        model.global_step+= int(file.readline())
        return model
    else:
        sys.out.err("No parameter file exists in specified path. Could not load model from empty parameter.\nPath: {}".format(path))
        return None

    def get_global_step(self):
        return self.global_step.numpy()[0]

def loadJson(path):
    myjson = open("{}/parameters.json".format(path),'r')
    lines = "".join(myjson.readlines())
    myjson.close()
    json_vals = json.JSONDecoder().decode(lines)
    data = {}
    for key in json_vals.keys():
        data[key] = json_vals[key]
    return data

def saveJson(path,data):
    if(not os.path.isdir(path)):
        os.makedirs(path)
    myjson = open("{}/parameters.json".format(path),'w')
    json_vals = json.dumps(data,separators=(",",":"),sort_keys=True,indent=4)
    myjson.write(json_vals)
    myjson.close()

def setup_model(params):
    model = Model(params)
    optimizer = torch.optim.RMSprop(model.parameters(),lr=Model.lr)
    return model,optimizer





