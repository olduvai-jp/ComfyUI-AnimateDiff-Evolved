from huggingface_hub import hf_hub_download

class MotionLoraInfo:
    def __init__(self, name: str, strength: float = 1.0, hash: str=""):
        self.name = name
        self.strength = strength
        self.hash = ""
    
    def set_hash(self, hash: str):
        self.hash = hash
    
    def clone(self):
        return MotionLoraInfo(self.name, self.strength, self.hash)

class MotionLoraInfoHf(MotionLoraInfo):
    def __init__(self, repo_name, filename, api_token, strength: float = 1.0, hash: str=""):
        name = f"{repo_name}/{filename}"
        super().__init__(name, strength, hash)
        self.repo_name = repo_name
        self.filename = filename
        self.api_token = api_token
    
    def clone(self):
        return MotionLoraInfoHf(self.name, self.strength, self.hash)
    
    def load_lora(self):
        lora_path = hf_hub_download(repo_id=self.repo_name, filename=self.filename, token=self.api_token)
        return lora_path

class MotionLoraList:
    def __init__(self):
        self.loras: list[MotionLoraInfo] = []
    
    def add_lora(self, lora: MotionLoraInfo):
        self.loras.append(lora)
    
    def clone(self):
        new_list = MotionLoraList()
        for lora in self.loras:
            new_list.add_lora(lora.clone())
        return new_list
